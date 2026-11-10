"""Reproducible single-cycle experiment with lazy cached data and matched controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from offgrid_da.data.arches_training import LazyArchesDataset, atomic_json
from offgrid_da.data.collate import single_cycle_collate
from offgrid_da.models import CrossAttentionAnalysisModel

OBSERVATION_KEYS = ("obs_values_normalized", "obs_locations", "obs_time_offsets",
                    "obs_variables", "obs_sources", "obs_quality", "observation_mask")


class CachedSamples(Dataset):
    def __init__(self, dataset):
        self.paths = [dataset.sample_path(i) for i in range(len(dataset))]
        self.signature = dataset.signature
        if not all(p.is_file() for p in self.paths):
            raise ValueError("Complete sample-cache preparation is required")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        sample = torch.load(self.paths[index], map_location="cpu", weights_only=False)
        if sample.get("cache_signature") != self.signature:
            raise ValueError("Sample cache provenance mismatch")
        return sample


class BackgroundCorrection(nn.Module):
    """Exactly the DA model's convolutional branch, with no observation inputs."""
    def __init__(self, template):
        super().__init__()
        import copy
        self.input = copy.deepcopy(template.input)
        self.blocks = copy.deepcopy(template.blocks)
        self.output = copy.deepcopy(template.output)

    def forward(self, background, observations=None):
        x = self.input(background)
        for block in self.blocks:
            x = x + block(x)
        return {"analysis": background + self.output(x)}


def make_model(kind, config):
    template = CrossAttentionAnalysisModel(**config)
    return template if kind == "da" else BackgroundCorrection(template)


def atomic_checkpoint(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def pack(batch, device):
    for i, ids in enumerate(batch["context_station_ids"]):
        if int(batch["observation_mask"][i].sum()) != len(ids):
            raise ValueError("Batch mask mismatch")
        if set(ids) & set(batch["heldout_station_ids"][i]):
            raise ValueError("Held-out stations entered model inputs")
    x = batch["background_normalized"].to(device, non_blocking=True)
    y = batch["reference_normalized"].to(device, non_blocking=True)
    obs = {k: batch[k].to(device, non_blocking=True) for k in OBSERVATION_KEYS}
    return x, obs, y


def weights_for(x):
    return torch.cos(torch.deg2rad(torch.linspace(90, -90, x.shape[-2], device=x.device))).clamp_min(0)[None, None, :, None]


def training_loss(prediction, target):
    w = weights_for(prediction)
    return ((prediction - target).square() * w).sum() / w.expand_as(prediction).sum()


def evaluate(model, loader, device, std):
    if model is not None:
        model.eval()
    error_sum, denominator, baseline_sum, cycles = 0., 0., 0., 0
    with torch.inference_mode():
        for batch in loader:
            x, obs, y = pack(batch, device)
            prediction = x if model is None else model(x, obs)["analysis"]
            if not torch.isfinite(prediction).all():
                raise ValueError("Nonfinite validation output")
            w = weights_for(x).double()
            denominator += float(w.expand_as(x).sum())
            error_sum += float(((prediction.double() - y.double()).square() * w).sum()) * std ** 2
            baseline_sum += float(((x.double() - y.double()).square() * w).sum()) * std ** 2
            cycles += len(x)
    return {"rmse_K": math.sqrt(error_sum / denominator),
            "baseline0_rmse_K": math.sqrt(baseline_sum / denominator), "cycles": cycles}


def observation_checks(model, batch, device):
    model.eval()
    x, obs, _ = pack(batch, device)
    with torch.inference_mode():
        expected = model(x, obs)["analysis"]
        empty = {k: v.clone() for k, v in obs.items()}
        empty["observation_mask"].zero_()
        if not torch.equal(model(x, empty)["analysis"], x):
            raise ValueError("No-observation identity violated")
        changed = {k: v.clone() for k, v in obs.items()}
        changed["obs_values_normalized"][obs["observation_mask"]] += 1.
        sensitivity = float((model(x, changed)["analysis"] - expected).abs().max())
        if sensitivity <= 1e-9:
            raise ValueError("Trained model is insensitive to observations")
    return {"empty_exact_identity": True, "observation_sensitivity_normalized": sensitivity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("da", "neural_correction"), default="da")
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("An explicitly assigned GPU is required for this experiment")
    device = torch.device("cuda:0")
    prep = json.loads(args.preparation.read_text())
    if not prep["complete"]:
        raise ValueError("Full train/validation preparation did not pass")
    physical = {s: LazyArchesDataset(ROOT, s) for s in ("train", "val")}
    for s, dataset in physical.items():
        if prep["signatures"][s] != dataset.signature:
            raise ValueError("Prepared data signature changed")
    data = {s: CachedSamples(d) for s, d in physical.items()}
    std = physical["train"].norm_stats.std("background", "2m_temperature")
    if args.trial:
        val_indices = np.linspace(0, len(data["val"]) - 1, cfg["trial_validation_cycles"], dtype=int).tolist()
    else:
        val_indices = list(range(len(data["val"])))
    val_data = Subset(data["val"], val_indices)
    loader_args = {"batch_size": cfg["batch_size"], "collate_fn": single_cycle_collate,
                   "num_workers": cfg["loader_workers"], "pin_memory": True}
    if cfg["loader_workers"]:
        loader_args.update(multiprocessing_context="spawn", persistent_workers=False)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    batches_per_epoch = math.ceil(len(data["train"]) / cfg["batch_size"])
    max_steps = cfg["trial_steps"] if args.trial else cfg["epochs"] * batches_per_epoch
    model = make_model(args.kind, cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **cfg["optimizer"])
    warmup = min(cfg["warmup_steps"], max_steps // 4)

    def schedule(step):
        if warmup and step < warmup:
            return max(1, step) / warmup
        phase = min(1., max(0., (step - warmup) / max(1, max_steps - warmup)))
        return .05 + .95 * .5 * (1 + math.cos(math.pi * phase))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    args.output.mkdir(parents=True, exist_ok=True)
    step, best, bad_epochs = 0, float("inf"), 0
    resume_signature = {"config_sha256": config_sha, "data": prep["signatures"],
                        "kind": args.kind, "trial": args.trial, "max_steps": max_steps}
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt["signature"] != resume_signature:
            raise ValueError("Incompatible checkpoint/config/data on resume")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step, best, bad_epochs = ckpt["step"], ckpt["best"], ckpt["bad_epochs"]
        random.setstate(ckpt["rng_python"])
        np.random.set_state(ckpt["rng_numpy"])
        torch.set_rng_state(ckpt["rng_torch"])
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    import swanlab
    run = swanlab.init(project="offgrid-da", experiment_name=f"{cfg['name']}-{args.kind}-{'trial' if args.trial else 'formal'}",
                       mode=cfg["swanlab_mode"], logdir=str(args.output / "swanlogs"), config=cfg)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()

    def save(name):
        atomic_checkpoint(args.output / name, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "step": step, "best": best,
                          "bad_epochs": bad_epochs, "signature": resume_signature,
                          "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                          "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()})

    def validate():
        nonlocal best, bad_epochs
        result = evaluate(model, val_loader, device, std)
        result.update(step=step, learning_rate=optimizer.param_groups[0]["lr"])
        with (args.output / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        swanlab.log({"val_rmse_K": result["rmse_K"], "baseline0_rmse_K": result["baseline0_rmse_K"]}, step=step)
        print("VALIDATION", json.dumps(result), flush=True)
        if result["rmse_K"] < best:
            best = result["rmse_K"]
            bad_epochs = 0
            save("best.pt")
        else:
            bad_epochs += 1
        return result

    if not args.resume:
        validate()
    train_seconds = 0.
    last_validated = -1
    while step < max_steps:
        epoch = step // batches_per_epoch
        offset = (step % batches_per_epoch) * cfg["batch_size"]
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(data["train"]), generator=generator).tolist()[offset:]
        train_loader = DataLoader(Subset(data["train"], order), shuffle=False,
                                  generator=torch.Generator().manual_seed(seed + epoch), **loader_args)
        for batch in train_loader:
            model.train()
            begin = time.perf_counter()
            x, obs, target = pack(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, obs)["analysis"]
            loss = training_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            train_seconds += elapsed
            step += 1
            if step % cfg["log_every_steps"] == 0 or step == 1:
                item = {"step": step, "epoch": epoch, "train_loss": float(loss),
                        "gradient_norm": float(gradient_norm), "step_seconds": elapsed,
                        "peak_memory_MiB": torch.cuda.max_memory_allocated() / 2**20}
                with (args.output / "training.jsonl").open("a") as stream:
                    stream.write(json.dumps(item) + "\n")
                swanlab.log(item, step=step)
                print("TRAIN", json.dumps(item), flush=True)
            if step % cfg["save_every_steps"] == 0:
                save(f"step_{step}.pt")
                save("last.pt")
            validate_now = step % cfg["trial_validate_every"] == 0 if args.trial else step % batches_per_epoch == 0
            if validate_now or step == max_steps:
                validate()
                last_validated = step
            if step >= max_steps or (not args.trial and bad_epochs >= cfg["patience_epochs"]):
                break
        if not args.trial and bad_epochs >= cfg["patience_epochs"]:
            break
    if last_validated != step:
        validate()
    save("last.pt")
    # Check the trained last state before selecting best (which can legitimately be identity).
    first_batch = next(iter(val_loader))
    checks = observation_checks(model, first_batch, device) if args.kind == "da" else {"observations_unused": True}
    clone = make_model(args.kind, cfg["model"]).to(device)
    clone.load_state_dict(torch.load(args.output / "last.pt", map_location="cpu", weights_only=False)["model"])
    clone.eval()
    model.eval()
    x, obs, _ = pack(first_batch, device)
    with torch.inference_mode():
        torch.testing.assert_close(clone(x, obs)["analysis"], model(x, obs)["analysis"], rtol=0, atol=0)
    checks["checkpoint_reload_exact"] = True
    best_ckpt = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    selected = evaluate(model, val_loader, device, std)
    result = {"kind": args.kind, "trial": args.trial, "steps": step, "best_step": best_ckpt["step"],
              "train_cycles": len(data["train"]), "validation_indices": val_indices,
              "selected_validation": selected, "checks": checks,
              "parameters": sum(p.numel() for p in model.parameters()),
              "peak_memory_MiB": torch.cuda.max_memory_allocated() / 2**20,
              "training_compute_seconds": train_seconds, "wall_seconds": time.time() - started,
              "signature": resume_signature}
    atomic_json(args.output / "result.json", result)
    swanlab.finish()
    print("COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

