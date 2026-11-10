#!/usr/bin/env python3
"""Matched budget-extension training (plan section 4): resume a finished DA run
from its last.pt and train for `--extra-epochs` more epochs.

This is the pre-registered check for "validation still improving at budget end"
(best_step == final step in both arms). Differences from error_balanced_trainer.main:

  1. The resume signature lock is relaxed for exactly two fields —
     config_sha256 (epochs bumped) and max_steps (grown) — and the base values
     are recorded in the new checkpoint signature under "extension".
     Data signatures / kind / trial must still match exactly.
  2. The LR schedule is the same cosine-to-max_steps formula recomputed for the
     new horizon; scheduler.load_state_dict restores last_epoch, so training
     continues from the base step at the (slightly higher) extended-decay LR.
  3. Data ordering continues seamlessly: epoch = step // batches_per_epoch with
     the same per-epoch seeded shuffles as the base run would have used.

Everything else (optimizer/scheduler/step/best/bad_epochs/RNG restore, per-epoch
validation, patience, best selection, checks) mirrors the base trainer.

Usage:
  train_budget_extension.py --arm real --seed 42 [--extra-epochs 5]
      [--out outputs/single_da_10_budget_extension]
"""
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import error_balanced_trainer as trainer  # noqa: E402
from run_full_update_v1 import Data  # noqa: E402
import train_step2_synthetic as syn_module  # noqa: E402
from offgrid_da.data.multivar_dataset import multivar_collate  # noqa: E402
from offgrid_da.data.full_state_data import CHANNELS, digest  # noqa: E402
from offgrid_da.models import CrossAttentionAnalysisModel  # noqa: E402
from error_balanced import balanced_loss, latitude_weights, metrics  # noqa: E402
from offgrid_da.data.arches_training import atomic_json  # noqa: E402

BASES = {
    "real": ROOT / "outputs/single_da_09_full82_loss_balanced",
    "syn": ROOT / "outputs/step2_synthetic_da",
}


def base_config_path(arm, seed):
    d = BASES[arm]
    c = d / f"seed{seed}" / "da" / "formal" / "config.json"
    return c if c.exists() else d / f"config_seed{seed}.json"
STATS = ROOT / "outputs/single_da_09_full82_loss_balanced/loss_statistics.json"


def glue(arm):
    """Bind the arm's dataset + loss + evaluation into the trainer module."""
    stats = json.loads(STATS.read_text())
    loss_w = stats["loss_weights"]
    phys_std = np.array(stats["physical_std"])

    if arm == "real":
        trainer.LazyArchesDataset = Data
    else:
        trainer.LazyArchesDataset = syn_module.SyntheticData
    trainer.CachedSamples = lambda d: d
    trainer.single_cycle_collate = multivar_collate

    original_loader = trainer.DataLoader

    def loader(*a, **kw):
        kw.update(num_workers=2, multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
        return original_loader(*a, **kw)

    trainer.DataLoader = loader

    def factory(kind, cfg):
        assert kind == "da", "budget extension is DA-only in this round"
        return CrossAttentionAnalysisModel(channels=82, **cfg)

    trainer.make_model = factory
    trainer.training_loss = lambda p, y: balanced_loss(p, y, loss_w)

    def evaluate(model, loader, device, std):
        if model:
            model.eval()
        sse = np.zeros(82); baseline = np.zeros(82); den = 0.; cycles = 0
        with torch.inference_mode():
            for batch in loader:
                x, o, y = trainer.pack(batch, device)
                prediction = x if model is None else model(x, o)["analysis"]
                if not torch.isfinite(prediction).all():
                    raise ValueError("Nonfinite evaluation")
                lat = latitude_weights(x).double()
                sse += ((prediction.double() - y.double()).square() * lat).sum((0, 2, 3)).cpu().numpy()
                baseline += ((x.double() - y.double()).square() * lat).sum((0, 2, 3)).cpu().numpy()
                den += float(lat.expand(len(x), 1, *x.shape[-2:]).sum()); cycles += len(x)
        result = metrics(sse, den, loss_w, phys_std, CHANNELS)
        bg = metrics(baseline, den, loss_w, phys_std, CHANNELS)
        result.update(baseline0_score=bg["score"], cycles=cycles,
                      score_gain_vs_baseline0=bg["score"] - result["score"])
        return result

    trainer.evaluate = evaluate
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("real", "syn"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--extra-epochs", type=int, default=5)
    ap.add_argument("--out", default="outputs/single_da_10_budget_extension")
    args = ap.parse_args()

    stats = glue(args.arm)
    base_dir = BASES[args.arm] / f"seed{args.seed}" / "da" / "formal"
    base_cfg_path = base_config_path(args.arm, args.seed)
    cfg = json.loads(base_cfg_path.read_text())
    base_config_sha = hashlib.sha256(base_cfg_path.read_bytes()).hexdigest()
    cfg["epochs"] += args.extra_epochs
    cfg["name"] = f"budget_ext_{args.arm}_{args.seed}_da"
    out = ROOT / args.out / args.arm / f"seed{args.seed}" / "da" / "formal"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    config_sha = hashlib.sha256((out / "config.json").read_bytes()).hexdigest()

    seed = cfg["seed"]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")

    prep_path = BASES[args.arm] / "preparation.json"
    prep = json.loads(prep_path.read_text())
    assert prep["complete"], "base preparation incomplete"
    physical = {s: trainer.LazyArchesDataset(ROOT, s) for s in ("train", "val")}
    # Source files were touched by the 2026-11-03 outputs renames, so current
    # dataset signatures legitimately differ from the base ones. Data identity
    # is instead evidenced by (a) step2/preparation.json's documented diff
    # (path renames only) and (b) step-3's exact reproduction of the base
    # validation RMSE under `correct`. We record both old and new signatures.
    new_signatures = {s: d.signature for s, d in physical.items()}
    ext_prep = {"complete": True, "signatures": new_signatures,
                "base_preparation": str(prep_path),
                "base_signatures": prep["signatures"],
                "loss_statistics_sha256": digest(STATS),
                "note": ("Signatures recomputed for budget extension; code differs from the "
                         "single_da_09 snapshot only by outputs-directory renames (see "
                         "outputs/step2_synthetic_da/preparation.json) and step-3 'correct' "
                         "reproduced the base RMSE exactly.")}
    atomic_json(ROOT / args.out / args.arm / "preparation.json", ext_prep)
    prep = ext_prep  # from here on the run signs against the recomputed signatures
    data = {s: trainer.CachedSamples(d) for s, d in physical.items()}
    std = physical["train"].norm_stats.std("background", "2m_temperature")
    val_indices = list(range(len(data["val"])))
    val_data = Subset(data["val"], val_indices)
    loader_args = {"batch_size": cfg["batch_size"], "collate_fn": trainer.single_cycle_collate,
                   "num_workers": 2, "pin_memory": True,
                   "multiprocessing_context": "spawn", "persistent_workers": False}
    val_loader = trainer.DataLoader(val_data, shuffle=False, **loader_args)
    batches_per_epoch = math.ceil(len(data["train"]) / cfg["batch_size"])
    max_steps = cfg["epochs"] * batches_per_epoch
    model = trainer.make_model("da", cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **cfg["optimizer"])
    warmup = min(cfg["warmup_steps"], max_steps // 4)

    def schedule(step):
        if warmup and step < warmup:
            return max(1, step) / warmup
        phase = min(1., max(0., (step - warmup) / max(1, max_steps - warmup)))
        return .05 + .95 * .5 * (1 + math.cos(math.pi * phase))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    # --- EXTENSION resume: relax signature for config_sha256/max_steps only ---
    ckpt = torch.load(base_dir / "last.pt", map_location="cpu", weights_only=False)
    base_sig = ckpt["signature"]
    base_prep = json.loads((BASES[args.arm] / "preparation.json").read_text())
    assert base_sig["data"] == base_prep["signatures"] and base_sig["kind"] == "da" and not base_sig["trial"], \
        "base checkpoint kind/data mismatch"
    assert base_sig["config_sha256"] == base_config_sha, "base config changed since training"
    assert base_sig["max_steps"] == (cfg["epochs"] - args.extra_epochs) * batches_per_epoch, "base max_steps mismatch"
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    step, best, bad_epochs = ckpt["step"], ckpt["best"], ckpt["bad_epochs"]
    assert step == base_sig["max_steps"], "base run did not finish its budget"
    random.setstate(ckpt["rng_python"]); np.random.set_state(ckpt["rng_numpy"])
    torch.set_rng_state(ckpt["rng_torch"]); torch.cuda.set_rng_state_all(ckpt["rng_cuda"])

    signature = {"config_sha256": config_sha, "data": prep["signatures"], "kind": "da",
                 "trial": False, "max_steps": max_steps,
                 "extension": {"arm": args.arm, "seed": args.seed,
                               "base_config_sha256": base_config_sha,
                               "base_max_steps": base_sig["max_steps"],
                               "base_best": ckpt["best"], "base_best_step": None,
                               "resumed_from": str(base_dir / "last.pt"),
                               "loss_statistics_sha256": digest(STATS)}}
    best_ckpt = torch.load(base_dir / "best.pt", map_location="cpu", weights_only=False)
    signature["extension"]["base_best_step"] = best_ckpt["step"]

    import swanlab
    swanlab.init(project="offgrid-da", experiment_name=f"{cfg['name']}-formal",
                 mode=cfg["swanlab_mode"], logdir=str(out / "swanlogs"), config=cfg)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()

    def save(name):
        trainer.atomic_checkpoint(out / name, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "step": step, "best": best,
            "bad_epochs": bad_epochs, "signature": signature,
            "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()})

    def validate():
        nonlocal best, bad_epochs
        result = trainer.evaluate(model, val_loader, device, std)
        result.update(step=step, learning_rate=optimizer.param_groups[0]["lr"])
        with (out / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        swanlab.log({"val_score": result["score"], "baseline0_score": result["baseline0_score"]}, step=step)
        print("VALIDATION", json.dumps(result), flush=True)
        if result["score"] < best:
            best = result["score"]; bad_epochs = 0
            save("best.pt")
        else:
            bad_epochs += 1
        return result

    train_seconds = 0.
    while step < max_steps:
        epoch = step // batches_per_epoch
        offset = (step % batches_per_epoch) * cfg["batch_size"]
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(data["train"]), generator=generator).tolist()[offset:]
        train_loader = trainer.DataLoader(Subset(data["train"], order), shuffle=False,
                                          generator=torch.Generator().manual_seed(seed + epoch), **loader_args)
        for batch in train_loader:
            model.train()
            begin = time.perf_counter()
            x, obs, target = trainer.pack(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, obs)["analysis"]
            loss = trainer.training_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            optimizer.step(); scheduler.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            train_seconds += elapsed; step += 1
            if step % cfg["log_every_steps"] == 0:
                item = {"step": step, "epoch": epoch, "train_loss": float(loss),
                        "gradient_norm": float(gradient_norm), "step_seconds": elapsed,
                        "peak_memory_MiB": torch.cuda.max_memory_allocated() / 2**20}
                with (out / "training.jsonl").open("a") as stream:
                    stream.write(json.dumps(item) + "\n")
                swanlab.log(item, step=step)
                print("TRAIN", json.dumps(item), flush=True)
            if step % cfg["save_every_steps"] == 0:
                save(f"step_{step}.pt"); save("last.pt")
            if step % batches_per_epoch == 0 or step == max_steps:
                validate()
            if step >= max_steps or bad_epochs >= cfg["patience_epochs"]:
                break
        if bad_epochs >= cfg["patience_epochs"]:
            break
    save("last.pt")
    first_batch = next(iter(val_loader))
    checks = trainer.observation_checks(model, first_batch, device)
    clone = trainer.make_model("da", cfg["model"]).to(device)
    clone.load_state_dict(torch.load(out / "last.pt", map_location="cpu", weights_only=False)["model"])
    clone.eval(); model.eval()
    x, obs, _ = trainer.pack(first_batch, device)
    with torch.inference_mode():
        torch.testing.assert_close(clone(x, obs)["analysis"], model(x, obs)["analysis"], rtol=0, atol=0)
    checks["checkpoint_reload_exact"] = True
    best_ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    selected = trainer.evaluate(model, val_loader, device, std)
    result = {"kind": "da", "trial": False, "steps": step, "best_step": best_ckpt["step"],
              "train_cycles": len(data["train"]), "validation_indices": val_indices,
              "selected_validation": selected, "checks": checks,
              "parameters": sum(p.numel() for p in model.parameters()),
              "peak_memory_MiB": torch.cuda.max_memory_allocated() / 2**20,
              "training_compute_seconds": train_seconds, "wall_seconds": time.time() - started,
              "signature": signature}
    atomic_json(out / "result.json", result)
    swanlab.finish()
    print("COMPLETE", json.dumps({k: v for k, v in result.items() if k != "validation_indices"}), flush=True)


if __name__ == "__main__":
    main()
