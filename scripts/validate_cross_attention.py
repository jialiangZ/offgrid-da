"""Bounded real-data integration and tiny-training diagnostic; not a skill benchmark."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from offgrid_da.data.dataset import SingleCycleDADataset
from offgrid_da.data.collate import single_cycle_collate
from offgrid_da.models import CrossAttentionAnalysisModel


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/single_da_00_model_check_crossattn"
    output.mkdir(parents=True, exist_ok=True)
    dataset = SingleCycleDADataset(split="minimal", start="2020-01-01",
                                   end="2020-01-04", require_normalization=True)
    print("loading six real cycles", flush=True)
    samples = [dataset[i] for i in range(6)]
    assert all(set(s["context_station_ids"]).isdisjoint(s["heldout_station_ids"]) for s in samples)
    stats = dataset.norm_stats
    mean = torch.tensor([stats.mean("background", v) for v in dataset.variables], device=device)[None,:,None,None]
    std = torch.tensor([stats.std("background", v) for v in dataset.variables], device=device)[None,:,None,None]

    def pack(items):
        batch = single_cycle_collate(items)
        bg_before, ref_before = batch["background"].clone(), batch["reference"].clone()
        bg = batch["background_normalized"].to(device)
        # Only the explicit context-input allowlist enters the model.
        keys = ("obs_values_normalized", "obs_locations", "obs_time_offsets",
                "obs_variables", "obs_sources", "obs_quality", "observation_mask")
        obs = {k: batch[k].to(device) for k in keys}
        ref = batch["reference"].to(device)
        target = (ref - mean) / std  # residual output shares background coordinates
        return batch, bg, obs, target, bg_before, ref_before

    batch, bg, obs, target, bg_before, ref_before = pack(samples[:2])
    model = CrossAttentionAnalysisModel(dim=16, heads=2, depth=2).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    predicted = model(bg, obs)
    assert torch.isfinite(predicted["analysis"]).all()
    (predicted["analysis"] - target).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    torch.testing.assert_close(batch["background"], bg_before)
    torch.testing.assert_close(batch["reference"], ref_before)
    model.eval()
    with torch.no_grad():
        expected = model(bg, obs)["analysis"]
        changed = {k: v.clone() for k, v in obs.items()}
        changed["obs_values_normalized"][obs["observation_mask"]] += 1
        sensitivity = (model(bg, changed)["analysis"] - expected).abs().max().item()
        assert sensitivity > 1e-6
        perm = torch.randperm(obs["observation_mask"].shape[1], device=device)
        permuted = {k: v[:, perm] for k,v in obs.items()}
        permutation_error = (model(bg, permuted)["analysis"] - expected).abs().max().item()
        assert permutation_error < 1e-4
        padded = {k: v.clone() for k,v in obs.items()}
        for key in padded:
            if key != "observation_mask":
                padded[key][~obs["observation_mask"]] = 12345
        torch.testing.assert_close(model(bg, padded)["analysis"], expected)
    report = {
        "purpose": "integration and tiny-training diagnostic, not generalization evidence",
        "seed": 42, "device": str(device), "innovation_enabled": False,
        "real_batch": {"sample_ids": batch["sample_ids"], "background_shape": list(bg.shape),
                       "token_shape": list(predicted["tokens"].shape),
                       "observation_counts": obs["observation_mask"].sum(1).tolist(),
                       "forward_backward_seconds": elapsed,
                       "peak_allocated_mb": torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else None,
                       "observation_perturbation_max_change": sensitivity,
                       "permutation_max_error": permutation_error,
                       "padding_and_raw_fields_unchanged": True},
        "normalization": {"path": dataset.cfg["normalization"]["stats_path"],
                          "output_and_target_coordinates": "background training mean/std"},
    }
    print(json.dumps(report), flush=True)
    train_items, diagnostic_items = samples[:4], samples[4:6]
    lat = train_items[0]["grid_latitude"].to(device)
    weights = torch.cos(torch.deg2rad(lat)).clamp_min(0)[None,None,:,None]

    def loss(pred, truth):
        return ((pred-truth).square()*weights).sum() / (weights.expand_as(pred).sum())

    def evaluate(net, items):
        _, x, o, y, _, _ = pack(items)
        with torch.no_grad():
            estimate = net(x,o)["analysis"] if net is not None else x
            return {"normalized_mse": loss(estimate,y).item(),
                    "area_weighted_rmse_K": loss(estimate*std,y*std).sqrt().item()}

    report["train_sample_ids"] = [s["sample_id"] for s in train_items]
    report["diagnostic_sample_ids"] = [s["sample_id"] for s in diagnostic_items]
    report["diagnostic_split_note"] = "disjoint nearby cycles inside the official training period; not official validation"
    report["baseline0"] = {"train": evaluate(None, train_items),
                           "diagnostic": evaluate(None, diagnostic_items)}
    report["training"] = {}
    for mode in ("entry", "layered"):
        torch.manual_seed(42)
        net = CrossAttentionAnalysisModel(dim=16, heads=2, depth=2, injection=mode).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        initial = evaluate(net, train_items)
        history = []
        start = time.perf_counter()
        net.train()
        for step in range(40):
            _, x, o, y, _, _ = pack([train_items[step % 4]])
            optimizer.zero_grad(set_to_none=True)
            objective = loss(net(x,o)["analysis"], y)
            assert torch.isfinite(objective)
            objective.backward()
            assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            history.append(objective.item())
            if (step+1) % 10 == 0:
                print(f"{mode}: {step+1}/40 loss={history[-1]:.6f}", flush=True)
        net.eval()
        final = evaluate(net, train_items)
        assert final["normalized_mse"] < initial["normalized_mse"]
        report["training"][mode] = {"initial_train": initial, "final_train": final,
                                    "diagnostic": evaluate(net, diagnostic_items),
                                    "steps": 40, "losses": history,
                                    "seconds": time.perf_counter()-start,
                                    "parameters": sum(p.numel() for p in net.parameters())}
        torch.save(net.state_dict(), output / f"{mode}_diagnostic.pt")
        (output / "results.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
