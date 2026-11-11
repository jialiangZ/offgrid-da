#!/usr/bin/env python3
"""Step-3 observation-content utilization diagnostics (plan §5).

Inference-time interventions on TRAINED models — no new training. For each arm
(real METAR: single_da_09; synthetic: step2) and seed (42/43), evaluate 2021
under conditions that keep stations/times/masks fixed and change only obs content:

  correct   : the arm's own training-time observation values
  zero_innov: value = H(x_b) (bilinear background at station; innovation == 0;
              carries position/mask information only, no value information)
  shuffle0/1/2: values permuted within each variable (3 fixed permutations)
  attn_off  : correct values, model's local-attention updates disabled
              (conv branch only, same trained weights)

Primary read-out: dimensionless error-balanced score + physical RMSE of the four
surface channels, and their delta vs `correct`. If `zero_innov` keeps a large
share of the gain, the network exploits positions rather than values (structural
waste); if it collapses to NC level, the value pathway works and the gap is
capacity/training.

Outputs: outputs/step3_obs_content/{results.json, report.md}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from offgrid_da.data.metar_multivar import SPECS  # noqa: E402
from offgrid_da.data.alignment import sample_background_at_obs  # noqa: E402
from offgrid_da.data.multivar_dataset import multivar_collate  # noqa: E402
from offgrid_da.data.full_state_data import CHANNELS  # noqa: E402
from offgrid_da.models import CrossAttentionAnalysisModel  # noqa: E402
from error_balanced import metrics  # noqa: E402
import run_full_update_v1 as base_runner  # noqa: E402
import train_step2_synthetic as syn_module  # noqa: E402

OBS_KEYS = ("obs_values_normalized", "obs_locations", "obs_time_offsets",
            "obs_variables", "obs_sources", "obs_quality", "observation_mask")
SYNTH_VARS = (0, 1, 2, 4)
STATS = ROOT / "outputs/single_da_09_full82_loss_balanced/loss_statistics.json"
MODELS = {
    "real": [ROOT / f"outputs/single_da_09_full82_loss_balanced/seed{s}/da/formal/best.pt" for s in (42, 43)],
    "syn": [ROOT / f"outputs/step2_synthetic_da/seed{s}/da/formal/best.pt" for s in (42, 43)],
}
MODEL_CFG = {"dim": 128, "heads": 8, "depth": 4, "injection": "layered",
             "radius_km": 1000.0, "max_neighbors": 80, "query_chunk": 256}
SURFACE_IDX = {"2m_temperature": 2, "10m_u_component_of_wind": 0,
               "10m_v_component_of_wind": 1, "mean_sea_level_pressure": 3}


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Non-parameter architecture hyperparameters (radius/max_neighbors/schedules)
    # are NOT stored in state_dict; recover them from the sibling config.json
    # so models trained with schedule variants run with their own neighborhoods.
    cfg = {"dim": 128, "heads": 8, "depth": 4, "injection": "layered",
           "radius_km": 1000.0, "max_neighbors": 80, "query_chunk": 256}
    cfg_path = Path(ckpt_path).parent / "config.json"
    if cfg_path.exists():
        for k in ("radius_km", "max_neighbors", "radius_schedule", "neighbors_schedule"):
            if k in (mc := json.loads(cfg_path.read_text()).get("model", {})):
                cfg[k] = mc[k]
    model = CrossAttentionAnalysisModel(channels=82, **{k: v for k, v in cfg.items() if v is not None})
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model


def attn_off(model):
    """Return a conv-branch-only view of the trained model (attention zeroed)."""
    import copy
    m = copy.deepcopy(model)
    m._local = lambda q, tokens, mask, locations, h, w, layer: torch.zeros_like(q)
    return m


def zero_innov_batch(batch, dataset):
    """Set the four surface obs values to H(x_b); dewpoint to background T2M."""
    x = batch["background_normalized"][0]  # (82,H,W) normalized
    mean = dataset.base.mean  # (82,1,1)
    std = dataset.base.std
    bg_phys = x * std + mean
    lat = batch["grid_latitude"].numpy().astype(np.float64)
    lon = batch["grid_longitude"].numpy().astype(np.float64)
    var_ids = batch["obs_variables"][0].numpy()
    locs = batch["obs_locations"][0].numpy().astype(np.float64)
    vals = batch["obs_values_normalized"][0].clone()
    for v in list(SYNTH_VARS) + [3]:
        ch = SPECS[v]["background_channel"]
        m = var_ids == v
        if not m.any():
            continue
        hb = sample_background_at_obs(bg_phys[ch].numpy().astype(np.float64),
                                      lat, lon, locs[m, 0], locs[m, 1])
        ok = np.isfinite(hb)
        idx = np.where(m)[0][ok]
        vals[torch.from_numpy(idx)] = torch.from_numpy(
            ((hb[ok] - float(mean[ch])) / float(std[ch])).astype(np.float32))
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    out["obs_values_normalized"] = vals[None]
    return out


def shuffle_batch(batch, perm_seed):
    g = torch.Generator().manual_seed(perm_seed)
    var_ids = batch["obs_variables"][0]
    vals = batch["obs_values_normalized"][0].clone()
    for v in torch.unique(var_ids):
        m = var_ids == v
        n = int(m.sum())
        if n < 2:
            continue
        vals[m] = vals[m][torch.randperm(n, generator=g)]
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    out["obs_values_normalized"] = vals[None]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1, help="evaluate every Nth val cycle")
    ap.add_argument("--out", default="outputs/step3_obs_content")
    args = ap.parse_args()
    device = torch.device("cuda:0")
    stats = json.loads(STATS.read_text())
    loss_w = stats["loss_weights"]
    phys_std = np.array(stats["physical_std"])

    datasets = {"real": base_runner.Data(ROOT, "val"),
                "syn": syn_module.SyntheticData(ROOT, "val")}
    models = {}
    for arm, paths in MODELS.items():
        models[arm] = []
        for p in paths:
            m = load_model(p, device)
            models[arm].append({"full": m, "off": attn_off(m)})

    lat_w = torch.cos(torch.deg2rad(torch.linspace(90, -90, 121, device=device))).clamp_min(0)[None, :, None]
    results = {}
    for arm, ds in datasets.items():
        idxs = list(range(0, len(ds), args.stride))
        sse = {(si, cond): torch.zeros(82, dtype=torch.float64, device=device)
               for si in (0, 1) for cond in ("correct", "zero_innov", "attn_off", "shuffle0", "shuffle1", "shuffle2")}
        den = 0.0
        n_cycles = 0
        with torch.inference_mode():
            for i in idxs:
                sample = ds[i]
                batch = multivar_collate([sample])
                x = batch["background_normalized"].to(device)
                y = batch["reference_normalized"].to(device)
                obs = {k: batch[k].to(device) for k in OBS_KEYS}
                batch_dev = {"obs_values_normalized": obs["obs_values_normalized"]}
                obs_zero = {k: v for k, v in obs.items()}
                zb = zero_innov_batch({"grid_latitude": batch["grid_latitude"],
                                       "grid_longitude": batch["grid_longitude"],
                                       "background_normalized": x.cpu(),
                                       "obs_variables": obs["obs_variables"].cpu(),
                                       "obs_locations": obs["obs_locations"].cpu(),
                                       "obs_values_normalized": obs["obs_values_normalized"].cpu()}, ds)
                obs_zero["obs_values_normalized"] = zb["obs_values_normalized"].to(device)
                shuf = [shuffle_batch({"obs_variables": obs["obs_variables"].cpu(),
                                       "obs_values_normalized": obs["obs_values_normalized"].cpu()}, s_)
                        for s_ in (0, 1, 2)]
                cond_obs = {"correct": obs, "zero_innov": obs_zero,
                            "shuffle0": {**obs, "obs_values_normalized": shuf[0]["obs_values_normalized"].to(device)},
                            "shuffle1": {**obs, "obs_values_normalized": shuf[1]["obs_values_normalized"].to(device)},
                            "shuffle2": {**obs, "obs_values_normalized": shuf[2]["obs_values_normalized"].to(device)}}
                for si in (0, 1):
                    for cond, ob in cond_obs.items():
                        pred = models[arm][si]["full"](x, ob)["analysis"]
                        sse[(si, cond)] += ((pred.double() - y.double()).square() * lat_w[None]).sum((0, 2, 3))
                    pred = models[arm][si]["off"](x, obs)["analysis"]
                    sse[(si, "attn_off")] += ((pred.double() - y.double()).square() * lat_w[None]).sum((0, 2, 3))
                den += float(lat_w.sum()) * x.shape[-1] * x.shape[0]
                n_cycles += 1
                if n_cycles % 100 == 0:
                    print(f"[{arm}] {n_cycles}/{len(idxs)} cycles", flush=True)
        results[arm] = {}
        for si in (0, 1):
            for cond in ("correct", "zero_innov", "attn_off", "shuffle0", "shuffle1", "shuffle2"):
                r = metrics(sse[(si, cond)].cpu().numpy(), den, loss_w, phys_std, CHANNELS)
                rmse = np.sqrt(sse[(si, cond)].cpu().numpy() / den) * phys_std
                results[arm][f"seed{42 + si}_{cond}"] = {
                    "score": r["score"],
                    "surface_rmse": {n: float(rmse[c]) for n, c in SURFACE_IDX.items()},
                }
        print(f"[{arm}] done: {n_cycles} cycles", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    lines = ["# Step-3 observation-content diagnostics (2021)", ""]
    for arm in ("real", "syn"):
        lines.append(f"## {arm} arm")
        lines.append("| condition | score | T2M | U10 | V10 | MSLP |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for si in (0, 1):
            for cond in ("correct", "zero_innov", "shuffle0", "shuffle1", "shuffle2", "attn_off"):
                r = results[arm][f"seed{42 + si}_{cond}"]
                s = r["surface_rmse"]
                lines.append(f"| s{42 + si} {cond} | {r['score']:.5f} | {s['2m_temperature']:.4f} | "
                             f"{s['10m_u_component_of_wind']:.4f} | {s['10m_v_component_of_wind']:.4f} | "
                             f"{s['mean_sea_level_pressure']:.2f} |")
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
