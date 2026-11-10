#!/usr/bin/env python3
"""Step-4 noise dose-response (plan section 8, first extension, inference-only).

Inject y = H(x_ref) + k * sigma_innov * eps into the SYNTHETIC observations
(k in 0/0.25/0.5/1.0, eps fixed per cycle/var, sigma = per-batch innovation
std = std of y_syn - H(x_b), self-calibrated) and evaluate all four trained
models (syn arms trained on noise-free obs; real arms trained on real METAR).

Dose-response question: does k=1 (noise at the level of real obs-vs-reference
differences) shrink the synthetic arm's gain to the real arm's level
(~ -0.0065 K)? If yes, H2 is fully explained by noise magnitude; if the
synthetic arm stays above the real arm, other representation differences
remain.

Outputs: outputs/step4_noise_response/{results.json, report.md}
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
from offgrid_da.data.full_state_data import CHANNELS  # noqa: E402
from offgrid_da.data.multivar_dataset import multivar_collate  # noqa: E402
from offgrid_da.models import CrossAttentionAnalysisModel  # noqa: E402
from error_balanced import metrics  # noqa: E402
import run_full_update_v1 as base_runner  # noqa: E402
import train_step2_synthetic as syn_module  # noqa: E402
from step3_obs_content_diag import OBS_KEYS, SYNTH_VARS, load_model, MODELS  # noqa: E402

KS = (0.0, 0.25, 0.5, 1.0)
SURFACE_IDX = {"2m_temperature": 2, "10m_u_component_of_wind": 0,
               "10m_v_component_of_wind": 1, "mean_sea_level_pressure": 3}
STATS = ROOT / "outputs/single_da_09_full82_loss_balanced/loss_statistics.json"


def hb_normalized(batch, dataset):
    """H(x_b) in normalized space for the four synthetic vars -> (mask idx, values)."""
    x = batch["background_normalized"][0]
    mean, std = dataset.base.mean, dataset.base.std
    bg_phys = x * std + mean
    lat = batch["grid_latitude"].numpy().astype(np.float64)
    lon = batch["grid_longitude"].numpy().astype(np.float64)
    var_ids = batch["obs_variables"][0].numpy()
    locs = batch["obs_locations"][0].numpy().astype(np.float64)
    out = {}
    for v in SYNTH_VARS:
        ch = SPECS[v]["background_channel"]
        m = var_ids == v
        if not m.any():
            continue
        hb = sample_background_at_obs(bg_phys[ch].numpy().astype(np.float64),
                                      lat, lon, locs[m, 0], locs[m, 1])
        out[v] = (np.where(m)[0], ((hb - float(mean[ch])) / float(std[ch])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", default="outputs/step4_noise_response")
    args = ap.parse_args()
    device = torch.device("cuda:0")
    stats = json.loads(STATS.read_text())
    loss_w = stats["loss_weights"]
    phys_std = np.array(stats["physical_std"])

    ds = syn_module.SyntheticData(ROOT, "val")
    real_ds = base_runner.Data(ROOT, "val")
    models = {arm: [load_model(p, device) for p in paths] for arm, paths in MODELS.items()}

    lat_w = torch.cos(torch.deg2rad(torch.linspace(90, -90, 121, device=device))).clamp_min(0)[None, :, None]
    groups = ["syn_s42", "syn_s43", "real_s42", "real_s43"]
    sse = {(g, k): torch.zeros(82, dtype=torch.float64, device=device) for g in groups for k in list(KS) + ["real_obs"]}
    den = 0.0
    n_cycles = 0
    sig_log = {v: [] for v in SYNTH_VARS}
    with torch.inference_mode():
        for i in range(0, len(ds), args.stride):
            batch = multivar_collate([ds[i]])
            x = batch["background_normalized"].to(device)
            y = batch["reference_normalized"].to(device)
            obs0 = {k: batch[k].to(device) for k in OBS_KEYS}
            hb = hb_normalized(batch, ds)
            # sigma from REAL-obs innovations (y_real - H(x_b)): the noise we emulate
            # is the obs-vs-reference difference d (~94.8% of innovation variance),
            # not the background error r that synthetic innovations carry.
            rng = np.random.default_rng(1000 + i)
            rb = multivar_collate([real_ds[i]])
            real_vals = rb["obs_values_normalized"][0].numpy()
            vals0 = batch["obs_values_normalized"][0]  # cpu
            noisy_vals = {k: vals0.clone() for k in KS if k > 0.0}
            for v, (idx, hbn) in hb.items():
                sigma = float((real_vals[idx] - hbn).std())
                sig_log[v].append(sigma)
                eps = rng.standard_normal(len(idx)) * sigma
                idx_t = torch.from_numpy(idx)
                for k in KS:
                    if k == 0.0:
                        continue
                    noisy_vals[k][idx_t] += torch.from_numpy((k * eps).astype(np.float32))
            cond_obs = {0.0: obs0}
            for k, vals in noisy_vals.items():
                ob = {kk: vv for kk, vv in obs0.items()}
                ob["obs_values_normalized"] = vals[None].to(device)
                cond_obs[k] = ob
            for si in (0, 1):
                for k in KS:
                    p = models["syn"][si](x, cond_obs[k])["analysis"]
                    sse[(f"syn_s{42+si}", k)] += ((p.double() - y.double()).square() * lat_w[None]).sum((0, 2, 3))
            # real arms evaluated on real obs (their own correct condition) for reference
            robs = {k: rb[k].to(device) for k in OBS_KEYS}
            for si in (0, 1):
                p = models["real"][si](x, robs)["analysis"]
                sse[(f"real_s{42+si}", "real_obs")] += ((p.double() - y.double()).square() * lat_w[None]).sum((0, 2, 3))
            den += float(lat_w.sum()) * x.shape[-1] * x.shape[0]
            n_cycles += 1
            if n_cycles % 60 == 0:
                print(f"{n_cycles} cycles", flush=True)

    results = {"cycles": n_cycles, "stride": args.stride, "ks": list(KS),
               "innov_sigma_norm_median": {str(v): float(np.median(s)) for v, s in sig_log.items()},
               "groups": groups}
    for g in groups:
        conds = ["real_obs"] if g.startswith("real") else list(KS)
        for k in conds:
            key = (g, k)
            if key not in sse:
                continue
            r = metrics(sse[key].cpu().numpy(), den, loss_w, phys_std, CHANNELS)
            rmse = np.sqrt(sse[key].cpu().numpy() / den) * phys_std
            results[f"{g}/{k}"] = {"score": r["score"],
                                   "surface_rmse": {n: float(rmse[c]) for n, c in SURFACE_IDX.items()}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    lines = ["# Step-4 noise dose-response (synthetic obs, 2021)", "",
             "T2M RMSE (K) vs noise dose k (sigma = per-batch innovation std):", "",
             "| group | k=0 | k=0.25 | k=0.5 | k=1.0 | real-obs level |", "|---|---:|---:|---:|---:|---:|"]
    for g in ("syn_s42", "syn_s43", "real_s42", "real_s43"):
        row = [f"{results[g+'/'+str(k)]['surface_rmse']['2m_temperature']:.4f}" if g + '/' + str(k) in results else "—" for k in KS]
        ref = results.get(g + "/real_obs", {}).get("surface_rmse", {}).get("2m_temperature")
        lines.append(f"| {g} | " + " | ".join(row) + f" | {ref:.4f} |" if ref else f"| {g} | " + " | ".join(row) + " | — |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
