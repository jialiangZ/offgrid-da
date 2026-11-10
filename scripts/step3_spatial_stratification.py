#!/usr/bin/env python3
"""Step-3 spatial stratification (plan section 6 gap): where does the DA gain land?

For each validation cycle (default stride 4 -> 365 cycles) evaluate background
and both DA arms (real = single_da_09, syn = step2, seeds 42/43, correct obs)
on 2021 and report physical-unit T2M/U10/MSLP RMSE and bias stratified by

  1. distance-to-nearest-observation grid bins: [0,300), [300,600),
     [600,1000), [1000, inf) km  — quantifies coverage dilution of global
     metrics (92% of stations are in 30-60N);
  2. station positions split by heldout vs training role (bilinear field
     difference at station locations).

Outputs: outputs/step3_spatial/{results.json, report.md}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from offgrid_da.data.alignment import sample_background_at_obs  # noqa: E402
from offgrid_da.data.full_state_data import CHANNELS  # noqa: E402
from offgrid_da.data.multivar_dataset import multivar_collate  # noqa: E402
import run_full_update_v1 as base_runner  # noqa: E402
import train_step2_synthetic as syn_module  # noqa: E402
from step3_obs_content_diag import load_model, MODELS, OBS_KEYS  # noqa: E402

CH = {"t2m": 2, "u10": 0, "mslp": 3}
BINS = (0., 300., 600., 1000., np.inf)
R_EARTH = 6371.0


def unit_xyz(lat, lon):
    la = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lo = np.deg2rad(np.asarray(lon, dtype=np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", default="outputs/step3_spatial")
    args = ap.parse_args()
    device = torch.device("cuda:0")
    stats = json.loads((ROOT / "outputs/single_da_09_full82_loss_balanced/loss_statistics.json").read_text())
    phys_std = stats["physical_std"]

    datasets = {"real": base_runner.Data(ROOT, "val"), "syn": syn_module.SyntheticData(ROOT, "val")}
    models = {arm: [load_model(p, device) for p in paths] for arm, paths in MODELS.items()}
    n_groups = 5  # background + 2 arms x 2 seeds

    # accumulators: dist bins x channels for sse/bias, station-role for sse/bias
    dist_sse = np.zeros((n_groups, len(BINS) - 1, len(CH)))
    dist_bias = np.zeros_like(dist_sse)
    dist_den = np.zeros((len(BINS) - 1, len(CH)))  # cos-lat weight sums per bin/ch
    n_dist_cells = np.zeros(len(BINS) - 1, dtype=np.int64)
    role_names = ["train", "heldout"]
    st_sse = np.zeros((n_groups, 2, len(CH)))
    st_bias = np.zeros_like(st_sse)
    st_n = np.zeros(2, dtype=np.int64)

    latg, long = None, None
    with torch.inference_mode():
        for i in range(0, len(datasets["real"]), args.stride):
            sample = datasets["real"][i]
            batch = multivar_collate([sample])
            x = batch["background_normalized"].to(device)
            y = batch["reference_normalized"].to(device)
            obs = {k: batch[k].to(device) for k in OBS_KEYS}
            preds = [x]  # group 0 = background
            for si in (0, 1):
                preds.append(models["real"][si](x, obs)["analysis"])
            syn_batch = multivar_collate([datasets["syn"][i]])
            syn_obs = {k: syn_batch[k].to(device) for k in OBS_KEYS}
            for si in (0, 1):
                preds.append(models["syn"][si](x, syn_obs)["analysis"])
            preds = torch.stack(preds)  # (G,1,82,H,W)
            err = (preds.double() - y.double()) * torch.tensor(phys_std, device=device).double()[None, None, :, None, None]

            latg = batch["grid_latitude"].numpy().astype(np.float64)
            long = batch["grid_longitude"].numpy().astype(np.float64)
            locs = batch["obs_locations"][0].numpy().astype(np.float64)
            valid_obs = batch["observation_mask"][0].numpy().astype(bool)
            locs = locs[valid_obs]

            # --- distance-to-nearest-observation bins (grid cells) ---
            tree = cKDTree(unit_xyz(locs[:, 0], locs[:, 1]))
            glat, glon = np.meshgrid(latg, long, indexing="ij")
            chord, _ = tree.query(unit_xyz(glat.ravel(), glon.ravel()), k=1)
            dist_km = 2 * R_EARTH * np.arcsin(np.clip(chord / 2, 0, 1)).reshape(glat.shape)
            bidx = np.digitize(dist_km.ravel(), BINS) - 1
            cosw = np.repeat(np.cos(np.deg2rad(latg))[:, None], len(long), 1).ravel()
            n_dist_cells += np.bincount(bidx, minlength=len(BINS) - 1)
            for ci, ch in enumerate(CH.values()):
                e = err[:, 0, ch].reshape(n_groups, -1).cpu().numpy()
                for b in range(len(BINS) - 1):
                    m = bidx == b
                    if m.any():
                        w = cosw[m]
                        dist_sse[:, b, ci] += (e[:, m] ** 2 * w).sum(1)
                        dist_bias[:, b, ci] += (e[:, m] * w).sum(1)
                        dist_den[b, ci] += w.sum()

            # --- station-position stratification: context obs vs heldout obs ---
            # Field error (analysis - reference) at station positions, bilinear.
            # Heldout stations never enter the DA input (separate heldout_obs_* fields).
            role_locs = [batch["obs_locations"][0].numpy().astype(np.float64),
                         batch["heldout_obs_locations"][0].numpy().astype(np.float64)]
            for r_id, loc_r in enumerate(role_locs):
                if len(loc_r) == 0:
                    continue
                rl, rn = loc_r[:, 0], loc_r[:, 1]
                st_n[r_id] += len(loc_r)
                for ci, ch in enumerate(CH.values()):
                    e = err[:, 0, ch].cpu().numpy()  # (G,H,W)
                    for g in range(n_groups):
                        d = sample_background_at_obs(e[g], latg, long, rl, rn)
                        ok = np.isfinite(d)
                        st_bias[g, r_id, ci] += d[ok].sum()
                        st_sse[g, r_id, ci] += (d[ok] ** 2).sum()
            if (i // args.stride + 1) % 60 == 0:
                print(f"{i + 1}/{len(range(0, len(datasets['real']), args.stride))} cycles", flush=True)

    # finalize station denominators (finite counts equal across groups; recount)
    results = {"cycles": len(range(0, len(datasets["real"]), args.stride)), "stride": args.stride,
               "groups": ["background", "real_s42", "real_s43", "syn_s42", "syn_s43"],
               "channels": list(CH), "bins_km": [list(BINS[:-1]) + ["inf"]],
               "dist_rmse": {}, "dist_bias": {}, "dist_gridcell_share": (n_dist_cells / n_dist_cells.sum()).tolist(),
               "station_rmse": {}, "station_bias": {}, "station_n": st_n.tolist()}
    for ci, name in enumerate(CH):
        rm = np.sqrt(dist_sse[:, :, ci] / dist_den[None, :, ci])
        results["dist_rmse"][name] = rm.round(5).tolist()
        results["dist_bias"][name] = (dist_bias[:, :, ci] / dist_den[None, :, ci]).round(5).tolist()
        # station: divide by finite count per role (~st_n)
        for r_id, rname in enumerate(role_names):
            n_fin = max(1, st_n[r_id])
            results["station_rmse"].setdefault(name, {})[rname] = np.sqrt(st_sse[:, r_id, ci] / n_fin).round(5).tolist()
            results["station_bias"].setdefault(name, {})[rname] = (st_bias[:, r_id, ci] / n_fin).round(5).tolist()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    lines = ["# Step-3 spatial stratification (2021)", ""]
    for name in CH:
        lines += [f"## {name}: RMSE by distance-to-nearest-observation", "",
                  "| group | <300km | 300-600 | 600-1000 | >1000km | global |", "|---|---:|---:|---:|---:|---:|"]
        rm = results["dist_rmse"][name]
        for gi, g in enumerate(results["groups"]):
            row = rm[gi]
            glob = float(np.sqrt(dist_sse[gi, :, (list(CH)).index(name)].sum() / dist_den[:, (list(CH)).index(name)].sum()))
            lines.append(f"| {g} | {row[0]:.4f} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.4f} | {glob:.4f} |")
        lines.append("")
    lines += ["## T2M RMSE at station positions (role split)", "",
              "| group | train stations | heldout stations |", "|---|---:|---:|"]
    for gi, g in enumerate(results["groups"]):
        t = results["station_rmse"]["t2m"]["train"][gi]
        h = results["station_rmse"]["t2m"]["heldout"][gi]
        lines.append(f"| {g} | {t:.4f} | {h:.4f} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
