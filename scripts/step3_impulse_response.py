#!/usr/bin/env python3
"""Step-3 impulse-response probe: empirical receptive field of trained DA models.

For one validation cycle, all observations are set to zero innovation
(value = H(x_b)); a single T2M station then receives a +1 std (normalized)
pulse. The analysis increment field (pulse minus base, physical K) maps the
network's actual spatial response to one observation, comparable against the
300 km Gaussian kernel that won in the local-baseline experiment.

Reports peak |increment|, the radius where binned |increment| falls below 10%
of peak, share of |increment| mass within 300/1000 km, and local observation
density context. Two stations: dense Europe (50N, 8.5E) and sparse Pacific
(0N, 150W). Both seed-42 models (real and synthetic arms).

Outputs: outputs/step3_impulse/{impulse.json, fields.npz}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from step3_obs_content_diag import (load_model, zero_innov_batch, MODELS,  # noqa: E402
                                    OBS_KEYS)
from offgrid_da.data.multivar_dataset import multivar_collate  # noqa: E402
import train_step2_synthetic as syn_module  # noqa: E402

STATIONS = {"europe_dense": (50.0, 8.5), "pacific_sparse": (0.0, -150.0)}
R_EARTH = 6371.0


def haversine_1d(lats, lons, lat0, lon0):
    la = np.deg2rad(np.asarray(lats, dtype=float))
    lo = np.deg2rad(np.asarray(lons, dtype=float))
    la0, lo0 = np.deg2rad(lat0), np.deg2rad(lon0)
    a = np.sin((la - la0) / 2) ** 2 + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def haversine_grid(lat, lon, lat0, lon0):
    la = np.deg2rad(lat)[:, None]
    lo = np.deg2rad(lon)[None, :]
    la0, lo0 = np.deg2rad(lat0), np.deg2rad(lon0)
    a = np.sin((la - la0) / 2) ** 2 + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main():
    device = torch.device("cuda:0")
    ds = syn_module.SyntheticData(ROOT, "val")
    sample = ds[2]  # a stable mid-January 2021 cycle
    batch = multivar_collate([sample])
    valid_time = str(sample["valid_time"])
    print("cycle:", valid_time)

    base_cpu = zero_innov_batch({"grid_latitude": batch["grid_latitude"],
                                 "grid_longitude": batch["grid_longitude"],
                                 "background_normalized": batch["background_normalized"],
                                 "obs_variables": batch["obs_variables"],
                                 "obs_locations": batch["obs_locations"],
                                 "obs_values_normalized": batch["obs_values_normalized"]}, ds)

    lat = batch["grid_latitude"].numpy().astype(float)
    lon = batch["grid_longitude"].numpy().astype(float)
    var_ids = batch["obs_variables"][0].numpy()
    locs = batch["obs_locations"][0].numpy().astype(float)
    std = ds.base.std
    mean = ds.base.mean

    t2m_idx = np.where(var_ids == 0)[0]
    results = {"cycle": valid_time, "pulse_std_normalized": 1.0}
    fields = {}
    for arm in ("real", "syn"):
        model = load_model(MODELS[arm][0], device)
        x = batch["background_normalized"].to(device)
        with torch.inference_mode():
            obs_base = {k: v.to(device) for k, v in {**{k: batch[k] for k in OBS_KEYS},
                       "obs_values_normalized": base_cpu["obs_values_normalized"]}.items()}
            a_base = model(x, obs_base)["analysis"]
            for name, (tlat, tlon) in STATIONS.items():
                d = haversine_1d(locs[t2m_idx, 0], locs[t2m_idx, 1], tlat, tlon)
                j = t2m_idx[int(np.argmin(d))]
                n_within_1000 = int((d < 1000).sum())
                obs_pulse = {k: v.clone() for k, v in obs_base.items()}
                obs_pulse["obs_values_normalized"][0, j] += 1.0
                a_pulse = model(x, obs_pulse)["analysis"]
                incr = ((a_pulse - a_base)[0, 2].double() * float(std[2])).cpu().numpy()
                dist = haversine_grid(lat, lon, locs[j, 0], locs[j, 1])
                peak = float(np.abs(incr).max())
                bins = np.arange(0, 3001, 100)
                prof = []
                for b0 in bins[:-1]:
                    m = (dist >= b0) & (dist < b0 + 100)
                    prof.append(float(np.abs(incr)[m].mean()) if m.any() else 0.0)
                prof = np.array(prof)
                below = np.where(prof < 0.10 * peak)[0]
                r10 = float(bins[below[0]]) if len(below) else float("nan")
                cosw = np.repeat(np.cos(np.deg2rad(lat))[:, None], len(lon), 1)
                mass = np.abs(incr) * cosw
                tot = mass.sum()
                key = f"{arm}_{name}"
                fields[key] = incr.astype(np.float32)
                results[key] = {
                    "station": [float(locs[j, 0]), float(locs[j, 1])],
                    "n_t2m_obs_within_1000km": n_within_1000,
                    "peak_increment_K": peak,
                    "r10_km": r10,
                    "mass_within_300km": float(mass[dist < 300].sum() / tot),
                    "mass_within_1000km": float(mass[dist < 1000].sum() / tot),
                    "profile_100km_bins": np.round(prof, 6).tolist(),
                }
                print(key, json.dumps({k: v for k, v in results[key].items() if k != "profile_100km_bins"}))

    out = Path("outputs/step3_impulse")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "fields.npz", **fields)
    (out / "impulse.json").write_text(json.dumps(results, indent=1) + "\n")
    print("written:", out)


if __name__ == "__main__":
    main()
