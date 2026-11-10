#!/usr/bin/env python3
"""Step-2 simple baseline: local Gaussian innovation correction (plan §4).

For each surface variable (T2M/U10/V10/MSLP), the analysis increment is
    x_a = x_b + gain * sum_i w_i (y_i - H(x_b)_i) / (sum_i w_i + eps),
    w_i = exp(-(d_i / R)^2 / 2)   (great-circle distance, gaussian radius R).

Arms: real METAR values, and synthetic y_i = H(x_ref) (noiseless).

Because the increment field is linear in `gain`, the area-weighted SSE of a
surface channel is an exact quadratic in gain: SSE(g) = A - 2 g B + g^2 C with
A = sum w (bg-ref)^2, B = sum w (bg-ref) f, C = sum w f^2. Workers return only
(A, B, C) per (variable, radius, arm), so tuning over radius and gain is exact
and nearly free. For radii >= 1200 km the increment field is computed on a
strided grid and bilinearly upsampled (fields are smooth at that scale).

Tuning uses a subsample of training cycles (2018-2020) only; evaluation is the
full 2021 validation year. Non-surface channels stay at background values.
Diagnostic probe, not a claimed optimal traditional DA.

Outputs: outputs/step2_local_baseline/{tuning.json, results.json, report.md}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from offgrid_da.data.alignment import sample_background_at_obs  # noqa: E402

SURFACE = ["10m_u_component_of_wind", "10m_v_component_of_wind",
           "2m_temperature", "mean_sea_level_pressure"]
LEVEL = ["geopotential", "u_component_of_wind", "v_component_of_wind",
         "temperature", "specific_humidity", "vertical_velocity"]
PRESSURE = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
CHANNELS = SURFACE + [f"{v}@{p}hPa" for v in LEVEL for p in PRESSURE]
VAR_TO_CHANNEL = {0: 2, 1: 0, 2: 1, 4: 3}  # obs var id -> channel index
# ids per SPECS: 0=T2M(ch2), 1=U10(ch0), 2=V10(ch1), 3=dewpoint(excluded), 4=MSLP(ch3)
VAR_NAME = {0: "2m_temperature", 1: "10m_u_component_of_wind",
            2: "10m_v_component_of_wind", 4: "mean_sea_level_pressure"}
MANIFEST = ROOT / "outputs/single_da_05_input82_ablation/backgrounds/manifest.json"
OBS_DIR = ROOT / ("outputs/metar_multivar_cache/82d9f3badec8c492d7b80f17"
                  "994d12cd9352e4875f2ad2753bda22c572a93c12")
NORM = ROOT / "outputs/single_da_05_input82_ablation/normalization.json"
R_EARTH_KM = 6371.0
RADII_KM = [300.0, 600.0, 1200.0, 2400.0]
GAINS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
W_EPS = 1e-3


@lru_cache(maxsize=8)
def opened_nc(path):
    import xarray as xr
    return xr.open_dataset(path)


def unit_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def load_cycle(row):
    d = np.load(MANIFEST.parent / row["file"], allow_pickle=False)
    surface = d["surface"][:, 0]
    level = d["level"]
    background = np.concatenate([surface, level.reshape(6 * 13, *surface.shape[1:])], 0)
    t = row["t"]
    ds = opened_nc(str(ROOT / f"experiments/archesweather-reproduction/data/era5_240/full/era5_240_{t[0]}_{t[1]}h.nc"))
    ds = ds.sel(time=np.datetime64(t[2])).sortby("latitude", ascending=False).sortby("longitude")
    a = [ds[v].transpose("latitude", "longitude").values for v in SURFACE]
    for v in LEVEL:
        for p in PRESSURE:
            a.append(ds[v].sel(level=p).transpose("latitude", "longitude").values)
    reference = np.stack(a)
    return background.astype(np.float64), reference.astype(np.float64)


def gaussian_field(grid_lat, grid_lon, st_xyz, innov, radius_km):
    """Gaussian-weighted innovation field on the (lat × lon) grid given."""
    from scipy.spatial import cKDTree

    lat2, lon2 = np.meshgrid(grid_lat, grid_lon, indexing="ij")
    pts = unit_xyz(lat2.ravel(), lon2.ravel())
    tree = cKDTree(st_xyz)
    theta_max = 3 * radius_km / R_EARTH_KM
    chord = 2.0 * np.sin(min(theta_max, np.pi) / 2.0)
    gtree = cKDTree(pts)
    pairs = gtree.query_ball_tree(tree, r=chord)
    num = np.zeros(len(pts))
    den = np.zeros(len(pts))
    for i, js in enumerate(pairs):
        if not js:
            continue
        js = np.asarray(js)
        chord_d = np.linalg.norm(pts[i] - st_xyz[js], axis=1)
        d_km = 2.0 * np.arcsin(np.clip(chord_d / 2.0, 0, 1)) * R_EARTH_KM
        w = np.exp(-0.5 * (d_km / radius_km) ** 2)
        num[i] = (w * innov[js]).sum()
        den[i] = w.sum()
    field = np.where(den > W_EPS, num / np.maximum(den, W_EPS), 0.0)
    return field.reshape(len(grid_lat), len(grid_lon))


def upsample(coarse, full_shape):
    """Bilinear upsample a (H', W') field to full grid (pure numpy)."""
    from scipy.ndimage import zoom
    return zoom(coarse, (full_shape[0] / coarse.shape[0],
                         full_shape[1] / coarse.shape[1]), order=1)


def process_cycle(payload):
    row_json, = payload
    row = json.loads(row_json)
    import pandas as pd
    t = pd.Timestamp(row["t"][2])
    row = {"t": (row["t"][0], row["t"][1], t), "file": row["file"]}
    try:
        import torch
        s = torch.load(OBS_DIR / (t.strftime("%Y%m%dT%H") + ".pt"),
                       map_location="cpu", weights_only=False)
    except FileNotFoundError:
        return None
    vars_ = s["obs_variables"].numpy()
    vals = s["obs_values"].numpy().astype(np.float64)
    locs = s["obs_locations"].numpy().astype(np.float64)
    keep = np.isfinite(vals) & np.isin(vars_, [0, 1, 2, 4])
    vars_, vals, locs = vars_[keep], vals[keep], locs[keep]
    if not keep.any():
        return None
    background, reference = load_cycle(row)
    lat = np.linspace(90, -90, background.shape[1])
    lon = np.arange(240) * 1.5
    glat, glon = np.meshgrid(lat, lon, indexing="ij")
    grid_xyz = unit_xyz(glat.ravel(), glon.ravel())
    cosw = np.repeat(np.cos(np.deg2rad(lat))[:, None], 240, axis=1)
    err = background - reference  # (82,H,W)
    out = {
        "A_bg": [float((err[ch] ** 2 * cosw).sum()) for ch in range(82)],
        "cosw": float(cosw.sum()),
        "per_var": {}, "n_obs": int(keep.sum()),
    }
    for v, ch in VAR_TO_CHANNEL.items():
        m = vars_ == v
        if m.sum() < 10:
            continue
        hb = sample_background_at_obs(background[ch], lat, lon, locs[m, 0], locs[m, 1])
        hr = sample_background_at_obs(reference[ch], lat, lon, locs[m, 0], locs[m, 1])
        good = np.isfinite(hb) & np.isfinite(hr)
        if good.sum() < 10:
            continue
        st_xyz = unit_xyz(locs[m][:, 0][good], locs[m][:, 1][good])
        innov = {"real": vals[m][good] - hb[good], "syn": hr[good] - hb[good]}
        out["per_var"][str(v)] = {"n": int(good.sum())}
        for radius in RADII_KM:
            stride = max(1, int(radius // 300))
            lat_c = lat[::stride]
            lon_c = lon[::stride]
            for arm, iv in innov.items():
                field_c = gaussian_field(lat_c, lon_c, st_xyz, iv, radius)
                field = upsample(field_c, cosw.shape) if stride > 1 else field_c
                # project err of this channel onto the field for exact quadratic
                e = err[ch]
                B = float((e * field * cosw).sum())
                C = float((field * field * cosw).sum())
                out["per_var"][str(v)][f"{radius}_{arm}"] = [B, C]
    return out


def aggregate(cycles):
    agg = {"A_bg": np.zeros(82), "cosw": 0.0,
           "BC": defaultdict(lambda: np.zeros(2)), "n": 0, "n_obs": 0}
    for c in cycles:
        agg["A_bg"] += np.asarray(c["A_bg"])
        agg["cosw"] += c["cosw"]
        agg["n"] += 1
        agg["n_obs"] += c["n_obs"]
        for v, dd in c["per_var"].items():
            for k, bc in dd.items():
                if k != "n":
                    agg["BC"][f"{v}_{k}"] += np.asarray(bc)
    return agg


def channel_sse(agg, var, radius, arm, gain):
    """Per-channel SSE map after applying corrections to surface channels.

    Analysis applies x_a = x_b + gain * f, so SSE(g) = A + 2 g B + g^2 C for the
    corrected channel (B is typically negative: f tracks ref - bg).
    """
    sse = agg["A_bg"].copy()
    ch = VAR_TO_CHANNEL[var]
    B, C = agg["BC"][f"{var}_{radius}_{arm}"]
    sse[ch] = agg["A_bg"][ch] + 2 * gain * B + gain * gain * C
    return sse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-every", type=int, default=24)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--out", default="outputs/step2_local_baseline")
    args = ap.parse_args()

    import pandas as pd

    manifest = json.loads(MANIFEST.read_text())
    rows = []
    for x in manifest["samples"]:
        t = pd.Timestamp(x["valid_utc"]).tz_convert("UTC").tz_localize(None)
        rows.append({"t": (t.year, t.hour, t.isoformat()), "file": x["file"]})
    train_rows = [r for r in rows if r["t"][0] <= 2020]
    val_rows = [r for r in rows if r["t"][0] == 2021]
    tune_rows = train_rows[:: args.tune_every]
    print(f"cycles: tune={len(tune_rows)}/{len(train_rows)} train, val={len(val_rows)}")

    def run_set(row_list):
        payload = [(json.dumps(r),) for r in row_list]
        with Pool(args.procs) as pool:
            return [x for x in pool.imap_unordered(process_cycle, payload) if x]

    tune_cycles = run_set(tune_rows)
    print(f"tuning cycles processed: {len(tune_cycles)}")
    agg_t = aggregate(tune_cycles)

    # ---- tuning: per (var, arm) choose radius and gain on training subset
    tuning = {}
    for v in VAR_TO_CHANNEL:
        for arm in ("real", "syn"):
            best = None
            for radius in RADII_KM:
                key = f"{v}_{radius}_{arm}"
                if key not in agg_t["BC"]:
                    continue
                B, C = agg_t["BC"][key]
                A = agg_t["A_bg"][VAR_TO_CHANNEL[v]]
                # SSE(g) = A + 2 g B + g^2 C  ->  optimum at g* = -B/C
                g_opt = float(np.clip(-B / C, 0.02, 2.0)) if C > 0 else 0.0
                g_grid = min(GAINS, key=lambda g: A + 2 * g * B + g * g * C)
                for g, tag in ((g_opt, "opt"), (g_grid, "grid")):
                    sse = max(float(A + 2 * g * B + g * g * C), 0.0)
                    cand = (sse, radius, g, tag)
                    if best is None or cand[0] < best[0]:
                        best = cand
            tuning[f"v{v}_{arm}"] = {"radius_km": best[1], "gain": best[2],
                                     "tune_sse": best[0], "mode": best[3]}
    print("tuning:", json.dumps(tuning, indent=1))

    # ---- evaluation on 2021
    val_cycles = run_set(val_rows)
    print(f"validation cycles processed: {len(val_cycles)}")
    agg_v = aggregate(val_cycles)

    norm = json.loads(NORM.read_text())
    std = np.asarray(norm["input_std"], dtype=np.float64)

    def eval_arm(arm, use_tuning):
        sse = agg_v["A_bg"].copy()
        detail = {}
        for v, ch in VAR_TO_CHANNEL.items():
            tcfg = tuning[f"v{v}_{arm}"]
            sse_v = channel_sse(agg_v, v, tcfg["radius_km"], arm, tcfg["gain"])
            sse[ch] = sse_v[ch]
            detail[VAR_NAME[v]] = {
                "radius_km": tcfg["radius_km"], "gain": round(tcfg["gain"], 3),
                "rmse": float(np.sqrt(sse_v[ch] / agg_v["cosw"])),
                "rmse_bg": float(np.sqrt(agg_v["A_bg"][ch] / agg_v["cosw"])),
            }
        rmse_ch = np.sqrt(sse / agg_v["cosw"])
        return {"per_surface_var": detail,
                "physical_rmse_by_channel": {c: float(r) for c, r in zip(CHANNELS, rmse_ch)},
                "group_balanced_score": float(np.sqrt(
                    (rmse_ch[:4] ** 2 / std[:4] ** 2).mean())),  # 4-surface balanced
                }

    results = {
        "cycles": {"tune": agg_t["n"], "val": agg_v["n"], "val_obs": agg_v["n_obs"]},
        "baseline_background": {
            "physical_rmse_by_channel": {c: float(r) for c, r in
                                         zip(CHANNELS, np.sqrt(agg_v["A_bg"] / agg_v["cosw"]))},
            "group_balanced_score": float(np.sqrt(
                (np.sqrt(agg_v["A_bg"][:4] / agg_v["cosw"]) ** 2 / std[:4] ** 2).mean())),
        },
        "real": eval_arm("real", True),
        "synthetic": eval_arm("syn", True),
        "tuning": tuning,
        "radii_km": RADII_KM, "gains_grid": GAINS,
        "notes": "gain tuned continuously per variable on training subsample; "
                 "non-surface channels equal background; area-weighted metrics",
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tuning.json").write_text(json.dumps(tuning, indent=1) + "\n")
    (out_dir / "results.json").write_text(json.dumps(results, indent=1) + "\n")

    lines = ["# Step-2 local innovation baseline (2021)", ""]
    b = results["baseline_background"]["physical_rmse_by_channel"]
    for arm in ("real", "synthetic"):
        lines.append(f"## {arm}")
        for name, dd in results[arm]["per_surface_var"].items():
            lines.append(f"- {name}: R={dd['radius_km']:.0f}km g={dd['gain']:.2f} "
                         f"RMSE {dd['rmse']:.4f} (bg {dd['rmse_bg']:.4f}, "
                         f"{100 * (1 - dd['rmse'] / dd['rmse_bg']):.1f}% better)")
        lines.append(f"- 4-surface balanced score: {results[arm]['group_balanced_score']:.4f} "
                     f"(bg {results['baseline_background']['group_balanced_score']:.4f})")
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("written:", out_dir)


if __name__ == "__main__":
    main()
