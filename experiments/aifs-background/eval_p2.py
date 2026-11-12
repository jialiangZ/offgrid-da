#!/usr/bin/env python3
"""P2 interim evaluation: AIFS-1.1 background T2M RMSE vs ERA5 reference (2021).

Pre-registered gate (vs ArchesWeather 2021 anchor 0.441409 K):
  < 0.38  strong pass  -> proceed to P3
  0.38-0.43 partial    -> cautious proceed
  > 0.43  stop-and-record -> negative result, no P3

Usage (.venv of repo root): python eval_p2.py [bg_dir]
Evaluates every state_*.npz found; latitude convention: npz north-first (90->-90),
era5_240 shards ascending (-90..90) -> flip reference.
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

BG = Path(sys.argv[1] if len(sys.argv) > 1 else "data/bg_aifs_2021")
ERA5 = Path("/home/jlzhang/Documents/offgrid-da/experiments/archesweather-reproduction/data/era5_240/full")
ANCHOR = 0.441409  # ArchesWeather 2021 T2M 6h RMSE (250k contract)

shards = {h: xr.open_dataset(ERA5 / f"era5_240_2021_{h}h.nc") for h in (0, 6, 12, 18)}
w = np.cos(np.deg2rad(np.linspace(90, -90, 121)))[:, None]  # north-first weights

files = sorted(BG.glob("state_*.npz"))
assert files, f"no state_*.npz under {BG}"
rmse_list, bias_list = [], []
extra = {"10u": (0, "10m_u_component_of_wind"), "msl": (3, "mean_sea_level_pressure")}
extra_rmse = {k: [] for k in extra}

for f in files:
    stem = f.stem.replace("state_", "")  # YYYYMMDDTHH
    y, m, d, hh = int(stem[:4]), int(stem[4:6]), int(stem[6:8]), int(stem[9:11])
    doy = (np.datetime64(f"{y:04d}-{m:02d}-{d:02d}", "D")
           - np.datetime64(f"{y:04d}-01-01", "D")).astype(int)
    ref = shards[hh]
    d = np.load(f)
    assert d["latitude"][0] > d["latitude"][-1], "expected north-first npz"

    def ref_field(var, level=None):
        da = ref[var].isel(time=doy)
        if level is not None:
            da = da.isel(level=level)
        return np.flipud(da.transpose("latitude", "longitude").values.astype(np.float64))

    err = d["surface"][2, 0].astype(np.float64) - ref_field("2m_temperature")
    rmse_list.append(float(np.sqrt((err**2 * w).mean())))
    bias_list.append(float(err.mean()))
    for k, (idx, var) in extra.items():
        e = d["surface"][idx, 0].astype(np.float64) - ref_field(var)
        extra_rmse[k].append(float(np.sqrt((e**2 * w).mean())))

r = np.array(rmse_list)
print(f"n timesteps: {len(r)}  (Jan-Apr 2021 interim)")
print(f"T2M RMSE: mean {r.mean():.4f} K | median {np.median(r):.4f} | p10 {np.percentile(r,10):.4f} | p90 {np.percentile(r,90):.4f}")
print(f"bias: mean {np.mean(bias_list):+.4f} K")
for k, v in extra_rmse.items():
    print(f"{k} RMSE: mean {np.mean(v):.4f}")
print(f"\nanchor (ArchesWeather 2021): {ANCHOR}")
verdict = "STRONG PASS (<0.38)" if r.mean() < 0.38 else ("PARTIAL (0.38-0.43)" if r.mean() < 0.43 else "STOP-AND-RECORD (>0.43)")
monthly = {}
for f, rr in zip(files, rmse_list):
    monthly.setdefault(f.stem[10:12], []).append(rr)
print("monthly means:", {k: round(float(np.mean(v)), 4) for k, v in sorted(monthly.items())})
print(f"interim verdict: {verdict}")
