from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "era5_240" / "full"
OUTPUT = ROOT / "stats" / "era5_norm_stats_1998_2017.pt"
YEARS = range(1998, 2018)
HOURS = (0, 6, 12, 18)
CHUNK = 16
LEVEL_VARS = (
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "vertical_velocity",
)
SURFACE_VARS = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
)
PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

def init_acc():
    return {"sum": None, "sumsq": None, "count": 0}

def update(acc, values, axes):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite value encountered in training data")
    v = values.astype(np.float64, copy=False)
    s = v.sum(axis=axes, dtype=np.float64)
    q = np.square(v, dtype=np.float64).sum(axis=axes, dtype=np.float64)
    if acc["sum"] is None:
        acc["sum"] = np.zeros_like(s, dtype=np.float64)
        acc["sumsq"] = np.zeros_like(q, dtype=np.float64)
    acc["sum"] += s
    acc["sumsq"] += q
    sample_count = values.shape[0]
    acc["count"] += sample_count * values.shape[-2] * values.shape[-1]

def moments(acc):
    mean = acc["sum"] / acc["count"]
    variance = np.maximum(acc["sumsq"] / acc["count"] - mean * mean, 0.0)
    return mean, np.sqrt(variance)

level = {name: init_acc() for name in LEVEL_VARS}
surface = {name: init_acc() for name in SURFACE_VARS}
level_delta = {name: init_acc() for name in LEVEL_VARS}
surface_delta = {name: init_acc() for name in SURFACE_VARS}

for year in YEARS:
    days = 366 if calendar.isleap(year) else 365
    for hour in HOURS:
        path = DATA / f"era5_240_{year}_{hour}h.nc"
        if not path.is_file() or path.stat().st_size < 4_580_000_000:
            raise FileNotFoundError(f"Missing or incomplete training shard: {path}")
        print(f"processing {year} {hour:02d}Z", flush=True)
        with xr.open_dataset(path, engine="netcdf4") as ds:
            if ds.sizes.get("time") != days:
                raise ValueError(f"Unexpected sample count in {path.name}")
            for name in LEVEL_VARS + SURFACE_VARS:
                target = level[name] if name in level else surface[name]
                shape_axes = (0, 2, 3) if name in level else (0, 1, 2)
                n = ds.sizes["time"]
                for start in range(0, n, CHUNK):
                    stop = min(start + CHUNK, n)
                    block = ds[name].isel(time=slice(start, stop)).to_numpy()
                    update(target, block, shape_axes)
                delta_target = level_delta[name] if name in level_delta else surface_delta[name]
                for start in range(0, n - 1, CHUNK):
                    stop = min(start + CHUNK + 1, n)
                    block = ds[name].isel(time=slice(start, stop)).to_numpy()
                    differences = block[1:] - block[:-1]
                    update(delta_target, differences, shape_axes)

# Include the 24-hour pair crossing each training-year boundary.
for year in range(1998, 2017):
    for hour in HOURS:
        first_path = DATA / f"era5_240_{year}_{hour}h.nc"
        next_path = DATA / f"era5_240_{year + 1}_{hour}h.nc"
        with xr.open_dataset(first_path, engine="netcdf4") as a, xr.open_dataset(next_path, engine="netcdf4") as b:
            for name in LEVEL_VARS + SURFACE_VARS:
                axes = (0, 2, 3) if name in level_delta else (0, 1, 2)
                last = a[name].isel(time=-1).to_numpy()[None, ...]
                first = b[name].isel(time=0).to_numpy()[None, ...]
                diff = first - last
                target = level_delta[name] if name in level_delta else surface_delta[name]
                update(target, diff, axes)

level_means, level_stds, level_delta_stds = [], [], []
for name in LEVEL_VARS:
    mean, std = moments(level[name])
    _, delta_std = moments(level_delta[name])
    level_means.append(mean)
    level_stds.append(std)
    level_delta_stds.append(delta_std)
surface_means, surface_stds, surface_delta_stds = [], [], []
for name in SURFACE_VARS:
    mean, std = moments(surface[name])
    _, delta_std = moments(surface_delta[name])
    surface_means.append(mean)
    surface_stds.append(std)
    surface_delta_stds.append(delta_std)

stats = {
    "surface_mean": torch.tensor(np.asarray(surface_means), dtype=torch.float32).reshape(4, 1, 1, 1),
    "surface_std": torch.tensor(np.asarray(surface_stds), dtype=torch.float32).reshape(4, 1, 1, 1),
    "level_mean": torch.tensor(np.asarray(level_means), dtype=torch.float32).reshape(6, 13, 1, 1),
    "level_std": torch.tensor(np.asarray(level_stds), dtype=torch.float32).reshape(6, 13, 1, 1),
    "surface_delta_std": torch.tensor(np.asarray(surface_delta_stds), dtype=torch.float32).reshape(4, 1, 1, 1),
    "level_delta_std": torch.tensor(np.asarray(level_delta_stds), dtype=torch.float32).reshape(6, 13, 1, 1),
    "metadata": {"years": [1998, 2017], "sampling": "6-hourly", "grid": [121, 240], "reduction": "unweighted time-lat-lon population moments"},
}
if (stats["surface_std"] <= 0).any() or (stats["level_std"] <= 0).any() or (stats["surface_delta_std"] <= 0).any() or (stats["level_delta_std"] <= 0).any():
    raise ValueError("Found zero standard deviation")
torch.save(stats, OUTPUT)
print(f"saved {OUTPUT}")
for key in ("surface_mean", "surface_std", "surface_delta_std", "level_std", "level_delta_std"):
    print(key, tuple(stats[key].shape), stats[key].flatten()[:6].tolist())
