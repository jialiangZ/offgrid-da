#!/usr/bin/env python3
"""Generate DA background fields (82ch, 240x121, physical units) from AIFS-single-1.1.

For every 6h valid time t in the extracted input set:
  1. load local ARCO shards (t-6h, t): 90ch x 0.25deg each
  2. regrid 0.25 -> N320 (official earthkit-regrid linear weights, batched)
  3. SimpleRunner (aifs-single-mse-1.1.ckpt) 6h forecast from t's perspective
     (input fields = stacked [t-6h, t] per anemoi multistep convention)
  4. regrid N320 -> 1.5deg (earthkit-regrid linear weights, batched, 82ch)
  5. write state_YYYYMMDDThh.npz in the exact format of
     outputs/single_da_05_input82_ablation/backgrounds/ (surface/level/lat/lon/pressure)

Manifest mirrors the existing schema with AIFS provenance fields.

Usage (aifs .venv): python run_aifs_background.py --inputs data/aifs_inputs/2021 \
    --out ../../outputs/bg_aifs_val_2021 [--device cuda] [--start ...] [--end ...]
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from earthkit.regrid.db import find as find_matrix

HERE = Path(__file__).resolve().parent
CKPT = str(HERE / "v1.1" / "aifs-single-mse-1.1.ckpt")  # run with v1.1/.venv
REF_NPZ = Path("/home/jlzhang/Documents/offgrid-da/outputs/single_da_05_input82_ablation/backgrounds/state_20180101T00.npz")

SURFACE = ["10u", "10v", "2t", "msl"]                      # -> npz surface rows
LEVEL_VARS = ["z", "u", "v", "t", "q", "w"]                # -> npz level rows (era5_240 order)
PRESSURES = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
CONSTANTS = {"lsm": "land_sea_mask", "z": "geopotential_at_surface",
             "slor": "slope_of_sub_gridscale_orography", "sdor": "standard_deviation_of_orography"}


def ensure_constants(cache: Path) -> np.ndarray:
    """Fetch the 4 constant fields once from ARCO (0.25 deg), shape (4, 721, 1440)."""
    if cache.exists():
        return np.load(cache)
    import xarray as xr
    ds = xr.open_zarr(ARCO, storage_options={"token": "anon"}, consolidated=True)
    t = "2021-01-01T00:00:00"
    arr = np.stack([ds[v].sel(time=[t]).values[0] for v in CONSTANTS.values()]).astype(np.float32)
    ds.close()
    assert arr.shape == (4, 721, 1440) and np.isfinite(arr).all()
    np.save(cache, arr)
    return arr
IN_CHANNELS = json.loads((HERE / "data" / "aifs_inputs" / "meta_channels.json").read_text()) \
    if (HERE / "data" / "aifs_inputs" / "meta_channels.json").exists() else None


def input_channel_names():
    """Order of the 90 channels in the extracted .npy shards (prepare_aifs_inputs.py)."""
    surf = ["2t", "10u", "10v", "2d", "msl", "sp", "skt", "tcw", "stl1", "stl2", "swvl1", "swvl2"]
    return surf + [f"{b}_{lv}" for b in ("u", "v", "t", "w", "z", "q") for lv in PRESSURES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="dir with YYYYMMDDHH.npy shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    in_dir = Path(args.inputs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from anemoi.inference.runners.simple import SimpleRunner
    import anemoi.inference.runner as _R

    _orig_run = _R.Runner.run

    def _run_bf16(self, *a, **k):
        # anemoi-inference 0.6.3 defines Runner.autocast but never applies it;
        # without this the 1.1 model forward runs fp32 and OOMs at 24 GB.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return _orig_run(self, *a, **k)

    _R.Runner.run = _run_bf16
    runner = SimpleRunner(CKPT, device=args.device)

    z_in, _ = find_matrix({"grid": (0.25, 0.25)}, {"grid": "N320"}, "linear")
    z_out, out_shape = find_matrix({"grid": "N320"}, {"grid": (1.5, 1.5)}, "linear")

    consts = ensure_constants(in_dir.parent / "constants_0p25.npy")
    consts_n320 = (z_in @ consts.reshape(4, -1).T).T  # (4, 542080)
    const_fields = {name: np.stack([v, v]) for name, v in zip(CONSTANTS, consts_n320)}

    ref = np.load(REF_NPZ)
    lat, lon = ref["latitude"], ref["longitude"]
    pressure_hpa = ref["pressure_hpa"]
    np.testing.assert_array_equal(pressure_hpa, PRESSURES)

    names = input_channel_names()
    out_names = SURFACE + [f"{b}_{lv}" for b in LEVEL_VARS for lv in PRESSURES]

    files = sorted(in_dir.glob("??????????.npy"))
    if args.limit:
        files = files[:args.limit]
    t0 = time.time()
    for k, f in enumerate(files):
        stamp = f.stem  # YYYYMMDDHH = valid time t
        import datetime as dt
        t = dt.datetime(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]), int(stamp[8:10]))
        # The 6h forecast from inputs [t-6h, t] is valid at t+6h: this is x_b(t+6h).
        dest = out_dir / f"state_{(t + dt.timedelta(hours=6)).strftime('%Y%m%dT%H')}.npz"
        if dest.exists():
            continue
        t_prev = t - dt.timedelta(hours=6)
        f_prev = in_dir / (t_prev.strftime("%Y%m%d%H") + ".npy")
        if not f_prev.exists():
            print("SKIP (missing t-6h):", f_prev.name, flush=True)
            continue
        cur = np.load(f, mmap_mode="r")
        prev = np.load(f_prev, mmap_mode="r")
        flat = np.empty((180, 721 * 1440), dtype=np.float32)
        flat[0::2] = np.asarray(prev, dtype=np.float32).reshape(90, -1)
        flat[1::2] = np.asarray(cur, dtype=np.float32).reshape(90, -1)
        n320 = (z_in @ flat.T).T  # (180, 542080): row pairs [t-6h, t] per channel
        fields = {name: n320[2 * i:2 * i + 2].copy() for i, name in enumerate(names)}
        fields.update(const_fields)
        state = dict(date=t, fields=fields)
        result = list(runner.run(input_state=state, lead_time="6h"))[-1]
        out_fields = result["fields"]
        arr = np.stack([np.asarray(out_fields[n], dtype=np.float32) for n in out_names])  # (82, 542080)
        grid = (z_out @ arr.T).T.reshape(82, *out_shape)
        surface = grid[:4].reshape(4, 1, 121, 240)
        level = grid[4:].reshape(6, 13, 121, 240)
        if not np.isfinite(grid).all():
            raise ValueError(f"nonfinite forecast at {stamp}")
        np.savez_compressed(dest, surface=surface, level=level,
                            latitude=lat, longitude=lon, pressure_hpa=pressure_hpa)
        if (k + 1) % 20 == 0:
            rate = (k + 1) / (time.time() - t0)
            print(f"{k+1}/{len(files)} ({rate*3600:.0f}/h) last={dest.name}", flush=True)

    manifest = {
        "schema": "single_da_05 background npz (surface 4x1x121x240, level 6x13x121x240, physical units)",
        "model": "aifs-single-1.1 (ecmwf/aifs-single-1.1, CC BY 4.0)",
        "checkpoint": CKPT,
        "checkpoint_sha256": hashlib.sha256(Path(CKPT).read_bytes()).hexdigest(),
        "inputs": str(in_dir),
        "regrid": "earthkit-regrid linear: 0.25->N320 (input), N320->1.5 (output)",
        "generator": str(Path(__file__).resolve()),
        "generated_at_unix": time.time(),
        "channel_names": out_names,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("DONE", out_dir, flush=True)


if __name__ == "__main__":
    main()
