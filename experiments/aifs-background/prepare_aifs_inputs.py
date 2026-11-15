#!/usr/bin/env python3
"""Extract AIFS-1.1 initial-condition inputs from ARCO v3 to local 0.25 deg shards.

AIFS-single-1.1 consumes 90 prognostic channels (verified from checkpoint):
12 surface + 6 vars x 13 levels. All are available in ARCO v3
(gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3).

Storage: one zstd-7 compressed .npy.zst per 6h timestep (lossless, ~1.8x), shape
(90, 721, 1440) float32, latitude 90->-90, longitude 0->359.75 (ARCO native
layout, same convention as the official notebook's Open Data path).
~291 GB/year, ~1.16 TB for 4 years.

Usage: prepare_aifs_inputs.py --year 2021 [--data-root ../data/aifs_inputs]
Run with the repo .venv (xarray/gcsfs; zarr reads unaffected by the gcsfs bug).
"""
import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import zstd_io
import xarray as xr

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
# anemoi name -> (ARCO variable, level or None), in canonical order (surface first)
MAP = {
    "2t": ("2m_temperature", None),
    "10u": ("10m_u_component_of_wind", None),
    "10v": ("10m_v_component_of_wind", None),
    "2d": ("2m_dewpoint_temperature", None),
    "msl": ("mean_sea_level_pressure", None),
    "sp": ("surface_pressure", None),
    "skt": ("skin_temperature", None),
    "tcw": ("total_column_water_vapour", None),
    "stl1": ("soil_temperature_level_1", None),
    "stl2": ("soil_temperature_level_2", None),
    "swvl1": ("volumetric_soil_water_layer_1", None),
    "swvl2": ("volumetric_soil_water_layer_2", None),
}
P3D = {  # anemoi base -> ARCO 3D variable
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t": "temperature",
    "w": "vertical_velocity",
    "z": "geopotential",
    "q": "specific_humidity",
}


_DS = None


def _open_ds():
    global _DS
    if _DS is None:
        _DS = xr.open_zarr(ARCO, storage_options={"token": "anon"}, consolidated=True)
    return _DS


def _init_worker():
    _open_ds()


def fetch_one(t, out_dir):
    """Download one 6h timestep from ARCO and write its zstd shard. Idempotent + atomic."""
    stem = str(t)[:13].replace("-", "").replace("T", "")
    dest = Path(out_dir) / (stem + ".npy.zst")
    if dest.exists() or (Path(out_dir) / (stem + ".npy")).exists():
        return "skip"
    ds = _open_ds()
    surf_names = [v[0] for v in MAP.values()]
    n_ch = len(MAP) + len(P3D) * len(LEVELS)
    surf = ds[surf_names].sel(time=[t]).load()
    p3d = ds[list(P3D.values())].sel(time=[t], level=LEVELS).load()
    arr = np.empty((n_ch, 721, 1440), dtype=np.float32)
    i = 0
    for name in MAP:
        arr[i] = surf[MAP[name][0]].values[0]
        i += 1
    for b, arco in P3D.items():
        a = p3d[arco].values[0]  # (13, 721, 1440), levels in LEVELS order
        assert a.shape == (13, 721, 1440)
        arr[i:i + 13] = a
        i += 13
    if not np.isfinite(arr).all():
        raise ValueError(f"nonfinite at {t}")
    zstd_io.save_npy_zst(dest, arr)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--start", default=None, help="YYYY-MM-DDThh; with --end, extracts this range instead of the whole year (e.g. 2017-12-31T18 tail for the 2018 train cycle)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DDThh, exclusive")
    ap.add_argument("--data-root", default=str(Path(__file__).resolve().parent / "data" / "aifs_inputs"))
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel timestep fetchers (ARCO is latency-bound: ~84 serial chunk GETs/timestep; workers hide it)")
    args = ap.parse_args()
    out_dir = Path(args.data_root) / str(args.year)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ch = len(MAP) + len(P3D) * len(LEVELS)
    assert n_ch == 90, n_ch

    meta = {"arco": ARCO, "year": args.year, "layout": "float32 (90, 721, 1440), lat 90->-90, lon 0->359.75, zstd-7 .npy.zst",
            "channels": list(MAP.keys()) + [f"{b}_{lv}" for b in P3D for lv in LEVELS],
            "map": {k: v[0] for k, v in MAP.items()}, "map_3d": P3D, "levels": LEVELS,
            "constants": "lsm/sdor/slor/z embedded in checkpoint (not stored here)"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")

    t0 = time.time()
    if args.start or args.end:
        t_start = np.datetime64(args.start or f"{args.year}-01-01T00:00:00", "ns")
        t_end = np.datetime64(args.end or f"{args.year + 1}-01-01T00:00:00", "ns")
        times = list(np.arange(t_start, t_end, np.timedelta64(6, "h")))
    else:
        ndays = 365 if args.year != 2020 else 366
        times = [np.datetime64(f"{args.year}-01-01T00:00:00", "ns") + np.timedelta64(day, "D") + np.timedelta64(hh, "h")
                 for day in range(ndays) for hh in (0, 6, 12, 18)]
    todo = [t for t in times if not (out_dir / (str(t)[:13].replace("-", "").replace("T", "") + ".npy.zst")).exists()]
    print(f"{len(todo)}/{len(times)} to fetch, workers={args.workers}", flush=True)
    done = 0

    def progress():
        nonlocal done
        done += 1
        if done % 100 == 0:
            rate = done / (time.time() - t0)
            print(f"{done}/{len(times)} timesteps, {rate * 3600:.0f}/h", flush=True)

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
            futs = [pool.submit(fetch_one, t, str(out_dir)) for t in times]
            for f in as_completed(futs):
                f.result()
                progress()
    else:
        for t in times:
            fetch_one(t, str(out_dir))
            progress()
    print("DONE", out_dir, flush=True)


if __name__ == "__main__":
    main()
