#!/usr/bin/env python3
"""Extract AIFS-1.1 initial-condition inputs from ARCO v3 to local 0.25 deg shards.

AIFS-single-1.1 consumes 90 prognostic channels (verified from checkpoint):
12 surface + 6 vars x 13 levels. All are available in ARCO v3
(gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3).

Storage: one uncompressed .npy per 6h timestep, shape (90, 721, 1440) float32,
latitude 90->-90, longitude 0->359.75 (ARCO native layout, same convention as
the official notebook's Open Data path). ~542 GB for 4 years, ~136 GB/year.

Usage: prepare_aifs_inputs.py --year 2021 [--data-root ../data/aifs_inputs]
Run with the repo .venv (xarray/gcsfs; zarr reads unaffected by the gcsfs bug).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--data-root", default=str(Path(__file__).resolve().parent / "data" / "aifs_inputs"))
    args = ap.parse_args()
    out_dir = Path(args.data_root) / str(args.year)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ch = len(MAP) + len(P3D) * len(LEVELS)
    assert n_ch == 90, n_ch

    meta = {"arco": ARCO, "year": args.year, "layout": "float32 (90, 721, 1440), lat 90->-90, lon 0->359.75",
            "channels": list(MAP.keys()) + [f"{b}_{lv}" for b in P3D for lv in LEVELS],
            "map": {k: v[0] for k, v in MAP.items()}, "map_3d": P3D, "levels": LEVELS,
            "constants": "lsm/sdor/slor/z embedded in checkpoint (not stored here)"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")

    ds = xr.open_zarr(ARCO, storage_options={"token": "anon"}, consolidated=True)
    surf_names = [v[0] for v in MAP.values()]
    t0 = time.time()
    ndays = 365 if args.year != 2020 else 366
    for day in range(ndays):
        base = np.datetime64(f"{args.year}-01-01T00:00:00", "ns") + np.timedelta64(day, "D")
        for hh in (0, 6, 12, 18):
            t = base + np.timedelta64(hh, "h")
            dest = out_dir / f"{str(t)[:13].replace('-', '').replace('T', '')}.npy"
            if dest.exists():
                continue
            surf = ds[surf_names].sel(time=[t]).load()
            p3d = ds[list(P3D.values())].sel(time=[t], level=LEVELS).load()
            arr = np.empty((n_ch, 721, 1440), dtype=np.float32)
            i = 0
            for name in MAP:
                arr[i] = surf[MAP[name][0]].values[0]; i += 1
            for b, arco in P3D.items():
                a = p3d[arco].values[0]  # (13, 721, 1440), levels in LEVELS order
                assert a.shape == (13, 721, 1440)
                arr[i:i + 13] = a; i += 13
            if not np.isfinite(arr).all():
                raise ValueError(f"nonfinite at {t}")
            tmp = dest.with_suffix(".tmp.npy")
            np.save(tmp, arr)
            tmp.replace(dest)
        if (day + 1) % 15 == 0:
            gb = sum(f.stat().st_size for f in out_dir.glob("*.npy")) / 2**30
            rate = (day + 1) * 4 / (time.time() - t0)
            print(f"day {day+1}/{ndays}: {gb:.1f} GiB, {rate*3600:.0f} timesteps/h", flush=True)
    print("DONE", out_dir, flush=True)


if __name__ == "__main__":
    main()
