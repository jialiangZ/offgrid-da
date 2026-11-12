#!/usr/bin/env python3
"""Decompose the AIFS background deficit: interface (regrid) vs model-vs-reference.

For 24 timesteps (Jan-May 2021, every ~5 days):
  E1  path error (NO MODEL): ARCO truth at t+6h through 0.25->N320->1.5 linear
      vs era5_240 truth            -> pure regrid-chain discrepancy
  E1b stride check: ARCO[::6,::6] vs era5_240  -> provenance of era5_240
  v1  AIFS N320 output -> 1.5 linear (current pipeline)  vs era5_240
  v2  AIFS N320 -> 0.25 -> 6x6 box mean -> 1.5           vs era5_240
      (v1 - v2 = downscale-method sensitivity at 1.5 deg)
  v3  AIFS N320 -> 0.25                                  vs ARCO truth 0.25
      (AIFS native-resolution quality vs its own input lineage)

Run: CUDA_VISIBLE_DEVICES=x ANEMOI_INFERENCE_NUM_CHUNKS=32 v1.1/.venv/bin/python
"""
import datetime as dt
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import anemoi.inference.runner as _R
from anemoi.inference.runners.simple import SimpleRunner
from earthkit.regrid.db import find as find_matrix

HERE = Path(__file__).resolve().parent
IN = HERE / "data" / "aifs_inputs" / "2021"
ERA5 = Path("/home/jlzhang/Documents/offgrid-da/experiments/archesweather-reproduction/data/era5_240/full")
CKPT = str(HERE / "v1.1" / "aifs-single-mse-1.1.ckpt")

PRESS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
NAMES = ["2t", "10u", "10v", "2d", "msl", "sp", "skt", "tcw", "stl1", "stl2", "swvl1", "swvl2"] + \
    [f"{b}_{lv}" for b in ("u", "v", "t", "w", "z", "q") for lv in PRESS]
OUT_NAMES = ["10u", "10v", "2t", "msl"] + [f"{b}_{lv}" for b in ("z", "u", "v", "t", "q", "w") for lv in PRESS]

# npy input channel indices (for truth fields)
T2T, MSL, T850, Z500 = 0, 4, 38 + 10, 64 + 7
# output grid arr rows
O_T2T, O_MSL, O_T850, O_Z500 = 2, 3, 4 + 39 + 10, 4 + 7

_orig_run = _R.Runner.run


def _run_bf16(self, *a, **k):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return _orig_run(self, *a, **k)


STAMPS = [(dt.datetime(2021, 1, 1) + dt.timedelta(days=5 * k)).strftime("%Y%m%d") + ("06" if k % 2 else "18")
          for k in range(24)]  # day 1..116, alternate 06/18Z


def main():
    runner = SimpleRunner(CKPT, device="cuda")
    z_in, _ = find_matrix({"grid": (0.25, 0.25)}, {"grid": "N320"}, "linear")
    z_out, out_shape = find_matrix({"grid": "N320"}, {"grid": (1.5, 1.5)}, "linear")
    z_025, _ = find_matrix({"grid": "N320"}, {"grid": (0.25, 0.25)}, "linear")
    consts = np.load(HERE / "data" / "aifs_inputs" / "constants_0p25.npy")
    cn = (z_in @ consts.reshape(4, -1).T).T
    const_fields = {n: np.stack([v, v]) for n, v in zip(("lsm", "z", "slor", "sdor"), cn)}

    shards = {h: xr.open_dataset(ERA5 / f"era5_240_2021_{h}h.nc") for h in (0, 6, 12, 18)}
    w = np.cos(np.deg2rad(np.linspace(90, -90, 121)))[:, None]
    w25 = np.cos(np.deg2rad(np.linspace(90, -90, 721)))[:, None]
    acc = {k: [] for k in ["path_2t", "path_msl", "path_t850", "path_z500",
                           "stride_2t", "v1_2t", "v1_msl", "v1_t850", "v1_z500",
                           "v2_2t", "v2_msl", "v2_t850", "v2_z500",
                           "v3_2t", "v3_msl", "v3_t850", "v3_z500"]}

    def rmse(x, y, wt):
        e = x.astype(np.float64) - y.astype(np.float64)
        return float(np.sqrt((e**2 * wt).mean()))

    for k, stamp in enumerate(STAMPS):
        t = dt.datetime(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]), int(stamp[8:10]))
        t_out = t + dt.timedelta(hours=6)
        prev = np.load(IN / ((t - dt.timedelta(hours=6)).strftime("%Y%m%d%H") + ".npy"))
        cur = np.load(IN / (stamp + ".npy"))
        truth = np.load(IN / (t_out.strftime("%Y%m%d%H") + ".npy"))  # ARCO 0.25 truth at t+6h

        # --- AIFS forecast ---
        flat = np.empty((180, 721 * 1440), dtype=np.float32)
        flat[0::2] = prev.reshape(90, -1)
        flat[1::2] = cur.reshape(90, -1)
        n320in = (z_in @ flat.T).T
        fields = {n: n320in[2 * i:2 * i + 2].copy() for i, n in enumerate(NAMES)}
        fields.update(const_fields)
        result = list(_run_bf16(runner, input_state=dict(date=t, fields=fields), lead_time="6h"))[-1]
        arr = np.stack([np.asarray(result["fields"][n], np.float32) for n in OUT_NAMES])  # (82, 542080)

        v1 = (z_out @ arr.T).T.reshape(82, *out_shape)                # current 1.5 pipeline
        f025 = (z_025 @ arr.T).T.reshape(82, 721, 1440)                # native-ish 0.25

        def box7_stride6(a):
            """7x7 box mean (covers each 1.5-deg cell) then stride-6 sample to 1.5 deg."""
            p = np.pad(a, ((0, 0), (3, 3), (3, 3)), mode="edge")
            o = np.zeros((a.shape[0], 121, 240), dtype=np.float64)
            for di in range(7):
                for dj in range(7):
                    o += p[:, di:di + 721:6, dj:dj + 1440:6]
            return (o / 49).astype(np.float32)

        v2 = box7_stride6(f025)                                        # box-mean 1.5

        # --- E1: truth through the same regrid chain (input path only, no model) ---
        tn320 = (z_in @ truth.reshape(90, -1).T).T
        def chain1p5(row_idx):
            return (z_out @ tn320[row_idx:row_idx + 1].T).T.reshape(121, 240)
        # --- E1b: stride sample (1.5 deg grid is an exact subset of 0.25 deg) ---
        def stride(ch):
            return truth[ch][::6, ::6].copy()

        # --- references at t_out ---
        hh = t_out.hour
        doy = (t_out.date() - dt.date(t_out.year, 1, 1)).days
        ref = shards[hh]

        def era5(var, level=None):
            da = ref[var].isel(time=doy)
            if level is not None:
                da = da.isel(level=level)
            return np.flipud(da.transpose("latitude", "longitude").values)  # north-first

        acc["path_2t"].append(rmse(chain1p5(T2T), era5("2m_temperature"), w))
        acc["path_msl"].append(rmse(chain1p5(MSL), era5("mean_sea_level_pressure"), w))
        acc["path_t850"].append(rmse(chain1p5(T850), era5("temperature", 10), w))
        acc["path_z500"].append(rmse(chain1p5(Z500), era5("geopotential", 7), w))
        acc["stride_2t"].append(rmse(stride(T2T), era5("2m_temperature"), w))
        for tag, row, ev in [("v1", O_T2T, era5("2m_temperature")), ("v2", O_T2T, era5("2m_temperature"))]:
            acc[f"{tag}_2t"].append(rmse((v1 if tag == "v1" else v2)[row], ev, w))
        acc["v1_msl"].append(rmse(v1[O_MSL], era5("mean_sea_level_pressure"), w))
        acc["v2_msl"].append(rmse(v2[O_MSL], era5("mean_sea_level_pressure"), w))
        acc["v1_t850"].append(rmse(v1[O_T850], era5("temperature", 10), w))
        acc["v2_t850"].append(rmse(v2[O_T850], era5("temperature", 10), w))
        acc["v1_z500"].append(rmse(v1[O_Z500], era5("geopotential", 7), w))
        acc["v2_z500"].append(rmse(v2[O_Z500], era5("geopotential", 7), w))
        acc["v3_2t"].append(rmse(f025[O_T2T], truth[T2T], w25))
        acc["v3_msl"].append(rmse(f025[O_MSL], truth[MSL], w25))
        acc["v3_t850"].append(rmse(f025[O_T850], truth[T850], w25))
        acc["v3_z500"].append(rmse(f025[O_Z500], truth[Z500], w25))
        print(f"[{k+1}/24] {stamp} done", flush=True)

    print(f"\n{'metric':>10} {'mean':>9}   interpretation")
    rows = [("path_2t", "regrid chain on TRUTH vs era5_240 (2t) [E1: interface cost]"),
            ("stride_2t", "stride ARCO[::6,::6] vs era5_240 (2t) [E1b: provenance]"),
            ("v1_2t", "AIFS->1.5 linear (current pipeline) vs era5_240"),
            ("v2_2t", "AIFS->0.25->box6x6->1.5 vs era5_240 [better downscale?]"),
            ("v3_2t", "AIFS at 0.25 vs ARCO truth 0.25 [native quality]"),
            ("path_msl", "E1 msl"), ("v1_msl", "v1 msl"), ("v2_msl", "v2 msl"), ("v3_msl", "v3 msl"),
            ("path_t850", "E1 t850"), ("v1_t850", "v1 t850"), ("v2_t850", "v2 t850"), ("v3_t850", "v3 t850"),
            ("path_z500", "E1 z500"), ("v1_z500", "v1 z500"), ("v2_z500", "v2 z500"), ("v3_z500", "v3 z500")]
    for key, desc in rows:
        print(f"{key:>10} {np.mean(acc[key]):>9.4f}   {desc}")


if __name__ == "__main__":
    main()
