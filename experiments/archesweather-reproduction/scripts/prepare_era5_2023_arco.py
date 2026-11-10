#!/usr/bin/env python3
"""Prepare ERA5 2023 shards for the 240x121 grid from ARCO-ERA5 v3.

Background: the WeatherBench2 bucket only provides the 240x121 conservative
regrid up to 2023-01-10 (`1959-2023_01_10-*`). Full-year 2023 fields exist only
in ARCO-ERA5 `ar/full_37-1h-0p25deg-chunk-1.zarr-v3` (0.25 deg). This script

  1. builds a first-order area-conservative regrid (0.25 deg -> WB2 240x121
     equiangular-with-poles, overlap-area x cos(lat) weights, pole cells
     clipped at +/-90, periodic longitude);
  2. validates the pipeline against the existing WB2 2022 shards
     (--validate-2022): same timestamps, regrid ARCO, report per-variable
     bias/RMS/max differences. Acceptance criterion discussed in report;
  3. on --make-2023, writes era5_240_{2023}_{0,6,12,18}h.nc with the same
     variables/levels/grid/validation as prepare_era5_2021_2022.py.

Run with experiments/archesweather-reproduction/geoarches/.venv/bin/python.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import xarray as xr
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/era5_240/full'
STAGE = ROOT / 'data/era5_240/staging_2023'
REPORT = ROOT / 'data/era5_240/prepare_2023_report.json'
ARCO = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
VARS = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
        'mean_sea_level_pressure', 'geopotential', 'u_component_of_wind',
        'v_component_of_wind', 'temperature', 'specific_humidity', 'vertical_velocity']
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SRC_LAT = 90.0 - np.arange(721) * 0.25          # 90 -> -90, step .25
SRC_LON = np.arange(1440) * 0.25                 # 0 -> 359.75
TGT_LAT = np.linspace(-90.0, 90.0, 121)          # -90 -> 90, step 1.5 (ascending, matches existing shards)
TGT_LON = np.arange(240) * 1.5                   # 0 -> 358.5


def build_regridder():
    # Latitude: weights with cos(lat) area factor via overlap midpoint.
    half_s, half_t = 0.125, 0.75
    rows, cols, vals = [], [], []
    for ti, tc in enumerate(TGT_LAT):
        tlo, thi = max(tc - half_t, -90.0), min(tc + half_t, 90.0)
        for si, sc in enumerate(SRC_LAT):
            lo, hi = max(sc - half_s, -90.0), min(sc + half_s, 90.0)
            ovlo, ovhi = max(tlo, lo), min(thi, hi)
            if ovhi <= ovlo:
                continue
            w = (ovhi - ovlo) * np.cos(np.deg2rad(0.5 * (ovlo + ovhi)))
            rows.append(ti); cols.append(si); vals.append(w)
    Wlat = sp.csr_matrix((vals, (rows, cols)), shape=(121, 721))
    Wlat = sp.diags(1.0 / Wlat.sum(axis=1).A.ravel()) @ Wlat
    # Longitude: periodic, uniform cos -> plain overlap fractions.
    rows, cols, vals = [], [], []
    for ti, tc in enumerate(TGT_LON):
        lo, hi = tc - half_t, tc + half_t
        for si, sc in enumerate(SRC_LON):
            slo, shi = sc - half_s, sc + half_s
            ov = min(hi, shi) - max(lo, slo)
            if ov <= 0 and (lo < 0 or hi > 360.0):
                # wrap: compare against shifted copies
                for shift in (-360.0, 360.0):
                    ov = max(ov, min(hi, shi + shift) - max(lo, slo + shift))
            if ov <= 0:
                continue
            rows.append(ti); cols.append(si); vals.append(ov)
    Wlon = sp.csr_matrix((vals, (rows, cols)), shape=(240, 1440))
    Wlon = sp.diags(1.0 / Wlon.sum(axis=1).A.ravel()) @ Wlon
    return Wlat.tocsr(), Wlon.tocsr()


def regrid(field, Wlat, Wlon):
    """field: (..., 721, 1440) -> (..., 121, 240)."""
    f = np.asarray(field)
    lead = f.shape[:-2]
    x = f.reshape(-1, 721, 1440)
    out = np.empty((x.shape[0], 121, 240), dtype=np.float64)
    for k in range(x.shape[0]):
        out[k] = Wlat @ x[k] @ Wlon.T
    return out.reshape(lead + (121, 240))


def open_arco():
    return xr.open_zarr(ARCO, storage_options={'token': 'anon'}, consolidated=True)


def arco_field(ds, var, t):
    da = ds[var].sel(time=[t])
    if 'level' in da.dims:
        da = da.sel(level=LEVELS)
    return da.values[0]  # (13,721,1440) or (721,1440)


def validate_2022(ntimes):
    Wlat, Wlon = build_regridder()
    ds = open_arco()
    summary = {}
    for hour in (0, 6, 12, 18):
        ref = xr.open_dataset(OUT / f'era5_240_2022_{hour}h.nc')
        times = ref.time.values
        picks = np.unique(np.linspace(0, len(times) - 1, ntimes).astype(int))
        for name in VARS:
            diffs = []
            for p in picks:
                t = times[p]
                a = regrid(arco_field(ds, name, t), Wlat, Wlon)
                r = ref[name].sel(time=[t])
                r = r.transpose(..., 'latitude', 'longitude')
                r = r.values[0]
                if a.shape != r.shape:
                    raise ValueError(f'{name} {t}: shape {a.shape} vs {r.shape}')
                if not np.isfinite(a).all():
                    raise ValueError(f'{name} {t}: nonfinite ARCO regrid')
                if not np.isfinite(r).all():
                    raise ValueError(f'{name} {t}: nonfinite reference')
                diffs.append(a - r)
            d = np.stack(diffs)
            # cos(lat) area weights, correct for both (T,lat,lon) and (T,lev,lat,lon)
            w = np.cos(np.deg2rad(TGT_LAT))[(None,) * (d.ndim - 2) + (slice(None), None)]
            norm = np.broadcast_to(w, d.shape).sum()
            summary[f'{hour}h/{name}'] = {
                'bias': float((d * w).sum() / norm),
                'rms': float(np.sqrt((d ** 2 * w).sum() / norm)),
                'max': float(np.abs(d).max()), 'n_times': int(len(picks))}
        ref.close()
        print(f'[validate-2022] hour {hour}h done', flush=True)
    ds.close()
    print(json.dumps(summary, indent=1))
    return summary


def validate_shard(path, year, hour):
    with xr.open_dataset(path) as ds:
        missing = set(VARS) - set(ds.data_vars)
        if missing: raise ValueError(f'{path}: missing {missing}')
        if dict(ds.sizes) != {'time': 365, 'longitude': 240, 'latitude': 121, 'level': 13}:
            raise ValueError(f'{path}: dimensions {dict(ds.sizes)}')
        expected = np.arange(np.datetime64(f'{year}-01-01T{hour:02d}', 'h'),
                             np.datetime64(f'{year + 1}-01-01T{hour:02d}', 'h'),
                             np.timedelta64(24, 'h')).astype('datetime64[ns]')
        np.testing.assert_array_equal(ds.time.values, expected)
        np.testing.assert_allclose(ds.longitude.values, np.arange(240) * 1.5)
        np.testing.assert_array_equal(ds.latitude.values, np.linspace(-90, 90, 121))
        np.testing.assert_array_equal(ds.level.values, LEVELS)
        for name in VARS:
            a = ds[name].isel(time=[0, -1]).values
            if not np.isfinite(a).all(): raise ValueError(f'{path}: nonfinite {name} at boundary')
    return path.stat().st_size


def make_2023(hour):
    Wlat, Wlon = build_regridder()
    ds = open_arco()
    target = OUT / f'era5_240_2023_{hour}h.nc'
    if target.exists():
        return {'hour': hour, 'status': 'existing_verified', 'bytes': validate_shard(target, 2023, hour), 'source': ARCO}
    temp = STAGE / f'era5_240_2023_{hour}h.{os.getpid()}.part.nc'
    print('starting 2023', hour, flush=True)
    t0 = time.time()
    fields = {name: [] for name in VARS}
    for day in range(365):
        t = np.datetime64('2023-01-01T00:00:00', 'ns') + np.timedelta64(day, 'D') + np.timedelta64(hour, 'h')
        snap = ds[VARS].sel(time=[t])
        if 'level' in snap.dims or 'level' in snap.coords:
            snap = snap.sel(level=LEVELS)
        snap.load()  # parallel zarr chunk reads via dask
        for name in VARS:
            fields[name].append(regrid(snap[name].values[0], Wlat, Wlon).astype(np.float32))
        if (day + 1) % 30 == 0:
            print(f'  {hour}h: day {day + 1}/365 ({time.time() - t0:.0f}s)', flush=True)
    ds.close()
    data_vars = {}
    is3d = {'geopotential', 'u_component_of_wind', 'v_component_of_wind', 'temperature',
            'specific_humidity', 'vertical_velocity'}
    for name in VARS:
        arr = np.stack(fields[name])  # (T, lev, lat, lon) or (T, lat, lon)
        if name in is3d:
            data_vars[name] = (('time', 'level', 'longitude', 'latitude'),
                               arr.transpose(0, 1, 3, 2).astype(np.float32))
        else:
            data_vars[name] = (('time', 'longitude', 'latitude'),
                               arr.transpose(0, 2, 1).astype(np.float32))
    coords = {'time': np.arange(np.datetime64('2023-01-01T00:00:00', 'ns'),
                                np.datetime64('2024-01-01T00:00:00', 'ns'),
                                np.timedelta64(24, 'h')) + np.timedelta64(hour, 'h'),
              'level': LEVELS, 'latitude': TGT_LAT, 'longitude': TGT_LON}
    xr.Dataset(data_vars, coords).to_netcdf(temp)
    size = validate_shard(temp, 2023, hour)
    os.rename(temp, target)
    print('completed 2023', hour, size, f'({time.time() - t0:.0f}s)', flush=True)
    return {'hour': hour, 'status': 'downloaded_verified', 'bytes': size, 'source': ARCO}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate-2022', type=int, metavar='NTIMES', help='validate regrid vs existing 2022 shards')
    ap.add_argument('--make-2023', action='store_true')
    ap.add_argument('--hour', type=int, choices=(0, 6, 12, 18), help='restrict to one shard')
    args = ap.parse_args()
    STAGE.mkdir(parents=True, exist_ok=True)
    if args.validate_2022:
        s = validate_2022(args.validate_2022)
        REPORT.write_text(json.dumps({'purpose': 'ARCO->240x121 regrid validation on 2022',
                                      'source': ARCO, 'validation': s}, indent=2) + '\n')
    if args.make_2023:
        hours = (args.hour,) if args.hour is not None else (0, 6, 12, 18)
        results = []
        for h in hours:
            results.append(make_2023(h))
        REPORT.write_text(json.dumps({'purpose': 'ERA5 2023 shards from ARCO v3 (first-order conservative regrid)',
                                      'source': ARCO, 'shards': results,
                                      'notes': 'WB2 bucket lacks full-year 2023; validated against 2022 WB2 shards first.'}, indent=2) + '\n')
        print('ALL VERIFIED', REPORT, flush=True)


if __name__ == '__main__':
    main()
