#!/usr/bin/env python3
"""Prepare validated ERA5 shards for 1998-2020 (migration rebuild, replaces geoarches dl_era path).

Same source family and validation as prepare_era5_2021_2022.py: official WB2
conservative 240x121 zarr -> era5_240_{year}_{h}h.nc per year/hour, atomic
staging, skip-if-exists-and-validates (idempotent/resumable).

Run with the repo root .venv: ../../.venv/bin/python prepare_era5_1998_2020.py
"""
import calendar
import concurrent.futures
import json
import os
import time
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path('/home/jlzhang/Documents/offgrid-da/experiments/archesweather-reproduction')
OUT = ROOT / 'data/era5_240/full'
STAGE = ROOT / 'data/era5_240/staging_1998_2020'
REPORT = ROOT / 'data/era5_240/prepare_1998_2020_report.json'
SRC = 'gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr'
VARS = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'mean_sea_level_pressure',
        'geopotential', 'u_component_of_wind', 'v_component_of_wind', 'temperature',
        'specific_humidity', 'vertical_velocity']
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
YEARS = range(1998, 2021)


def validate(path, year, hour):
    nday = 366 if calendar.isleap(year) else 365
    with xr.open_dataset(path) as ds:
        missing = set(VARS) - set(ds.data_vars)
        if missing:
            raise ValueError(f'{path}: missing {missing}')
        if dict(ds.sizes) != {'time': nday, 'longitude': 240, 'latitude': 121, 'level': 13}:
            raise ValueError(f'{path}: dimensions {dict(ds.sizes)}')
        expected = np.arange(np.datetime64(f'{year}-01-01T{hour:02d}', 'h'),
                             np.datetime64(f'{year+1}-01-01T{hour:02d}', 'h'),
                             np.timedelta64(24, 'h')).astype('datetime64[ns]')
        np.testing.assert_array_equal(ds.time.values, expected)
        np.testing.assert_allclose(ds.longitude.values, np.arange(240) * 1.5)
        np.testing.assert_allclose(np.sort(ds.latitude.values), np.linspace(-90, 90, 121))
        np.testing.assert_array_equal(ds.level.values, LEVELS)
        for name in VARS:
            a = ds[name].isel(time=[0, -1]).values
            if not np.isfinite(a).all():
                raise ValueError(f'{path}: nonfinite {name} at boundary')
    return path.stat().st_size


def job(year, hour):
    target = OUT / f'era5_240_{year}_{hour}h.nc'
    if target.exists():
        return {'year': year, 'hour': hour, 'status': 'existing_verified',
                'bytes': validate(target, year, hour), 'source': SRC}
    STAGE.mkdir(parents=True, exist_ok=True)
    temp = STAGE / f'era5_240_{year}_{hour}h.{os.getpid()}.part.nc'
    print('starting', year, hour, flush=True)
    ds = xr.open_zarr(SRC, storage_options={'token': 'anon'}, consolidated=True)
    sub = ds[VARS].sel(time=slice(f'{year}-01-01', f'{year}-12-31T23:59:59'))
    sub = sub.sel(time=sub.time.dt.hour == hour)
    sub.to_netcdf(temp)
    ds.close()
    size = validate(temp, year, hour)
    os.rename(temp, target)
    print('completed', year, hour, size, flush=True)
    return {'year': year, 'hour': hour, 'status': 'downloaded_verified',
            'bytes': size, 'source': SRC}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(job, y, h) for y in YEARS for h in (0, 6, 12, 18)]
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            results.append(f.result())
            if (i + 1) % 8 == 0:
                gb = sum(p.stat().st_size for p in OUT.glob('*.nc')) / 2**30
                rate = (i + 1) / (time.time() - t0)
                print(f'{i+1}/92 done | {gb:.0f} GiB total | {rate*3600:.1f} files/h', flush=True)
    report = {'purpose': 'training-archive ERA5 shards 1998-2020 (migration rebuild)',
              'variables': VARS, 'grid': 'WB2 conservative 1.5 degrees',
              'shards': sorted(results, key=lambda x: (x['year'], x['hour'])),
              'notes': 'Replaces geoarches dl_era path (dropped in migration); leap years validated at 366.'}
    temp = REPORT.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2) + '\n')
    os.replace(temp, REPORT)
    print('ALL VERIFIED', REPORT, flush=True)


if __name__ == '__main__':
    main()
