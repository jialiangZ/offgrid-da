#!/usr/bin/env python3
from __future__ import annotations

import calendar
from pathlib import Path
import sys

import numpy as np
import xarray as xr


root = Path(__file__).resolve().parents[1]
data_dir = root / "data" / "era5_240" / "full"
errors: list[str] = []
total_bytes = 0
files = 0

for year in range(1998, 2021):
    expected_samples = 366 if calendar.isleap(year) else 365
    for hour in (0, 6, 12, 18):
        path = data_dir / f"era5_240_{year}_{hour}h.nc"
        if not path.is_file():
            errors.append(f"missing: {path.name}")
            continue
        size = path.stat().st_size
        total_bytes += size
        files += 1
        if size < 4_580_000_000:
            errors.append(f"too small ({size} bytes): {path.name}")
            continue
        try:
            with xr.open_dataset(path) as ds:
                expected = {
                    "time": expected_samples,
                    "longitude": 240,
                    "latitude": 121,
                    "level": 13,
                }
                for dim, value in expected.items():
                    actual = ds.sizes.get(dim)
                    if actual != value:
                        errors.append(f"{path.name}: {dim}={actual}, expected {value}")
                times = ds["time"].values
                if len(times) == expected_samples:
                    first = np.datetime64(f"{year}-01-01T{hour:02d}:00:00")
                    last = np.datetime64(f"{year}-12-{31 if expected_samples == 365 else 31}T{hour:02d}:00:00")
                    if times[0] != first or times[-1] != last:
                        errors.append(
                            f"{path.name}: time bounds {times[0]}..{times[-1]}, expected {first}..{last}"
                        )
                    if any(np.datetime_as_string(t, unit="h")[-2:] not in ("00", "06", "12", "18") for t in times):
                        errors.append(f"{path.name}: unexpected UTC hour")
        except Exception as exc:
            errors.append(f"{path.name}: cannot open ({type(exc).__name__}: {exc})")

print(f"validated_files={files}/92")
print(f"total_bytes={total_bytes}")
if errors:
    print("ERRORS:")
    print("\n".join(errors))
    sys.exit(1)
print("ERA5 shard structure and time bounds are valid.")
