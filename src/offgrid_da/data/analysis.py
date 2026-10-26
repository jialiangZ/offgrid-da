"""ERA5 analysis reference reader (WeatherBench2 zarr).

Verified schema (Phase 0):
- 6-hourly (00/06/12/18 UTC), 1959-01-01 .. 2023-01-10T18, 62 variables
- same 240x121 grid as Pangu; native dims (time, longitude, latitude),
  latitude ascending -90..90, longitude 0..358.5
- 2m_temperature units: K

Canonical output identical to PanguReader: [C, 121, 240], lat descending.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr

from .variables import GRID_VARIABLE_MAP


@lru_cache(maxsize=4)
def _open_era5(url: str) -> xr.Dataset:
    return xr.open_zarr(url, chunks=None, storage_options={"token": "anon"})


class ERA5Reader:
    """Lazy reader for the ERA5 analysis zarr."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.ds = _open_era5(url)
        self._time_index = pd.DatetimeIndex(self.ds.time.values)

    @property
    def times(self) -> pd.DatetimeIndex:
        return self._time_index

    def has_time(self, time: pd.Timestamp) -> bool:
        return time in self._time_index

    def grid(self) -> dict[str, np.ndarray]:
        return {
            "latitude": self.ds.latitude.values[::-1].copy(),
            "longitude": self.ds.longitude.values.copy(),
        }

    def read_analysis(
        self,
        valid_time: pd.Timestamp | str,
        variables: list[str] | tuple[str, ...] = ("2m_temperature",),
    ) -> np.ndarray:
        """Read ERA5 analysis at ``valid_time`` -> [C, 121, 240] float32."""
        valid_time = pd.Timestamp(valid_time)
        arrays = []
        for var in variables:
            da = self.ds[GRID_VARIABLE_MAP[var]["era5"]]
            sel = da.sel(time=np.datetime64(valid_time, "ns"))
            values = sel.values  # (longitude, latitude)
            arrays.append(np.ascontiguousarray(values.T[::-1, :], dtype=np.float32))
        return np.stack(arrays, axis=0)
