"""Pangu forecast background reader (WeatherBench2 zarr).

Verified schema (Phase 0):
- dims: time=3652 (00Z/12Z inits, 2018-01-01 .. 2022-12-31T12),
        prediction_timedelta=40 (6h .. 240h step 6h), longitude=240, latitude=121
- coordinate name is ``prediction_timedelta`` (NOT ``lead_time``)
- native array order is (time, prediction_timedelta, longitude, latitude)
- latitude ascending -90..90, longitude 0..358.5 (1.5 deg equiangular with poles)

The reader converts to the project's canonical layout:
- field arrays shaped [C, H, W] = [C, 121, 240] with latitude DESCENDING
  (90 -> -90) and longitude ascending [0, 360).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr

from .variables import GRID_VARIABLE_MAP


@lru_cache(maxsize=4)
def _open_pangu(url: str) -> xr.Dataset:
    return xr.open_zarr(url, chunks=None, storage_options={"token": "anon"})


class PanguReader:
    """Lazy reader for the Pangu forecast zarr."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.ds = _open_pangu(url)
        self._time_index = pd.DatetimeIndex(self.ds.time.values)
        # lead times in integer hours
        self._lead_hours = np.array(
            [int(np.timedelta64(v, "h") / np.timedelta64(1, "h")) if isinstance(v, np.timedelta64) else int(v)
             for v in self.ds.prediction_timedelta.values]
        )

    @property
    def lead_hours(self) -> np.ndarray:
        return self._lead_hours.copy()

    @property
    def init_times(self) -> pd.DatetimeIndex:
        return self._time_index

    def has_init(self, init_time: pd.Timestamp) -> bool:
        return init_time in self._time_index

    def has_lead(self, lead_hours: int) -> bool:
        return int(lead_hours) in self._lead_hours

    def grid(self) -> dict[str, np.ndarray]:
        """Canonical grid: latitude descending, longitude ascending [0,360)."""
        return {
            "latitude": self.ds.latitude.values[::-1].copy(),   # 90 -> -90
            "longitude": self.ds.longitude.values.copy(),      # 0 -> 358.5
        }

    def read_background(
        self,
        init_time: pd.Timestamp | str,
        lead_hours: int,
        variables: list[str] | tuple[str, ...] = ("2m_temperature",),
    ) -> np.ndarray:
        """Read background fields valid at ``init_time + lead_hours``.

        Returns float32 array [C, H, W] in canonical orientation
        (lat descending, lon ascending).
        """
        init_time = pd.Timestamp(init_time)
        if not self.has_lead(lead_hours):
            raise KeyError(f"Pangu lead_time {lead_hours}h not available; available={sorted(set(self._lead_hours))}")
        lead_delta: object
        if np.issubdtype(self.ds.prediction_timedelta.dtype, np.timedelta64):
            lead_delta = np.timedelta64(lead_hours, "h")
        else:  # stored as integer hours (verified: int64 in this zarr)
            lead_delta = int(lead_hours)
        arrays = []
        for var in variables:
            da = self.ds[GRID_VARIABLE_MAP[var]["pangu"]]
            sel = da.sel(time=np.datetime64(init_time, "ns"), prediction_timedelta=lead_delta)
            values = sel.values  # (longitude, latitude)
            # (lon, lat) -> (lat, lon), then flip latitude to descending
            arrays.append(np.ascontiguousarray(values.T[::-1, :], dtype=np.float32))
        return np.stack(arrays, axis=0)
