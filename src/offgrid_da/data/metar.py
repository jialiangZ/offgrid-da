"""MADIS METAR observation reader (WeatherBench2 parquet over HTTPS).

Verified layout (Phase 0, timeObs-by-hour dataset):
``gs://weatherbench2/datasets/metar/metar-timeObs-by-hour_2001-07-01T00_2024-01-01T00/
      year=YYYY/month=M/day=D/hour=H/YYYY-MM-DDTHH.parquet``

Verified schema (43 columns), key fields:
- stationName (ICAO code), locationName, latitude, longitude ([-180,180]),
  elevation [m]
- timeObs, timeNominal, timeReceived (datetime64[ns])
- temperature [K], dewpoint [K], windDir [deg from], windSpeed [m/s],
  seaLevelPress [Pa], altimeter [Pa], visibility [m], windGust [m/s], ...
- QC: one ``<var>DD`` data-descriptor flag per variable (S/T/V/C/Z).
  Numeric MADIS quality codes are NOT part of the parquet export.

MADIS hourly files cover [H-15min, H+44min] of observation time, so an
observation window must span at least two hourly files.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ..utils.io import read_parquet_cached

METAR_COLUMNS = [
    "stationName", "locationName", "latitude", "longitude", "elevation",
    "timeObs", "timeNominal", "timeReceived", "reportType", "autoStationType",
    "visibility", "visibilityDD", "presWeather", "seaLevelPress", "seaLevelPressDD",
    "temperature", "temperatureDD", "dewpoint", "dewpointDD", "windDir", "windDirDD",
    "windSpeed", "windSpeedDD", "windGust", "windGustDD", "altimeter", "altimeterDD",
    "minTemp24Hour", "minTemp24HourDD", "maxTemp24Hour", "maxTemp24HourDD",
    "precip1Hour", "precip1HourDD", "precip3Hour", "precip3HourDD",
    "precip6Hour", "precip6HourDD", "precip24Hour", "precip24HourDD",
    "pressChange3Hour", "pressChange3HourDD", "snowCover", "snowCoverDD",
]


def metar_hour_url(base_url: str, layout: str, stamp: pd.Timestamp) -> str:
    """Object URL of the hourly parquet partition for ``stamp`` (floor to hour)."""
    stamp = stamp.floor("h")
    return (
        f"{base_url.rstrip('/')}/{layout}/"
        f"year={stamp.year}/month={stamp.month}/day={stamp.day}/hour={stamp.hour}/"
        f"{stamp.strftime('%Y-%m-%dT%H')}.parquet"
    )


@lru_cache(maxsize=256)
def _read_hour(base_url: str, layout: str, stamp: pd.Timestamp, cache_dir: str) -> pd.DataFrame:
    url = metar_hour_url(base_url, layout, stamp)
    df = read_parquet_cached(url, cache_dir)
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index(drop=True)
    return df


class MetarReader:
    """Reads raw METAR parquet partitions with local caching."""

    def __init__(self, base_url: str, layout: str, cache_dir: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.layout = layout
        self.cache_dir = str(cache_dir)

    def read_hour(self, stamp: pd.Timestamp) -> pd.DataFrame:
        return _read_hour(self.base_url, self.layout, stamp.floor("h"), self.cache_dir)

    def read_hours(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Concatenate hourly partitions covering (start.floor('h') .. end.floor('h'))."""
        if end < start:
            raise ValueError(f"end {end} before start {start}")
        stamps = pd.date_range(start.floor("h"), end.floor("h"), freq="h")
        frames = [self.read_hour(s) for s in stamps]
        df = pd.concat(frames, ignore_index=True)
        return df

    def read_window(self, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
        """Read all partitions that can contain timeObs in [window_start, window_end].

        Because a partition for hour H holds obs in [H-15min, H+44min], we read
        partitions from (window_start - 45min).floor(h) through window_end.floor(h)
        and let the QC/window filter apply the exact bounds.
        """
        first = (window_start - pd.Timedelta(minutes=45)).floor("h")
        last = window_end.floor("h")
        if last < first:
            last = first
        return self.read_hours(first, last)

    @staticmethod
    def normalize_longitude(lon: np.ndarray | pd.Series) -> np.ndarray:
        """Convert longitude from [-180, 180] to [0, 360)."""
        arr = np.asarray(lon, dtype=np.float64)
        return np.mod(arr, 360.0)
