"""Quality control for METAR observations (plan section 3.6).

Each filter records before/after counts so every QC decision is auditable.
We do NOT reject large innovations here (plan: "不要把所有大 innovation 直接视为错误观测").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .variables import DD_FLAG_MEANING


@dataclass
class QCFilters:
    """Configurable QC thresholds (all in SI units)."""

    lat_range: tuple[float, float] = (-90.0, 90.0)
    # after [-180,180] check; normalized separately to [0,360)
    lon_range: tuple[float, float] = (-180.0, 180.0)
    elevation_range_m: tuple[float, float] = (-450.0, 9000.0)
    # variable sanity ranges are taken from the VariableSpec.valid_range
    allow_dd_flags: tuple[str, ...] = ("S", "V", "C", "T", "B", "G", "Q")
    # Empirical Phase-0 finding: only 'Z' rows lack values; 'X' rows showed
    # implausible constants (333.15 K) -> both excluded by default.
    dedup_keep: str = "last_timeObs"  # per (station, variable) within a window


@dataclass
class QCReport:
    """Counts before/after each QC step for one window."""

    n_raw_reports: int = 0
    n_after_time_window: int = 0
    n_after_received_cutoff: int = 0
    n_after_location: int = 0
    n_after_dd_flag: int = 0
    n_after_finite_and_range: int = 0
    n_after_dedup: int = 0
    steps: list[tuple[str, int, int]] = field(default_factory=list)

    def add(self, name: str, before: int, after: int) -> None:
        self.steps.append((name, before, after))

    def as_dict(self) -> dict[str, int]:
        out = {
            "n_raw_reports": self.n_raw_reports,
            "n_after_time_window": self.n_after_time_window,
            "n_after_received_cutoff": self.n_after_received_cutoff,
            "n_after_location": self.n_after_location,
            "n_after_dd_flag": self.n_after_dd_flag,
            "n_after_finite_and_range": self.n_after_finite_and_range,
            "n_after_dedup": self.n_after_dedup,
        }
        return out

    def summary(self) -> str:
        lines = [f"  {name}: {before} -> {after}" for name, before, after in self.steps]
        return "\n".join(lines)


def apply_time_window(
    df: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp
) -> pd.DataFrame:
    """Keep reports with timeObs in (window_start, window_end] (half-open)."""
    mask = (df["timeObs"] > window_start) & (df["timeObs"] <= window_end)
    return df.loc[mask]


def apply_received_cutoff(df: pd.DataFrame, received_cutoff: pd.Timestamp) -> pd.DataFrame:
    """Keep reports received at or before the cutoff (latency tolerance)."""
    mask = df["timeReceived"] <= received_cutoff
    return df.loc[mask]


def apply_location_qc(df: pd.DataFrame, filters: QCFilters) -> pd.DataFrame:
    """Finite, in-range lat/lon/elevation."""
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    elev = pd.to_numeric(df["elevation"], errors="coerce")
    mask = (
        np.isfinite(lat) & np.isfinite(lon) & np.isfinite(elev)
        & lat.between(*filters.lat_range)
        & lon.between(*filters.lon_range)
        & elev.between(*filters.elevation_range_m)
    )
    return df.loc[mask]


def apply_variable_qc(
    df: pd.DataFrame,
    value_col: str,
    dd_col: str,
    valid_range: tuple[float, float],
    filters: QCFilters,
) -> pd.DataFrame:
    """DD-flag + finite + physical-range filter for one METAR variable."""
    value = pd.to_numeric(df[value_col], errors="coerce")
    dd = df[dd_col].astype("string").str.strip()
    mask = (
        dd.isin(list(filters.allow_dd_flags))
        & np.isfinite(value)
        & value.between(valid_range[0], valid_range[1])
    )
    return df.loc[mask]


def dedup_reports(
    df: pd.DataFrame, variable_name: str, filters: QCFilters
) -> pd.DataFrame:
    """Keep one report per station per window.

    Strategy: latest timeObs; ties broken by earliest timeReceived (first
    transmission wins). Applies to a single physical variable frame that has
    already passed variable QC.
    """
    if filters.dedup_keep == "last_timeObs":
        order = df.sort_values(
            ["stationName", "timeObs", "timeReceived"], ascending=[True, False, True]
        )
    else:  # pragma: no cover - alternative strategy kept explicit
        raise ValueError(f"unknown dedup_keep: {filters.dedup_keep}")
    keep_idx = order.groupby("stationName", sort=False).head(1).index
    return df.loc[sorted(keep_idx)]


__all__ = [
    "QCFilters",
    "QCReport",
    "apply_time_window",
    "apply_received_cutoff",
    "apply_location_qc",
    "apply_variable_qc",
    "dedup_reports",
    "DD_FLAG_MEANING",
]
