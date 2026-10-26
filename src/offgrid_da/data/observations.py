"""Unified irregular observation representation (plan section 3.5).

Each scalar observation becomes one ``ObservationRecord``. METAR rows with
multiple valid variables expand to multiple records; future sources (IGRA,
aircraft, buoy) can emit the same structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .metar import MetarReader
from .qc import (
    QCFilters,
    QCReport,
    apply_location_qc,
    apply_received_cutoff,
    apply_time_window,
    apply_variable_qc,
    dedup_reports,
)
from .variables import DD_QUALITY_CODE, VariableRegistry


SOURCE_METAR = 0  # source_id registry (extend: IGRA=1, AIRCRAFT=2, BUOY=3, ...)


@dataclass
class ObservationRecord:
    station_id: str
    source_id: int
    variable_id: int
    latitude: float          # degrees, [-90, 90]
    longitude: float         # degrees, [0, 360) canonical
    elevation_m: float
    valid_time: pd.Timestamp  # timeObs
    received_time: pd.Timestamp
    value: float
    qc_flag: str             # original DD descriptor
    quality_code: int        # numeric quality (derived from DD)


@dataclass
class ObservationSet:
    """A collection of scalar observations for one analysis cycle."""

    records: list[ObservationRecord] = field(default_factory=list)
    qc_report: QCReport | None = None
    window_start: pd.Timestamp | None = None
    window_end: pd.Timestamp | None = None

    def __len__(self) -> int:
        return len(self.records)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.records])


def metar_to_observation_set(
    reader: MetarReader,
    registry: VariableRegistry,
    variable_names: Iterable[str],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    received_cutoff: pd.Timestamp,
    filters: QCFilters | None = None,
    station_keep: set[str] | None = None,
) -> ObservationSet:
    """Build an ObservationSet from METAR for timeObs in (window_start, window_end].

    Steps (each counted):
      1. read hourly partitions covering the window
      2. timeObs window filter
      3. received-time cutoff (plan section 3.4: obs received <= t + latency)
      4. location QC (finite, physical ranges)
      5. per-variable DD flag + finite + range QC
      6. dedup per station (latest timeObs)
    """
    filters = filters or QCFilters()
    report = QCReport()

    raw = reader.read_window(window_start, window_end)
    report.n_raw_reports = len(raw)
    report.add("raw_partition_rows", len(raw), len(raw))

    df = apply_time_window(raw, window_start, window_end)
    report.n_after_time_window = len(df)
    report.add("timeObs_window", report.n_raw_reports, len(df))

    df = apply_received_cutoff(df, received_cutoff)
    report.n_after_received_cutoff = len(df)
    report.add("received_cutoff", report.n_after_time_window, len(df))

    df = apply_location_qc(df, filters)
    report.n_after_location = len(df)
    report.add("location_qc", report.n_after_received_cutoff, len(df))

    if station_keep is not None:
        before = len(df)
        df = df.loc[df["stationName"].isin(station_keep)]
        report.add("station_split_filter", before, len(df))

    lon0 = MetarReader.normalize_longitude(df["longitude"].to_numpy()) if len(df) else np.zeros(0)

    obs = ObservationSet(window_start=window_start, window_end=window_end)
    n_after_dedup_total = 0
    for var_name in variable_names:
        spec = registry.get(var_name)
        if spec.metar_column is None:
            continue
        vdf = apply_variable_qc(df, spec.metar_column, spec.metar_dd_column, spec.valid_range, filters)
        n_after_var = len(vdf)
        vdf = dedup_reports(vdf, var_name, filters)
        n_after_dedup_total += len(vdf)
        if var_name == list(variable_names)[0]:
            report.n_after_dd_flag = n_after_var  # reported for the primary variable
        sub_lon = MetarReader.normalize_longitude(vdf["longitude"].to_numpy())
        sub_lat = vdf["latitude"].to_numpy(dtype=np.float64)
        sub_elev = vdf["elevation"].to_numpy(dtype=np.float64)
        sub_val = vdf[spec.metar_column].to_numpy(dtype=np.float64)
        sub_time = pd.DatetimeIndex(vdf["timeObs"])
        sub_recv = pd.DatetimeIndex(vdf["timeReceived"])
        sub_dd = vdf[spec.metar_dd_column].astype(str).to_numpy()
        sub_station = vdf["stationName"].astype(str).to_numpy()
        for i in range(len(vdf)):
            obs.records.append(
                ObservationRecord(
                    station_id=sub_station[i],
                    source_id=SOURCE_METAR,
                    variable_id=spec.variable_id,
                    latitude=float(sub_lat[i]),
                    longitude=float(sub_lon[i]),
                    elevation_m=float(sub_elev[i]),
                    valid_time=sub_time[i],
                    received_time=sub_recv[i],
                    value=float(sub_val[i]),
                    qc_flag=sub_dd[i],
                    quality_code=DD_QUALITY_CODE.get(sub_dd[i], 0),
                )
            )
    report.n_after_finite_and_range = report.n_after_dd_flag  # primary variable
    report.n_after_dedup = n_after_dedup_total
    report.add("dedup_per_station", report.n_after_dd_flag, n_after_dedup_total)
    obs.qc_report = report
    # keep flake8 quiet about unused lon0 (kept for debugging)
    _ = lon0
    return obs
