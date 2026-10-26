"""QC behaviour tests on real METAR data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from offgrid_da.data.qc import (
    QCFilters,
    apply_location_qc,
    apply_time_window,
    apply_variable_qc,
    dedup_reports,
)

pytestmark = pytest.mark.requires_gcs


def test_dd_filter_rejects_missing_and_outlier_flags(metar_hour):
    filters = QCFilters()
    out = apply_variable_qc(
        metar_hour, "temperature", "temperatureDD", (180.0, 340.0), filters
    )
    dd = out["temperatureDD"].astype(str)
    assert not dd.isin(["Z", "X"]).any()
    assert dd.isin(list(filters.allow_dd_flags)).all()
    assert np.isfinite(out["temperature"]).all()
    assert out["temperature"].between(180.0, 340.0).all()


def test_location_qc(metar_hour):
    filters = QCFilters()
    out = apply_location_qc(metar_hour, filters)
    assert out["latitude"].between(-90, 90).all()
    assert out["longitude"].between(-180, 180).all()
    assert out["elevation"].between(*filters.elevation_range_m).all()


def test_time_window_half_open(metar_hour):
    start = pd.Timestamp("2020-01-01T05:30:00")
    end = pd.Timestamp("2020-01-01T06:00:00")
    out = apply_time_window(metar_hour, start, end)
    assert (out["timeObs"] > start).all() and (out["timeObs"] <= end).all()


def test_dedup_keeps_latest_per_station(metar_hour):
    filters = QCFilters()
    vdf = apply_variable_qc(metar_hour, "temperature", "temperatureDD", (180.0, 340.0), filters)
    out = dedup_reports(vdf, "2m_temperature", filters)
    assert out["stationName"].is_unique
    # each kept timeObs must be the max for that station
    max_t = vdf.groupby("stationName")["timeObs"].max()
    kept_t = out.set_index("stationName")["timeObs"].reindex(max_t.index)
    assert (kept_t == max_t).all()


def test_qc_elevation_outlier_rejected(metar_hour):
    filters = QCFilters()
    df = metar_hour.copy()
    df.iloc[0, df.columns.get_loc("elevation")] = 99999.0
    out = apply_location_qc(df, filters)
    assert 99999.0 not in out["elevation"].to_numpy()
