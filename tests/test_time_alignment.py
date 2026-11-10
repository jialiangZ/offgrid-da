"""Time alignment: valid_time = init_time + lead, exact matches on both sides."""
from __future__ import annotations

import pandas as pd
import pytest

from offgrid_da.data.alignment import build_sample_index

pytestmark = pytest.mark.requires_gcs


def test_index_time_rule(dataset):
    df = dataset.sample_table()
    assert (df["valid_time"] == df["init_time"] + pd.to_timedelta(df["lead_time_hours"], unit="h")).all()
    assert (df["observation_window_end"] == df["valid_time"]).all()
    assert (df["observation_window_start"] == df["valid_time"] - pd.Timedelta(minutes=60)).all()


def test_all_valid_times_in_era5(dataset, cfg):
    from offgrid_da.data.analysis import ERA5Reader

    era5 = ERA5Reader(cfg["data"]["era5"]["url"])
    assert all(era5.has_time(t) for t in dataset.sample_table()["valid_time"])


def test_analysis_hours(dataset):
    hours = set(dataset.sample_table()["valid_time"].dt.hour)
    assert hours == {0, 6, 12, 18}  # every ERA5 six-hour valid time
    table = dataset.sample_table()
    assert (table["valid_time"].diff().dropna() == pd.Timedelta(hours=6)).all()
    assert set(table["lead_time_hours"]) == {6, 12}


def test_lead_missing_raises(cfg):
    from offgrid_da.data.analysis import ERA5Reader
    from offgrid_da.data.forecast import PanguReader

    pangu = PanguReader(cfg["data"]["pangu"]["url"])
    era5 = ERA5Reader(cfg["data"]["era5"]["url"])
    with pytest.raises(KeyError):
        build_sample_index(pangu, era5, "2020-01-01", "2020-01-02", lead_hours=5)


def test_sample_background_matches_valid_time(dataset):
    """Background read via the dataset must equal a direct Pangu read at init+lead."""
    row = dataset.sample_table().iloc[0]
    bg = dataset[0]["background"][0].numpy()
    direct = dataset.pangu.read_background(row["init_time"], row["lead_time_hours"], ["2m_temperature"])[0]
    assert (bg == direct).all()
