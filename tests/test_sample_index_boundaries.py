"""Date range boundaries for single-cycle sample indexing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from offgrid_da.data.alignment import build_sample_index


class FakePangu:
    init_times = pd.date_range("2019-12-31 12:00", "2020-01-02 12:00", freq="12h")
    lead_hours = np.array([6, 12])

    def has_lead(self, lead_hours: int) -> bool:
        return lead_hours in self.lead_hours


class FakeERA5:
    times = pd.date_range("2020-01-01 00:00", "2020-01-03 00:00", freq="6h")

    def has_time(self, valid_time: pd.Timestamp) -> bool:
        return valid_time in self.times


def test_date_end_is_exclusive_and_includes_full_days() -> None:
    index = build_sample_index(FakePangu(), FakeERA5(), "2020-01-01", "2020-01-03")

    assert index["valid_time"].tolist() == list(
        pd.to_datetime(
            [
                "2020-01-01 06:00",
                "2020-01-01 18:00",
                "2020-01-02 06:00",
                "2020-01-02 18:00",
            ]
        )
    )


def test_sample_at_end_boundary_is_excluded() -> None:
    index = build_sample_index(FakePangu(), FakeERA5(), "2020-01-01", "2020-01-02")

    assert index["valid_time"].tolist() == list(
        pd.to_datetime(["2020-01-01 06:00", "2020-01-01 18:00"])
    )


def test_shortest_available_lead_covers_each_six_hour_valid_time() -> None:
    index = build_sample_index(
        FakePangu(), FakeERA5(), "2020-01-01", "2020-01-03",
        lead_hours=None, valid_interval_hours=6,
    )
    expected = pd.date_range("2020-01-01 00:00", "2020-01-02 18:00", freq="6h")
    assert index["valid_time"].tolist() == list(expected)
    assert index["lead_time_hours"].tolist() == [12, 6, 12, 6, 12, 6, 12, 6]
    assert (index["valid_time"] == index["init_time"] + pd.to_timedelta(index["lead_time_hours"], unit="h")).all()


def test_fixed_lead_behavior_is_preserved() -> None:
    index = build_sample_index(
        FakePangu(), FakeERA5(), "2020-01-01", "2020-01-02", lead_hours=6
    )
    assert index["valid_time"].tolist() == list(pd.to_datetime(["2020-01-01 06:00", "2020-01-01 18:00"]))
