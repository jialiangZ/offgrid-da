"""Observation window and no-future-leakage tests (plan sections 3.4, 9)."""
from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.requires_gcs


def _obs_records(dataset, idx):
    import numpy as np

    from offgrid_da.data.observations import metar_to_observation_set

    row = dataset.sample_table().iloc[idx]
    vt = row["valid_time"]
    obs = metar_to_observation_set(
        dataset.metar, dataset.registry, dataset.variables,
        window_start=vt - pd.Timedelta(minutes=60),
        window_end=vt,
        received_cutoff=vt + pd.Timedelta(minutes=30),
        filters=dataset.qc_filters,
        station_keep=dataset.context_stations,
    )
    return obs, vt


def test_obs_within_window_and_cutoff(dataset):
    for idx in (0, 1, len(dataset) - 1):
        obs, vt = _obs_records(dataset, idx)
        assert len(obs) > 100
        for r in obs.records:
            assert vt - pd.Timedelta(minutes=60) < r.valid_time <= vt
            assert r.received_time <= vt + pd.Timedelta(minutes=30)


def test_no_future_observation_leakage(dataset):
    """No observation token may come from after its valid_time."""
    sample = dataset[1]
    offsets = sample["obs_time_offsets"].numpy()
    assert (offsets <= 0).all()          # timeObs - valid_time <= 0
    assert (offsets > -60).all()         # within the 60-min window


def test_time_offsets_consistent_with_window(dataset):
    s = dataset[0]
    dt = s["obs_time_offsets"].numpy()
    assert dt.max() <= 0 and dt.min() > -60.0


def test_qc_counts_reported(dataset):
    s = dataset[0]
    qc = s["qc_report"]
    assert qc["n_raw_reports"] >= qc["n_after_time_window"] >= qc["n_after_location"] > 0
    assert qc["n_after_dedup"] > 0
    assert qc["n_after_dedup"] <= qc["n_after_time_window"]


def test_narrow_window_returns_near_zero_obs(dataset):
    """A 1-minute window holds almost no reports: the standard hourly METAR
    cadence gives none; only occasional SPECI specials (e.g. at :31) appear.
    Real finding: 14 SPECI reports globally in this specific minute."""
    from offgrid_da.data.observations import metar_to_observation_set

    obs = metar_to_observation_set(
        dataset.metar, dataset.registry, dataset.variables,
        window_start=pd.Timestamp("2020-01-02T03:30:00"),
        window_end=pd.Timestamp("2020-01-02T03:31:00"),
        received_cutoff=pd.Timestamp("2020-01-02T04:00:00"),
        filters=dataset.qc_filters,
    )
    assert 0 <= len(obs) < 50  # near-empty by construction (hourly cadence)
    for r in obs.records:
        assert r.qc_flag in dataset.qc_filters.allow_dd_flags


def test_zero_observation_sample_path(dataset, monkeypatch):
    """Samples with no observations must yield well-formed zero-size tensors."""
    import offgrid_da.data.dataset as ds_mod

    class _EmptySet:
        records = []
        window_start = None
        window_end = None
        qc_report = None

    monkeypatch.setattr(ds_mod, "metar_to_observation_set", lambda *a, **k: _EmptySet())
    s = dataset[0]
    assert s["obs_values"].shape == (0,)
    assert s["obs_locations"].shape == (0, 3)
    assert s["background"].shape == (1, 121, 240)
    assert s["reference"].shape == (1, 121, 240)
