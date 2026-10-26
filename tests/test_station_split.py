"""Station split tests (plan section 5.7)."""
from __future__ import annotations

import numpy as np
import pytest

from offgrid_da.data.stations import context_and_heldout_sets, station_role

pytestmark = pytest.mark.requires_gcs


def test_split_deterministic():
    a = station_role("KJFK", "default", 0.2)
    b = station_role("KJFK", "default", 0.2)
    assert a == b


def test_split_disjoint_and_complete(dataset):
    assert not (dataset.context_stations & dataset.heldout_stations)
    n_total = len(dataset.context_stations) + len(dataset.heldout_stations)
    assert n_total > 1000
    frac = len(dataset.heldout_stations) / n_total
    assert 0.15 < frac < 0.25  # ~20% by construction


def test_heldout_never_in_model_input(dataset):
    for idx in (0, 2, len(dataset) - 1):
        s = dataset[idx]
        roles = s["obs_station_role"].numpy()
        assert (roles == 0).all()  # only context stations enter the input


def test_heldout_obs_available_for_independent_eval(dataset):
    """Held-out stations still report in the raw windows -> usable for evaluation."""
    from offgrid_da.data.observations import metar_to_observation_set
    import pandas as pd

    row = dataset.sample_table().iloc[0]
    vt = row["valid_time"]
    obs = metar_to_observation_set(
        dataset.metar, dataset.registry, dataset.variables,
        window_start=vt - pd.Timedelta(minutes=60), window_end=vt,
        received_cutoff=vt + pd.Timedelta(minutes=30),
        filters=dataset.qc_filters,
        station_keep=dataset.heldout_stations,
    )
    assert len(obs) > 100  # held-out pool has plenty of reports
