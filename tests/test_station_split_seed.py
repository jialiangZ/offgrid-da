"""Seed behavior for deterministic station-role assignments."""
from __future__ import annotations

from offgrid_da.data.stations import context_and_heldout_sets, station_role


def test_station_assignment_is_stable_for_the_same_seed() -> None:
    assert station_role("KJFK", "default", 0.2, seed=42) == station_role(
        "KJFK", "default", 0.2, seed=42
    )


def test_changing_seed_changes_assignments() -> None:
    station_ids = [f"STATION-{i}" for i in range(256)]
    assert any(
        station_role(station_id, "default", 0.2, seed=42)
        != station_role(station_id, "default", 0.2, seed=43)
        for station_id in station_ids
    )


def test_seed_is_used_by_station_set_builder() -> None:
    station_ids = [f"STATION-{i}" for i in range(256)]
    context_42, heldout_42 = context_and_heldout_sets(
        station_ids, split_name="default", heldout_fraction=0.2, seed=42
    )
    context_43, heldout_43 = context_and_heldout_sets(
        station_ids, split_name="default", heldout_fraction=0.2, seed=43
    )

    assert context_42 | heldout_42 == set(station_ids)
    assert context_43 | heldout_43 == set(station_ids)
    assert heldout_42 != heldout_43
