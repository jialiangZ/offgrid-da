"""Padding/masking tests for the custom collate (plan section 3.8)."""
from __future__ import annotations

import torch

from offgrid_da.data.collate import single_cycle_collate


class _FakeSample(dict):
    pass


def _fake(n_obs: int) -> dict:
    return {
        "background": torch.zeros(1, 121, 240),
        "reference": torch.zeros(1, 121, 240),
        "background_normalized": torch.ones(1, 121, 240),
        "reference_normalized": torch.ones(1, 121, 240) * 2,
        "obs_values": torch.arange(n_obs, dtype=torch.float32) + 1,
        "obs_values_normalized": torch.arange(n_obs, dtype=torch.float32) / 10,
        "obs_variables": torch.zeros(n_obs, dtype=torch.int64),
        "obs_locations": torch.ones(n_obs, 3) * 7.0,
        "obs_time_offsets": torch.zeros(n_obs, dtype=torch.float32),
        "obs_sources": torch.zeros(n_obs, dtype=torch.int64),
        "obs_error": torch.ones(n_obs),
        "obs_error_instrument": torch.ones(n_obs),
        "obs_error_representativeness": torch.ones(n_obs) * 2.5,
        "obs_error_normalized": torch.ones(n_obs) * 0.1,
        "obs_quality": torch.full((n_obs,), 3, dtype=torch.int64),
        "obs_station_role": torch.zeros(n_obs, dtype=torch.int64),
        "heldout_obs_values": torch.arange(max(n_obs - 2, 0), dtype=torch.float32) + 10,
        "heldout_obs_values_normalized": torch.arange(max(n_obs - 2, 0), dtype=torch.float32) / 5,
        "heldout_obs_variables": torch.zeros(max(n_obs - 2, 0), dtype=torch.int64),
        "heldout_obs_locations": torch.ones(max(n_obs - 2, 0), 3) * 9.0,
        "heldout_obs_time_offsets": torch.zeros(max(n_obs - 2, 0)),
        "heldout_obs_sources": torch.zeros(max(n_obs - 2, 0), dtype=torch.int64),
        "heldout_obs_error": torch.ones(max(n_obs - 2, 0)) * 2.7,
        "heldout_obs_error_instrument": torch.ones(max(n_obs - 2, 0)),
        "heldout_obs_error_representativeness": torch.ones(max(n_obs - 2, 0)) * 2.5,
        "heldout_obs_error_normalized": torch.ones(max(n_obs - 2, 0)) * 0.2,
        "heldout_obs_quality": torch.full((max(n_obs - 2, 0),), 3, dtype=torch.int64),
        "heldout_obs_station_role": torch.ones(max(n_obs - 2, 0), dtype=torch.int64),
        "context_station_ids": [f"ctx_{i}" for i in range(n_obs)],
        "heldout_station_ids": [f"held_{i}" for i in range(max(n_obs - 2, 0))],
        "valid_time": 0,
        "sample_id": f"fake_{n_obs}",
        "grid_latitude": torch.zeros(121),
        "grid_longitude": torch.zeros(240),
    }


def test_mask_marks_real_tokens():
    batch = single_cycle_collate([_fake(5), _fake(0), _fake(3)])
    mask = batch["observation_mask"]
    assert mask.shape == (3, 5)
    assert mask[0].all()
    assert not mask[1].any()
    assert mask[2].sum() == 3


def test_padding_is_zero_and_excluded():
    batch = single_cycle_collate([_fake(5), _fake(0), _fake(3)])
    # padded slots must be zeroed
    assert (batch["obs_values"][1] == 0).all()
    # real token of sample 2 retained
    assert batch["obs_values"][2, :3].tolist() == [1.0, 2.0, 3.0]
    assert batch["obs_locations"][2, :3].eq(7.0).all()


def test_batch_shapes():
    batch = single_cycle_collate([_fake(4), _fake(9)])
    assert batch["background"].shape == (2, 1, 121, 240)
    assert batch["obs_locations"].shape == (2, 9, 3)
    assert batch["observation_mask"].dtype == torch.bool


def test_permutation_invariance_of_collate_content():
    """Collate output for one sample must not depend on other samples' sizes."""
    a = single_cycle_collate([_fake(3), _fake(100)])
    b = single_cycle_collate([_fake(3), _fake(2)])
    assert torch.equal(a["obs_values"][0, :3], b["obs_values"][0, :3])
    assert torch.equal(a["observation_mask"][0, :3], b["observation_mask"][0, :3])


def test_heldout_observations_are_collated_separately_from_inputs():
    batch = single_cycle_collate([_fake(5), _fake(0)])
    assert batch["obs_values"].shape == (2, 5)
    assert batch["heldout_obs_values"].shape == (2, 3)
    assert batch["heldout_obs_mask"][0].all()
    assert not batch["heldout_obs_mask"][1].any()
    assert "heldout_obs_values" in batch
    assert torch.equal(batch["background_normalized"][0], torch.ones(1, 121, 240))
    assert torch.allclose(
        batch["obs_values_normalized"][0, :5], torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    )
    assert batch["heldout_obs_values"][0, :3].tolist() == [10.0, 11.0, 12.0]
