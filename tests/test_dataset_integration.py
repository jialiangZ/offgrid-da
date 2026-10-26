"""End-to-end DataLoader smoke test with real data (plan section 3.8)."""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from offgrid_da.data.collate import single_cycle_collate

pytestmark = pytest.mark.requires_gcs


def test_dataloader_real_batch(dataset):
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=single_cycle_collate)
    batch = next(iter(loader))
    assert batch["background"].shape == (2, 1, 121, 240)
    assert batch["reference"].shape == (2, 1, 121, 240)
    n = batch["observation_mask"].shape[1]
    assert batch["observation_mask"].shape == (2, n)
    assert batch["obs_locations"].shape == (2, n, 3)
    # mask sums must equal per-sample obs counts
    counts = batch["observation_mask"].sum(dim=1)
    assert (counts > 100).all()
    # observations finite
    m = batch["observation_mask"]
    vals = batch["obs_values"][m]
    assert torch.isfinite(vals).all()
    # dtypes ready for training
    assert batch["background"].dtype == torch.float32
    assert batch["obs_variables"].dtype == torch.int64


def test_two_batches_iterate(dataset):
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=single_cycle_collate)
    it = iter(loader)
    b1 = next(it)
    b2 = next(it)
    assert b1["sample_ids"] != b2["sample_ids"]
    assert b1["background"].shape == b2["background"].shape
