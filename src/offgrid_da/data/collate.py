"""Collate variable-length observation samples into padded batches.

Padding must never be interpreted as real observations: every padded token is
masked out. Context observations are model inputs; held-out observations are
carried separately for independent station-space diagnostics.
"""
from __future__ import annotations

import torch


_INT_SUFFIXES = ("variables", "sources", "quality", "station_role")
_FLOAT_SUFFIXES = (
    "values",
    "time_offsets",
    "error",
    "error_instrument",
    "error_representativeness",
    "values_normalized",
    "error_normalized",
)


def _collate_observation_group(batch: list[dict], prefix: str) -> dict:
    value_key = f"{prefix}_values"
    present = [value_key in sample for sample in batch]
    if not any(present):
        return {}
    if not all(present):
        raise ValueError(f"inconsistent samples: missing {value_key}")

    n_max = max(sample[value_key].shape[0] for sample in batch)
    out: dict = {}
    for suffix in _INT_SUFFIXES:
        key = f"{prefix}_{suffix}"
        if all(key in sample for sample in batch):
            out[key] = torch.zeros(len(batch), n_max, dtype=torch.int64)
    for suffix in _FLOAT_SUFFIXES:
        key = f"{prefix}_{suffix}"
        if all(key in sample for sample in batch):
            out[key] = torch.zeros(len(batch), n_max, dtype=torch.float32)
        elif any(key in sample for sample in batch):
            raise ValueError(f"inconsistent samples: some omit {key}")
    loc_key = f"{prefix}_locations"
    out[loc_key] = torch.zeros(len(batch), n_max, 3, dtype=torch.float32)
    mask = torch.zeros(len(batch), n_max, dtype=torch.bool)

    for i, sample in enumerate(batch):
        n = sample[value_key].shape[0]
        for key, tensor in out.items():
            if key == loc_key or key.endswith("_mask"):
                continue
            if n:
                tensor[i, :n] = sample[key]
        if n:
            out[loc_key][i, :n] = sample[loc_key]
        mask[i, :n] = True

    mask_key = "observation_mask" if prefix == "obs" else f"{prefix}_mask"
    out[mask_key] = mask
    return out


def single_cycle_collate(batch: list[dict]) -> dict:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    out: dict = {}
    for key in ("background", "reference"):
        out[key] = torch.stack([sample[key] for sample in batch], dim=0)

    for key in ("background_normalized", "reference_normalized"):
        present = [key in sample for sample in batch]
        if any(present) and not all(present):
            raise ValueError(f"inconsistent samples: some omit {key}")
        if all(present):
            out[key] = torch.stack([sample[key] for sample in batch], dim=0)

    out.update(_collate_observation_group(batch, "obs"))
    out.update(_collate_observation_group(batch, "heldout_obs"))
    out["valid_times"] = [sample["valid_time"] for sample in batch]
    out["sample_ids"] = [sample["sample_id"] for sample in batch]
    out["grid_latitude"] = batch[0]["grid_latitude"]
    out["grid_longitude"] = batch[0]["grid_longitude"]
    if all("context_station_ids" in sample for sample in batch):
        out["context_station_ids"] = [sample["context_station_ids"] for sample in batch]
        out["heldout_station_ids"] = [sample["heldout_station_ids"] for sample in batch]
    return out
