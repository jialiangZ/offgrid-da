"""Collate variable-length observation samples into padded batches.

Padding must never be interpreted as real observations: every padded token is
masked out by ``observation_mask`` (plan section 3.8).
"""
from __future__ import annotations

import torch


def single_cycle_collate(batch: list[dict]) -> dict:
    b = len(batch)
    out: dict = {}

    for key in ("background", "reference"):
        out[key] = torch.stack([s[key] for s in batch], dim=0)

    varlen_int = ("obs_variables", "obs_sources", "obs_quality", "obs_station_role")
    varlen_float = ("obs_values", "obs_time_offsets", "obs_error")
    n_max = max(s["obs_values"].shape[0] for s in batch)

    for key in varlen_int:
        out[key] = torch.zeros(b, n_max, dtype=torch.int64)
    for key in varlen_float:
        out[key] = torch.zeros(b, n_max, dtype=torch.float32)
    out["obs_locations"] = torch.zeros(b, n_max, 3, dtype=torch.float32)
    mask = torch.zeros(b, n_max, dtype=torch.bool)

    for i, s in enumerate(batch):
        n = s["obs_values"].shape[0]
        for key in varlen_int + varlen_float:
            if n > 0:
                out[key][i, :n] = s[key]
        if n > 0:
            out["obs_locations"][i, :n] = s["obs_locations"]
        mask[i, :n] = True

    out["observation_mask"] = mask
    out["valid_times"] = [s["valid_time"] for s in batch]
    out["sample_ids"] = [s["sample_id"] for s in batch]
    out["grid_latitude"] = batch[0]["grid_latitude"]
    out["grid_longitude"] = batch[0]["grid_longitude"]
    return out
