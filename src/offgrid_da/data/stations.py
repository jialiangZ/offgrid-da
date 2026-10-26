"""Deterministic station split (plan section 5.7).

A station's role is a deterministic function of (split_name, station_id),
so the same station layout is reproduced in every process without storing
the assignment. Held-out stations never enter the model input at train or
test time; their observations serve independent station evaluation only.
"""
from __future__ import annotations

import hashlib

import pandas as pd


def station_role(station_id: str, split_name: str = "default", heldout_fraction: float = 0.2) -> str:
    """Stable hash split: 'context' or 'heldout'."""
    key = f"{split_name}:{station_id}".encode()
    h = hashlib.sha256(key).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return "heldout" if frac < heldout_fraction else "context"


def split_stations(
    station_ids: pd.Series | list[str],
    split_name: str = "default",
    heldout_fraction: float = 0.2,
) -> pd.Series:
    ids = pd.Series(station_ids if isinstance(station_ids, list) else station_ids)
    return ids.map(lambda s: station_role(str(s), split_name, heldout_fraction))


def context_and_heldout_sets(
    station_ids,
    split_name: str = "default",
    heldout_fraction: float = 0.2,
) -> tuple[set[str], set[str]]:
    """Return (context_ids, heldout_ids) for an iterable of station ids."""
    mapping = {
        s: station_role(s, split_name, heldout_fraction)
        for s in sorted(set(map(str, station_ids)))
    }
    ctx = {s for s, r in mapping.items() if r == "context"}
    held = {s for s, r in mapping.items() if r == "heldout"}
    return ctx, held
