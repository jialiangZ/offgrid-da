"""Normalization statistics (plan section 3.9).

Statistics are computed ONLY from the requested (training) samples and saved
as JSON. For the minimal Phase-1 window they describe Jan-2020; full
2018-2020 statistics will be recomputed before real training (documented in
docs/DATA_SPEC.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NormStats:
    """Per-variable mean/std for background and reference fields (physical units)."""

    def __init__(self) -> None:
        self.stats: dict[str, dict[str, dict[str, float]]] = {}

    def update(self, group: str, variable: str, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"no finite values for {group}/{variable}")
        self.stats.setdefault(group, {}).setdefault(
            variable, {"n": 0, "mean": 0.0, "m2": 0.0}
        )
        entry = self.stats[group][variable]
        for v in values.ravel():  # Welford online update, exact enough at this scale
            entry["n"] += 1
            delta = v - entry["mean"]
            entry["mean"] += delta / entry["n"]
            entry["m2"] += delta * (v - entry["mean"])

    def mean(self, group: str, variable: str) -> float:
        return float(self.stats[group][variable]["mean"])

    def std(self, group: str, variable: str) -> float:
        entry = self.stats[group][variable]
        return float(np.sqrt(entry["m2"] / max(entry["n"] - 1, 1)))

    def to_dict(self) -> dict:
        out: dict[str, dict[str, dict[str, float]]] = {}
        for group, variables in self.stats.items():
            out[group] = {}
            for var, entry in variables.items():
                out[group][var] = {
                    "mean": float(entry["mean"]),
                    "std": float(np.sqrt(entry["m2"] / max(entry["n"] - 1, 1))),
                    "n": int(entry["n"]),
                }
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "NormStats":
        obj = cls()
        with open(path) as f:
            loaded = json.load(f)
        for group, variables in loaded.items():
            for var, entry in variables.items():
                obj.stats.setdefault(group, {})[var] = {
                    "n": int(entry["n"]),
                    "mean": float(entry["mean"]),
                    "m2": (max(int(entry["n"]) - 1, 1)) * float(entry["std"]) ** 2,
                }
        return obj
