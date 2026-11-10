"""Normalization statistics (plan section 3.9).

Statistics are computed ONLY from the requested (training) samples and saved
as JSON. The configured 2018-2020 statistics are shared by train, validation,
and test samples; each sample keeps physical fields and adds normalized copies.
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
        batch_n = int(values.size)
        batch_mean = float(values.mean())
        centered = values - batch_mean
        batch_m2 = float(np.sum(centered * centered, dtype=np.float64))

        old_n = int(entry["n"])
        if old_n == 0:
            entry["n"] = batch_n
            entry["mean"] = batch_mean
            entry["m2"] = batch_m2
            return

        total_n = old_n + batch_n
        delta = batch_mean - entry["mean"]
        entry["mean"] += delta * batch_n / total_n
        entry["m2"] += batch_m2 + delta * delta * old_n * batch_n / total_n
        entry["n"] = total_n

    def merge(self, other: "NormStats") -> None:
        """Merge sufficient statistics from another independently computed batch."""
        for group, variables in other.stats.items():
            self.stats.setdefault(group, {})
            for variable, batch in variables.items():
                old = self.stats[group].get(variable)
                batch_n = int(batch["n"])
                if old is None or int(old["n"]) == 0:
                    self.stats[group][variable] = {
                        "n": batch_n,
                        "mean": float(batch["mean"]),
                        "m2": float(batch["m2"]),
                    }
                    continue
                old_n = int(old["n"])
                total_n = old_n + batch_n
                delta = float(batch["mean"]) - old["mean"]
                old["mean"] += delta * batch_n / total_n
                old["m2"] += float(batch["m2"]) + delta * delta * old_n * batch_n / total_n
                old["n"] = total_n

    def mean(self, group: str, variable: str) -> float:
        return float(self.stats[group][variable]["mean"])

    def std(self, group: str, variable: str) -> float:
        entry = self.stats[group][variable]
        return float(np.sqrt(entry["m2"] / max(entry["n"] - 1, 1)))

    def normalize(self, group: str, variable: str, values: np.ndarray) -> np.ndarray:
        """Standardize values with the frozen training-period mean and std."""
        mean = self.mean(group, variable)
        std = self.std(group, variable)
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"invalid normalization std for {group}/{variable}: {std}")
        values = np.asarray(values, dtype=np.float32)
        return ((values - np.float32(mean)) / np.float32(std)).astype(np.float32)

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
