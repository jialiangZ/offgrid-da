"""Lazy, versioned sample cache for the ArchesWeather single-cycle experiment."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from .arches_dataset import ArchesSingleCycleDADataset
from .stations import station_role
from ..utils.config import load_config


class StationMembership:
    def __init__(self, role, name, fraction, seed):
        self.role, self.name, self.fraction, self.seed = role, name, fraction, seed

    def __contains__(self, station):
        return station_role(str(station), self.name, self.fraction, self.seed) == self.role


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LazyArchesDataset(ArchesSingleCycleDADataset):
    """Construct only an index; load/cache one physical and normalized sample on demand."""

    def _collect_station_names(self):
        # Parent construction must not scan observations. Membership is a pure hash.
        return []

    def __init__(self, root, split, cache_root=None):
        root = Path(root).resolve()
        if split not in ("train", "val"):
            raise ValueError("This experiment deliberately accepts train and val only")
        self.split = split
        suffix = "train_2018_2020" if split == "train" else "val_2021"
        manifest = root / f"outputs/arches_backgrounds_{suffix}/manifest.json"
        norm = root / "outputs/normalization/arches_train_2018_2020.json"
        cfg = load_config()
        window = cfg["splits"][split]
        super().__init__(manifest, config=cfg, start=window["start"], end=window["end"],
                         normalization_path=norm, station_split_name="default")
        import pandas as pd
        expected = pd.date_range(window["start"], window["end"], freq="6h", inclusive="left")
        if not self.index.valid_time.reset_index(drop=True).equals(pd.Series(expected)):
            raise ValueError(f"Incomplete or off-cadence {split} forecast index")
        station = cfg.get("station_split", {})
        seed = int(station.get("seed", 42))
        fraction = float(station.get("heldout_fraction", .2))
        self.context_stations = StationMembership("context", "default", fraction, seed)
        self.heldout_stations = StationMembership("heldout", "default", fraction, seed)
        source_paths = [Path(__file__), Path(__file__).with_name("arches_dataset.py"),
                        Path(__file__).with_name("dataset.py"), Path(__file__).with_name("stations.py"),
                        Path(__file__).with_name("observations.py"), Path(__file__).with_name("qc.py")]
        provenance = {"manifest": sha256(manifest), "normalization": sha256(norm),
                      "config": cfg, "sources": {p.name: sha256(p) for p in source_paths}}
        self.signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        self.cache_dir = Path(cache_root or root / "outputs/da_sample_cache") / self.signature
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        metadata = self.cache_dir / "provenance.json"
        if not metadata.exists():
            atomic_json(metadata, provenance)

    def sample_path(self, index):
        return self.cache_dir / (self.index.iloc[index]["sample_id"] + ".pt")

    def __getitem__(self, index):
        path = self.sample_path(index)
        if path.exists():
            sample = torch.load(path, map_location="cpu", weights_only=False)
            if sample.get("cache_signature") != self.signature:
                raise ValueError(f"Invalid cached sample provenance: {path}")
            return sample
        sample = super().__getitem__(index)
        self.validate_sample(sample)
        sample["cache_signature"] = self.signature
        fd, name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=self.cache_dir)
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save(sample, stream)
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        return sample

    def validate_sample(self, sample):
        for key in ("background", "reference", "background_normalized", "reference_normalized"):
            if sample[key].shape != (1, 121, 240) or not torch.isfinite(sample[key]).all():
                raise ValueError(f"Invalid {key}: {sample['sample_id']}")
        mean = self.norm_stats.mean("background", "2m_temperature")
        std = self.norm_stats.std("background", "2m_temperature")
        torch.testing.assert_close(sample["reference_normalized"], (sample["reference"] - mean) / std)
        if set(sample["context_station_ids"]) & set(sample["heldout_station_ids"]):
            raise ValueError("Station leakage")
        for name in sample["context_station_ids"]:
            if name not in self.context_stations:
                raise ValueError("Wrong context station role")
        for name in sample["heldout_station_ids"]:
            if name not in self.heldout_stations:
                raise ValueError("Wrong held-out station role")
        for prefix in ("obs", "heldout_obs"):
            n = len(sample[f"{prefix}_values"])
            if sample[f"{prefix}_locations"].shape != (n, 3):
                raise ValueError("Misaligned observation arrays")
            for key in ("values", "values_normalized", "locations", "time_offsets", "error"):
                if not torch.isfinite(sample[f"{prefix}_{key}"]).all():
                    raise ValueError("Nonfinite observations")
            offsets = sample[f"{prefix}_time_offsets"]
            if not ((offsets > -60) & (offsets <= 0)).all():
                raise ValueError("Observation time window violation")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

