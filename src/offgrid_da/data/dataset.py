"""Single-cycle DA dataset (plan section 3.8).

``dataset[i]`` returns a dict of tensors:

    background            [C, 121, 240]   float32  (Pangu 6h forecast, physical units)
    reference             [C_t, 121, 240] float32  (ERA5 analysis, physical units)
    obs_values            [N_obs]         float32  (physical units)
    obs_variables         [N_obs]         int64
    obs_locations         [N_obs, 3]      float32  (lat, lon[0,360), elevation_m)
    obs_time_offsets      [N_obs]         float32  (timeObs - valid_time, minutes)
    obs_sources           [N_obs]         int64
    obs_error             [N_obs]         float32  (assumed obs error std)
    obs_quality           [N_obs]         int64    (DD-derived quality code)
    obs_station_role      [N_obs]         int64    (0=context, 1=heldout)
    valid_time            pd.Timestamp (metadata)
    sample_id             str          (metadata)

Samples with zero observations are valid and must be handled downstream
(Test B in plan section 4.7).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.config import load_config
from ..utils.normalization import NormStats
from .alignment import build_sample_index, sample_background_at_obs
from .analysis import ERA5Reader
from .forecast import PanguReader
from .metar import MetarReader
from .observations import metar_to_observation_set
from .qc import QCFilters
from .stations import context_and_heldout_sets
from .variables import DEFAULT_REGISTRY, VariableRegistry

ROLE_CODE = {"context": 0, "heldout": 1}


class SingleCycleDADataset(Dataset):
    """Real-data single-cycle DA samples (Pangu background + METAR + ERA5)."""

    def __init__(
        self,
        config: dict | None = None,
        split: str = "minimal",
        start: str | None = None,
        end: str | None = None,
        registry: VariableRegistry | None = None,
        station_split_name: str = "default",
        norm_stats: NormStats | None = None,   # optional; kept physical units if None
        cache_index: bool = True,
    ) -> None:
        self.cfg = config or load_config()
        self.registry = registry or DEFAULT_REGISTRY
        self.norm_stats = norm_stats

        data_cfg = self.cfg["data"]
        time_cfg = self.cfg["time"]
        self.variables: list[str] = list(self.cfg["variables"]["target_variables"])
        self.lead_hours: int = int(time_cfg["lead_time_hours"])
        self.obs_window_minutes: int = int(time_cfg["observation_window_minutes"])
        self.latency_minutes: int = int(time_cfg["analysis_latency_minutes"])

        self.pangu = PanguReader(data_cfg["pangu"]["url"])
        self.era5 = ERA5Reader(data_cfg["era5"]["url"])
        self.metar = MetarReader(
            data_cfg["metar"]["url"], data_cfg["metar"]["layout"], data_cfg["cache_dir"]
        )

        if split == "minimal":
            window = self.cfg["minimal_window"]
            start = start or window["start"]
            end = end or window["end"]
        else:
            window = self.cfg["splits"][split]
            start = start or window["start"]
            end = end or window["end"]

        self.index = build_sample_index(
            self.pangu, self.era5, start, end, lead_hours=self.lead_hours
        )

        # Station pools from all context observations in the window
        station_cfg = self.cfg.get("station_split", {})
        heldout_fraction = float(station_cfg.get("heldout_fraction", 0.2))
        all_stations = self._collect_station_names()
        self.context_stations, self.heldout_stations = context_and_heldout_sets(
            all_stations, split_name=station_split_name, heldout_fraction=heldout_fraction
        )
        self.station_split_name = station_split_name
        self.qc_filters = QCFilters()

    # ------------------------------------------------------------------ #

    def _collect_station_names(self) -> list[str]:
        """Scan the window's parquet partitions once to enumerate stations."""
        names: set[str] = set()
        for end in self.index["observation_window_end"]:
            start = end - pd.Timedelta(minutes=self.obs_window_minutes)
            df = self.metar.read_window(start, end)
            names.update(df["stationName"].astype(str).unique())
        return sorted(names)

    def __len__(self) -> int:
        return len(self.index)

    def sample_table(self) -> pd.DataFrame:
        return self.index.copy()

    def __getitem__(self, idx: int) -> dict:
        row = self.index.iloc[idx]
        valid_time: pd.Timestamp = row["valid_time"]

        background = self.pangu.read_background(
            row["init_time"], self.lead_hours, self.variables
        )
        reference = self.era5.read_analysis(valid_time, self.variables)
        grid = self.pangu.grid()

        window_start = valid_time - pd.Timedelta(minutes=self.obs_window_minutes)
        received_cutoff = valid_time + pd.Timedelta(minutes=self.latency_minutes)

        obs_set = metar_to_observation_set(
            self.metar,
            self.registry,
            self.variables,
            window_start=window_start,
            window_end=valid_time,
            received_cutoff=received_cutoff,
            filters=self.qc_filters,
            station_keep=self.context_stations,  # held-out stations never enter inputs
        )
        records = obs_set.records
        n = len(records)

        if n > 0:
            obs_values = np.array([r.value for r in records], dtype=np.float32)
            obs_vars = np.array([r.variable_id for r in records], dtype=np.int64)
            obs_loc = np.array(
                [[r.latitude, r.longitude, r.elevation_m] for r in records], dtype=np.float32
            )
            obs_dt = np.array(
                [(r.valid_time - valid_time).total_seconds() / 60.0 for r in records],
                dtype=np.float32,
            )
            obs_src = np.array([r.source_id for r in records], dtype=np.int64)
            spec = self.registry.by_id(int(obs_vars[0]))
            obs_err = np.full(n, spec.default_obs_error, dtype=np.float32)
            obs_q = np.array([r.quality_code for r in records], dtype=np.int64)
            obs_role = np.array(
                [ROLE_CODE[("context" if s.station_id in self.context_stations else "heldout")]
                 for s in records],
                dtype=np.int64,
            )
        else:
            obs_values = np.zeros(0, dtype=np.float32)
            obs_vars = np.zeros(0, dtype=np.int64)
            obs_loc = np.zeros((0, 3), dtype=np.float32)
            obs_dt = np.zeros(0, dtype=np.float32)
            obs_src = np.zeros(0, dtype=np.int64)
            obs_err = np.zeros(0, dtype=np.float32)
            obs_q = np.zeros(0, dtype=np.int64)
            obs_role = np.zeros(0, dtype=np.int64)

        sample = {
            "background": torch.from_numpy(background),
            "reference": torch.from_numpy(reference),
            "obs_values": torch.from_numpy(obs_values),
            "obs_variables": torch.from_numpy(obs_vars),
            "obs_locations": torch.from_numpy(obs_loc),
            "obs_time_offsets": torch.from_numpy(obs_dt),
            "obs_sources": torch.from_numpy(obs_src),
            "obs_error": torch.from_numpy(obs_err),
            "obs_quality": torch.from_numpy(obs_q),
            "obs_station_role": torch.from_numpy(obs_role),
            "valid_time": valid_time,
            "sample_id": row["sample_id"],
            "grid_latitude": torch.from_numpy(grid["latitude"].astype(np.float32)),
            "grid_longitude": torch.from_numpy(grid["longitude"].astype(np.float32)),
        }
        if obs_set.qc_report is not None:
            sample["qc_report"] = obs_set.qc_report.as_dict()
        return sample

    # ------------------------------------------------------------------ #

    def innovations(self, idx: int, use_reference: bool = False) -> np.ndarray:
        """obs - background_at_obs (default) or obs - ERA5_at_obs for analysis."""
        sample = self[idx]
        field = sample["reference" if use_reference else "background"][0].numpy()
        lat = sample["obs_locations"][:, 0].numpy()
        lon = sample["obs_locations"][:, 1].numpy()
        bg_at_obs = sample_background_at_obs(
            field, sample["grid_latitude"].numpy(), sample["grid_longitude"].numpy(), lat, lon
        )
        return sample["obs_values"].numpy().astype(np.float64) - bg_at_obs
