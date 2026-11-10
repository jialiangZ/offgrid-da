"""Single-cycle DA dataset (plan section 3.8).

``dataset[i]`` returns a dict of tensors:

    background            [C, 121, 240]   float32  (Pangu forecast, physical units)
    background_normalized [C, 121, 240]   float32  (when train norms are available)
    reference             [C_t, 121, 240] float32  (ERA5 reference, physical units)
    reference_normalized  [C_t, 121, 240] float32  (when train norms are available)
    obs_values            [N_obs]         float32  (context stations, physical units)
    obs_values_normalized [N_obs]         float32  (when train norms are available)
    heldout_obs_*         variable-length held-out station diagnostics
    obs_variables         [N_obs]         int64
    obs_locations         [N_obs, 3]      float32  (lat, lon[0,360), elevation_m)
    obs_time_offsets      [N_obs]         float32  (timeObs - valid_time, minutes)
    obs_sources           [N_obs]         int64
    obs_error             [N_obs]         float32  (instrument + representativeness std)
    obs_quality           [N_obs]         int64    (DD-derived quality code)
    obs_station_role      [N_obs]         int64    (0=context, 1=heldout)
    valid_time            pd.Timestamp (metadata)
    sample_id             str          (metadata)

Samples with zero observations are valid and must be handled downstream
(Test B in plan section 4.7).
"""
from __future__ import annotations

from pathlib import Path

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
        norm_stats: NormStats | None = None,
        require_normalization: bool = False,
        cache_index: bool = True,
    ) -> None:
        self.cfg = config or load_config()
        self.registry = registry or DEFAULT_REGISTRY
        norm_path = self.cfg.get("normalization", {}).get("stats_path")
        if norm_stats is None and norm_path:
            norm_path = Path(norm_path)
            if not norm_path.is_absolute():
                norm_path = Path(__file__).resolve().parents[3] / norm_path
            if norm_path.exists():
                norm_stats = NormStats.load(norm_path)
            elif require_normalization:
                raise FileNotFoundError(
                    f"training normalization statistics not found: {norm_path}; "
                    "run scripts/prepare_train_norms.py first"
                )
        if require_normalization and norm_stats is None:
            raise ValueError("require_normalization=True but no NormStats were provided")
        self.norm_stats = norm_stats

        data_cfg = self.cfg["data"]
        time_cfg = self.cfg["time"]
        self.variables: list[str] = list(self.cfg["variables"]["target_variables"])
        self.lead_hours: int | None = time_cfg.get("lead_time_hours")
        self.analysis_interval_hours = int(time_cfg["analysis_interval_hours"])
        self.obs_window_minutes: int = int(time_cfg["observation_window_minutes"])
        self.latency_minutes: int = int(time_cfg["analysis_latency_minutes"])
        self.representativeness_std = {
            str(name): float(value)
            for name, value in self.cfg.get("observation_error", {})
            .get("representativeness_std", {})
            .items()
        }

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
            self.pangu, self.era5, start, end, lead_hours=self.lead_hours,
            valid_interval_hours=self.analysis_interval_hours,
        )

        # Station pools from all context observations in the window
        station_cfg = self.cfg.get("station_split", {})
        heldout_fraction = float(station_cfg.get("heldout_fraction", 0.2))
        station_seed = int(station_cfg.get("seed", 42))
        all_stations = self._collect_station_names()
        self.context_stations, self.heldout_stations = context_and_heldout_sets(
            all_stations,
            split_name=station_split_name,
            heldout_fraction=heldout_fraction,
            seed=station_seed,
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
            row["init_time"], int(row["lead_time_hours"]), self.variables
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
        )
        # Keep the training context and station-evaluation targets separate.
        context_records = [
            r for r in obs_set.records if r.station_id in self.context_stations
        ]
        heldout_records = [
            r for r in obs_set.records if r.station_id in self.heldout_stations
        ]
        context = self._pack_observations(context_records, valid_time)
        heldout = self._pack_observations(heldout_records, valid_time)

        sample = {
            "background": torch.from_numpy(background),
            "reference": torch.from_numpy(reference),
            "obs_values": torch.from_numpy(context["values"]),
            "obs_variables": torch.from_numpy(context["variables"]),
            "obs_locations": torch.from_numpy(context["locations"]),
            "obs_time_offsets": torch.from_numpy(context["time_offsets"]),
            "obs_sources": torch.from_numpy(context["sources"]),
            "obs_error": torch.from_numpy(context["error"]),
            "obs_error_instrument": torch.from_numpy(context["error_instrument"]),
            "obs_error_representativeness": torch.from_numpy(
                context["error_representativeness"]
            ),
            "obs_quality": torch.from_numpy(context["quality"]),
            "obs_station_role": torch.full(
                (len(context_records),), ROLE_CODE["context"], dtype=torch.int64
            ),
            "heldout_obs_values": torch.from_numpy(heldout["values"]),
            "heldout_obs_variables": torch.from_numpy(heldout["variables"]),
            "heldout_obs_locations": torch.from_numpy(heldout["locations"]),
            "heldout_obs_time_offsets": torch.from_numpy(heldout["time_offsets"]),
            "heldout_obs_sources": torch.from_numpy(heldout["sources"]),
            "heldout_obs_error": torch.from_numpy(heldout["error"]),
            "heldout_obs_error_instrument": torch.from_numpy(
                heldout["error_instrument"]
            ),
            "heldout_obs_error_representativeness": torch.from_numpy(
                heldout["error_representativeness"]
            ),
            "heldout_obs_quality": torch.from_numpy(heldout["quality"]),
            "heldout_obs_station_role": torch.full(
                (len(heldout_records),), ROLE_CODE["heldout"], dtype=torch.int64
            ),
            "context_station_ids": [r.station_id for r in context_records],
            "heldout_station_ids": [r.station_id for r in heldout_records],
            "valid_time": valid_time,
            "sample_id": row["sample_id"],
            "grid_latitude": torch.from_numpy(grid["latitude"].astype(np.float32)),
            "grid_longitude": torch.from_numpy(grid["longitude"].astype(np.float32)),
        }
        if self.norm_stats is not None:
            sample["background_normalized"] = torch.from_numpy(
                self._normalize_grid(background, "background")
            )
            sample["reference_normalized"] = torch.from_numpy(
                self._normalize_grid(reference, "reference")
            )
            sample["obs_values_normalized"] = torch.from_numpy(
                self._normalize_observations(context)
            )
            sample["obs_error_normalized"] = torch.from_numpy(
                self._normalize_observation_error(context)
            )
            sample["heldout_obs_values_normalized"] = torch.from_numpy(
                self._normalize_observations(heldout)
            )
            sample["heldout_obs_error_normalized"] = torch.from_numpy(
                self._normalize_observation_error(heldout)
            )
        if obs_set.qc_report is not None:
            sample["qc_report"] = {
                **obs_set.qc_report.as_dict(),
                "n_context_observations": len(context_records),
                "n_heldout_observations": len(heldout_records),
            }
        return sample

    def _pack_observations(
        self, records, valid_time: pd.Timestamp
    ) -> dict[str, np.ndarray]:
        """Convert scalar records to aligned arrays and effective error scales."""
        n = len(records)
        if not n:
            return {
                "values": np.zeros(0, dtype=np.float32),
                "variables": np.zeros(0, dtype=np.int64),
                "locations": np.zeros((0, 3), dtype=np.float32),
                "time_offsets": np.zeros(0, dtype=np.float32),
                "sources": np.zeros(0, dtype=np.int64),
                "error": np.zeros(0, dtype=np.float32),
                "error_instrument": np.zeros(0, dtype=np.float32),
                "error_representativeness": np.zeros(0, dtype=np.float32),
                "quality": np.zeros(0, dtype=np.int64),
            }
        values = np.array([r.value for r in records], dtype=np.float32)
        variables = np.array([r.variable_id for r in records], dtype=np.int64)
        instrument_error = np.array(
            [self.registry.by_id(int(var)).default_obs_error for var in variables],
            dtype=np.float32,
        )
        representativeness_error = np.array(
            [
                self.representativeness_std.get(
                    self.registry.by_id(int(var)).name, 0.0
                )
                for var in variables
            ],
            dtype=np.float32,
        )
        effective_error = np.hypot(instrument_error, representativeness_error)
        return {
            "values": values,
            "variables": variables,
            "locations": np.array(
                [[r.latitude, r.longitude, r.elevation_m] for r in records],
                dtype=np.float32,
            ),
            "time_offsets": np.array(
                [(r.valid_time - valid_time).total_seconds() / 60.0 for r in records],
                dtype=np.float32,
            ),
            "sources": np.array([r.source_id for r in records], dtype=np.int64),
            "error": effective_error.astype(np.float32),
            "error_instrument": instrument_error,
            "error_representativeness": representativeness_error,
            "quality": np.array([r.quality_code for r in records], dtype=np.int64),
        }

    def _normalize_grid(self, field: np.ndarray, group: str) -> np.ndarray:
        if field.shape[0] != len(self.variables):
            raise ValueError("grid channels and configured target variables are misaligned")
        return np.stack(
            [
                self.norm_stats.normalize(group, variable, field[channel])
                for channel, variable in enumerate(self.variables)
            ],
            axis=0,
        )

    def _normalize_observations(
        self, observations: dict[str, np.ndarray]
    ) -> np.ndarray:
        values = observations["values"]
        variables = observations["variables"]
        normalized = np.empty_like(values, dtype=np.float32)
        for variable_id in np.unique(variables):
            spec = self.registry.by_id(int(variable_id))
            mask = variables == variable_id
            normalized[mask] = self.norm_stats.normalize(
                "background", spec.name, values[mask]
            )
        return normalized

    def _normalize_observation_error(
        self, observations: dict[str, np.ndarray]
    ) -> np.ndarray:
        errors = observations["error"]
        variables = observations["variables"]
        normalized = np.empty_like(errors, dtype=np.float32)
        for variable_id in np.unique(variables):
            spec = self.registry.by_id(int(variable_id))
            mask = variables == variable_id
            normalized[mask] = errors[mask] / np.float32(
                self.norm_stats.std("background", spec.name)
            )
        return normalized

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
