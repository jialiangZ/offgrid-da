"""Matched full-state/T2M-only input ablation using identical 82-slot networks."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .arches_training import LazyArchesDataset
from .collate import single_cycle_collate

SURFACE = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'mean_sea_level_pressure']
LEVEL = ['geopotential', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity', 'vertical_velocity']
PRESSURE = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
CHANNELS = SURFACE + [f'{v}@{p}hPa' for v in LEVEL for p in PRESSURE]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_state(path, expected_sha=None):
    path = Path(path)
    if expected_sha and digest(path) != expected_sha:
        raise ValueError('Full-state background checksum mismatch')
    with np.load(path, allow_pickle=False) as data:
        surface, level = data['surface'], data['level']
        if surface.shape != (4, 1, 121, 240) or level.shape != (6, 13, 121, 240):
            raise ValueError('Incorrect full-state channel layout')
        np.testing.assert_allclose(data['latitude'], np.linspace(90, -90, 121))
        np.testing.assert_allclose(data['longitude'], np.arange(240) * 1.5)
        np.testing.assert_array_equal(data['pressure_hpa'], PRESSURE)
        state = np.concatenate([surface[:, 0], level.reshape(78, 121, 240)]).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError('Nonfinite full state')
    return state


class FullStateInputDataset(Dataset):
    def __init__(self, root, split, experiment, input_mode):
        if input_mode not in ('t2m_masked', 'full82'):
            raise ValueError('Invalid input mode')
        self.input_mode = input_mode
        self.experiment = Path(experiment)
        legacy = LazyArchesDataset(root, split)
        self.norm_stats = legacy.norm_stats
        self.context_stations = legacy.context_stations
        self.heldout_stations = legacy.heldout_stations
        self.cache_paths = [legacy.sample_path(i) for i in range(len(legacy))]
        if not all(p.is_file() for p in self.cache_paths):
            raise ValueError('Validated T2M/observation cache must already be complete')
        manifest_path = self.experiment / 'backgrounds/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest['channel_names'] != CHANNELS:
            raise ValueError('Incorrect channel order')
        rows = {pd.Timestamp(x['valid_utc']).tz_convert('UTC').tz_localize(None): x for x in manifest['samples']}
        self.records = [rows[t] for t in legacy.index.valid_time]
        self.background_dir = manifest_path.parent
        norm_path = self.experiment / 'normalization.json'
        stats = json.loads(norm_path.read_text())
        if stats['train_cycles'] != 4384 or stats['channel_names'] != CHANNELS:
            raise ValueError('Incomplete training normalization')
        if stats['legacy_t2m_normalization_sha256'] != digest(Path(root) / 'outputs/normalization/arches_train_2018_2020.json'):
            raise ValueError('T2M normalization changed')
        self.mean = torch.tensor(stats['input_mean'], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(stats['input_std'], dtype=torch.float32)[:, None, None]
        if not (torch.isfinite(self.std).all() and (self.std > 0).all()):
            raise ValueError('Invalid input scales')
        provenance = {'base_signature': legacy.signature, 'full_manifest': digest(manifest_path),
                      'normalization': digest(norm_path), 'input_mode': input_mode,
                      'transform_source': digest(__file__)}
        self.signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        self.legacy_signature = legacy.signature

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sample = torch.load(self.cache_paths[index], map_location='cpu', weights_only=False)
        if sample.get('cache_signature') != self.legacy_signature:
            raise ValueError('Original sample cache changed')
        row = self.records[index]
        if pd.Timestamp(sample['valid_time']) != pd.Timestamp(row['valid_utc']).tz_convert('UTC').tz_localize(None):
            raise ValueError('Observation/full-state time mismatch')
        raw = torch.from_numpy(read_state(self.background_dir / row['file'], row['sha256']))
        normalized = (raw - self.mean) / self.std
        sample['background'] = raw[2:3].clone()
        sample['background_normalized'] = normalized[2:3].clone()
        if self.input_mode == 't2m_masked':
            masked = torch.zeros_like(normalized)
            masked[2] = normalized[2]
            normalized = masked
        sample['grid_state'] = normalized
        return sample


def full_state_collate(samples):
    result = single_cycle_collate(samples)
    result['grid_state'] = torch.stack([s['grid_state'] for s in samples])
    return result

