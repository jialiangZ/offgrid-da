"""Fit per-channel train-only scales, then validate all input/observation pairs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from offgrid_da.data.arches_training import LazyArchesDataset, atomic_json
from offgrid_da.data.full_state_data import FullStateInputDataset, CHANNELS, digest, read_state

DATA = {}


def initialize(experiment):
    torch.set_num_threads(1)
    for split in ('train', 'val'):
        DATA[split] = FullStateInputDataset(ROOT, split, experiment, 'full82')


def validate_one(task):
    split, i = task
    sample = DATA[split][i]
    LazyArchesDataset.validate_sample(DATA[split], sample)
    if sample['grid_state'].shape != (82, 121, 240) or not torch.isfinite(sample['grid_state']).all():
        raise ValueError('Invalid full-state input')
    torch.testing.assert_close(sample['grid_state'][2:3], sample['background_normalized'], rtol=0, atol=0)
    return {'split': split, 'index': i}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    manifest_path = args.experiment / 'backgrounds/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if not manifest['complete'] or manifest['channel_names'] != CHANNELS:
        raise ValueError('Full background generation is incomplete')
    train = [s for s in manifest['samples'] if '2018-01-01T00:00:00Z' <= s['valid_utc'] < '2021-01-01T00:00:00Z']
    expected = pd.date_range('2018-01-01', '2021-01-01', freq='6h', inclusive='left', tz='UTC')
    if not pd.DatetimeIndex([s['valid_utc'] for s in train]).equals(expected):
        raise ValueError('Incorrect training interval')
    count = 0
    mean = np.zeros(82, dtype=np.float64)
    m2 = np.zeros(82, dtype=np.float64)
    for i, sample in enumerate(train):
        values = read_state(manifest_path.parent / sample['file'], sample['sha256']).reshape(82, -1).astype(np.float64)
        batch_n = values.shape[1]
        batch_mean = values.mean(axis=1)
        batch_m2 = np.square(values - batch_mean[:, None]).sum(axis=1)
        delta = batch_mean - mean
        total = count + batch_n
        mean += delta * batch_n / total
        m2 += batch_m2 + delta ** 2 * count * batch_n / total
        count = total
        if (i + 1) % 100 == 0:
            print('NORMALIZATION', i + 1, len(train), flush=True)
    std = np.sqrt(m2 / (count - 1))
    if not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError('Degenerate channel scales')
    old_path = ROOT / 'outputs/normalization/arches_train_2018_2020.json'
    old = json.loads(old_path.read_text())['background']['2m_temperature']
    input_mean, input_std = mean.copy(), std.copy()
    input_mean[2], input_std[2] = old['mean'], old['std']
    stats = {'train_cycles': 4384, 'train_interval': ['2018-01-01', '2021-01-01'],
             'channel_names': CHANNELS, 'raw_train_mean': mean.tolist(), 'raw_train_std': std.tolist(),
             'input_mean': input_mean.tolist(), 'input_std': input_std.tolist(),
             'legacy_t2m_normalization_sha256': digest(old_path), 'full_manifest_sha256': digest(manifest_path),
             'validation_or_test_used': False, 'grid_points_per_channel': count,
             'T2M_scale': 'Keep existing train-background scale for both input arms, T2M output and observations'}
    atomic_json(args.experiment / 'normalization.json', stats)
    datasets = {s: FullStateInputDataset(ROOT, s, args.experiment, 'full82') for s in ('train', 'val')}
    tasks = [(s, i) for s, data in datasets.items() for i in range(len(data))]
    visited, errors = 0, []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(args.experiment,)) as pool:
        pending = {pool.submit(validate_one, t): t for t in tasks}
        for f in as_completed(pending):
            try:
                f.result()
                visited += 1
            except Exception as error:
                errors.append({'task': pending[f], 'error': repr(error)})
            if (visited + len(errors)) % 100 == 0:
                print('VALIDATION', visited, len(tasks), 'failed', len(errors), flush=True)
    for mode in ('t2m_masked', 'full82'):
        current = {s: FullStateInputDataset(ROOT, s, args.experiment, mode) for s in ('train', 'val')}
        report = {'complete': visited == 5844 and not errors, 'visited': visited, 'errors': errors,
                  'signatures': {s: d.signature for s, d in current.items()}, 'input_mode': mode}
        atomic_json(args.experiment / f'preparation_{mode}.json', report)
    if errors or visited != 5844:
        raise RuntimeError('Full-state data validation failed')
    # Check the controlled input ablation on real train and val samples.
    for split in ('train', 'val'):
        full = FullStateInputDataset(ROOT, split, args.experiment, 'full82')[0]
        masked = FullStateInputDataset(ROOT, split, args.experiment, 't2m_masked')[0]
        torch.testing.assert_close(full['background_normalized'], masked['background_normalized'], rtol=0, atol=0)
        torch.testing.assert_close(full['obs_values_normalized'], masked['obs_values_normalized'], rtol=0, atol=0)
        other = [j for j in range(82) if j != 2]
        assert torch.count_nonzero(masked['grid_state'][other]) == 0
        torch.testing.assert_close(full['grid_state'][2], masked['grid_state'][2], rtol=0, atol=0)
    print('PREPARATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()

