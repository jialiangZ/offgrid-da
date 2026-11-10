"""Frozen, validation-selected seed-42/43/44 evaluation on the complete 2022 test year."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from offgrid_da.data.arches_dataset import ArchesSingleCycleDADataset
from offgrid_da.data.arches_training import LazyArchesDataset, StationMembership, atomic_json
from offgrid_da.data.collate import single_cycle_collate
from offgrid_da.utils.config import load_config
from train_single_cycle import CachedSamples, make_model, pack
from analyze_single_cycle import circular_block_bootstrap

WORKER_DATA = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_selected_checkpoint(checkpoint, result, config_digest, data_signatures):
    if checkpoint['step'] != result['best_step']:
        raise ValueError('Checkpoint is not the validation-selected best step')
    if checkpoint['signature']['config_sha256'] != config_digest:
        raise ValueError('Checkpoint/config fingerprint mismatch')
    if checkpoint['signature']['data'] != data_signatures:
        raise ValueError('Checkpoint training/validation data fingerprint mismatch')
    if checkpoint['signature']['trial']:
        raise ValueError('A trial checkpoint cannot be used for final testing')


def freeze_protocol(out):
    original = ROOT / 'outputs/single_da_02_t2m_dim32_main'
    fingerprints = json.loads((original / 'implementation.json').read_text())
    for relative, expected in fingerprints.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f'Original implementation changed: {relative}')
    prep = json.loads((original / 'preparation.json').read_text())
    if not prep['complete']:
        raise ValueError('Incomplete original training/validation preparation')
    norm = ROOT / 'outputs/normalization/arches_train_2018_2020.json'
    train_provenance = json.loads((ROOT / 'outputs/da_sample_cache' / prep['signatures']['train'] / 'provenance.json').read_text())
    if digest(norm) != train_provenance['normalization']:
        raise ValueError('Training normalization changed')
    cfg = json.loads((original / 'config.json').read_text())
    selected = []
    for seed in (42, 43, 44):
        directory = original if seed == 42 else original / f'diagnostics/replicates/seed{seed}'
        config = directory / 'config.json'
        for kind in ('da', 'neural_correction'):
            path = directory / kind / 'best.pt'
            result = json.loads((directory / kind / 'result.json').read_text())
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            assert_selected_checkpoint(ckpt, result, digest(config), prep['signatures'])
            selected.append({'seed': seed, 'kind': kind, 'checkpoint': str(path),
                             'checkpoint_sha256': digest(path), 'validation_selected_step': ckpt['step'],
                             'config': str(config), 'config_sha256': digest(config)})
            del ckpt
    manifest = ROOT / 'outputs/bg_t2m_test_2022/manifest.json'
    background = json.loads(manifest.read_text())
    if background['checkpoint_sha256'] != cfg['forecast_checkpoint_sha256']:
        raise ValueError('Test backgrounds use a different forecast checkpoint')
    protocol = {'year': 2022, 'start_inclusive': '2022-01-01', 'end_exclusive': '2023-01-01',
                'expected_cycles': 1460, 'seed_selection': 'all predeclared seeds 42,43,44; no test-based ranking or selection',
                'checkpoint_selection': 'best full-2021-validation RMSE within each trained seed/model',
                'selected_checkpoints': selected, 'normalization_sha256': digest(norm),
                'test_forecast_manifest_sha256': digest(manifest), 'data_config': load_config(),
                'primary_metric': 'pooled cosine-latitude-weighted grid RMSE in K',
                'secondary_metrics': ['pooled weighted MAE', 'paired 7/14/30-day block intervals per seed'],
                'no_training_or_tuning': True, 'evaluation_code_sha256': digest(__file__),
                'original_implementation': fingerprints}
    path = out / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Frozen test protocol changed; use a different output and document the reason')
    atomic_json(path, protocol)
    return protocol


class FrozenTestDataset(ArchesSingleCycleDADataset):
    def _collect_station_names(self):
        return []

    def __init__(self, protocol):
        norm = ROOT / 'outputs/normalization/arches_train_2018_2020.json'
        manifest = ROOT / 'outputs/bg_t2m_test_2022/manifest.json'
        if digest(norm) != protocol['normalization_sha256'] or digest(manifest) != protocol['test_forecast_manifest_sha256']:
            raise ValueError('Frozen inputs changed')
        super().__init__(manifest, config=protocol['data_config'], start='2022-01-01', end='2023-01-01',
                         normalization_path=norm, station_split_name='default')
        expected = pd.date_range('2022-01-01', '2023-01-01', freq='6h', inclusive='left')
        if not self.index.valid_time.reset_index(drop=True).equals(pd.Series(expected)):
            raise ValueError('Test time coverage is incomplete')
        station = self.cfg['station_split']
        self.context_stations = StationMembership('context', 'default', station['heldout_fraction'], station['seed'])
        self.heldout_stations = StationMembership('heldout', 'default', station['heldout_fraction'], station['seed'])
        self.signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        self.cache_dir = ROOT / 'outputs/da_test_sample_cache' / self.signature
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def sample_path(self, index):
        return self.cache_dir / (self.index.iloc[index]['sample_id'] + '.pt')

    def __getitem__(self, index):
        path = self.sample_path(index)
        if path.is_file():
            sample = torch.load(path, map_location='cpu', weights_only=False)
            if sample.get('cache_signature') != self.signature:
                raise ValueError('Test cache fingerprint mismatch')
            return sample
        sample = super().__getitem__(index)
        LazyArchesDataset.validate_sample(self, sample)
        sample['cache_signature'] = self.signature
        fd, temporary = tempfile.mkstemp(prefix=path.name, suffix='.tmp', dir=self.cache_dir)
        try:
            with os.fdopen(fd, 'wb') as stream:
                torch.save(sample, stream)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return sample


def initialize_worker(protocol):
    global WORKER_DATA
    torch.set_num_threads(1)
    WORKER_DATA = FrozenTestDataset(protocol)


def prepare_one(index):
    sample = WORKER_DATA[index]
    LazyArchesDataset.validate_sample(WORKER_DATA, sample)
    return {'index': index, 'sample_id': sample['sample_id'], 'context': len(sample['obs_values']),
            'heldout': len(sample['heldout_obs_values'])}


def prepare(protocol, out, workers):
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker, initargs=(protocol,)) as pool:
        pending = {pool.submit(prepare_one, i): i for i in range(1460)}
        for future in as_completed(pending):
            try:
                rows.append(future.result())
            except Exception as error:
                errors.append({'index': pending[future], 'error': repr(error)})
            if (len(rows) + len(errors)) % 50 == 0:
                progress = {'stage': 'data_preparation', 'success': len(rows), 'failed': len(errors), 'expected': 1460}
                atomic_json(out / 'progress.json', progress)
                print(progress, flush=True)
    result = {'complete': len(rows) == 1460 and not errors, 'cycles': len(rows), 'errors': errors,
              'protocol_sha256': digest(out / 'protocol.json'), 'samples': sorted(rows, key=lambda x: x['index'])}
    atomic_json(out / 'preparation.json', result)
    if not result['complete']:
        raise RuntimeError('Test data preparation failed; no model evaluation performed')


def evaluate(protocol, out):
    preparation = json.loads((out / 'preparation.json').read_text())
    if not preparation['complete'] or preparation['protocol_sha256'] != digest(out / 'protocol.json'):
        raise ValueError('Test preparation/protocol mismatch')
    dataset = FrozenTestDataset(protocol)
    loader = DataLoader(CachedSamples(dataset), batch_size=1, shuffle=False, num_workers=2,
                        multiprocessing_context='spawn', pin_memory=True, collate_fn=single_cycle_collate)
    if not torch.cuda.is_available():
        raise ValueError('Assign a GPU for frozen-model evaluation')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    methods = ['background']
    models = []
    for selection in protocol['selected_checkpoints']:
        if digest(selection['checkpoint']) != selection['checkpoint_sha256']:
            raise ValueError('Selected weights changed')
        config = json.loads(Path(selection['config']).read_text())
        model = make_model(selection['kind'], config['model']).cuda()
        ckpt = torch.load(selection['checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        models.append(model.eval())
        methods.append(f"{selection['kind']}_seed{selection['seed']}")
    std = dataset.norm_stats.std('background', '2m_temperature')
    squared = np.zeros((1460, len(methods)), dtype=np.float64)
    absolute = np.zeros_like(squared)
    weight_sums = np.zeros(1460, dtype=np.float64)
    times = []
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            x, obs, target = pack(batch, 'cuda:0')
            w = torch.cos(torch.deg2rad(torch.linspace(90, -90, 121, device=x.device))).clamp_min(0)[None, None, :, None].double()
            weight_sums[i] = float(w.expand_as(x).sum())
            predictions = [x] + [model(x, obs)['analysis'] for model in models]
            for j, prediction in enumerate(predictions):
                if not torch.isfinite(prediction).all():
                    raise ValueError('Nonfinite frozen-model test prediction')
                error = (prediction.double() - target.double()) * std
                squared[i, j] = float((error.square() * w).sum())
                absolute[i, j] = float((error.abs() * w).sum())
            times.append(pd.Timestamp(batch['valid_times'][0]).isoformat())
            if (i + 1) % 50 == 0:
                atomic_json(out / 'progress.json', {'stage': 'evaluation', 'completed': i + 1, 'expected': 1460})
                print('TEST', i + 1, 1460, flush=True)
    expected = pd.date_range('2022-01-01', '2023-01-01', freq='6h', inclusive='left')
    if not pd.DatetimeIndex(times).equals(expected):
        raise ValueError('Final test coverage mismatch')
    rmse = np.sqrt(squared.sum(axis=0) / weight_sums.sum())
    mae = absolute.sum(axis=0) / weight_sums.sum()
    per_seed = []
    for seed in (42, 43, 44):
        da = methods.index(f'da_seed{seed}')
        nc = methods.index(f'neural_correction_seed{seed}')
        daily_sse = squared[:, [0, nc, da]].reshape(365, 4, 3).sum(axis=1)
        daily_weights = weight_sums.reshape(365, 4).sum(axis=1)
        per_seed.append({'seed': seed, 'background_rmse_K': float(rmse[0]),
                         'DA_rmse_K': float(rmse[da]), 'neural_correction_rmse_K': float(rmse[nc]),
                         'DA_minus_background_K': float(rmse[da] - rmse[0]),
                         'DA_minus_neural_correction_K': float(rmse[da] - rmse[nc]),
                         'paired_daily_block_bootstrap': circular_block_bootstrap(daily_sse, daily_weights),
                         'cycle_win_fraction_DA_vs_background': float(np.mean(squared[:, da] < squared[:, 0])),
                         'cycle_win_fraction_DA_vs_NC': float(np.mean(squared[:, da] < squared[:, nc]))})
    summary = {'year': 2022, 'cycles': 1460, 'protocol_sha256': digest(out / 'protocol.json'),
               'weight': 'cos(latitude)', 'reference': 'ERA5', 'normalization': '2018–2020 training-background statistics',
               'test_used_for_selection_or_tuning': False, 'per_seed': per_seed,
               'method_rmse_K': dict(zip(methods, map(float, rmse))),
               'method_mae_K': dict(zip(methods, map(float, mae))),
               'caveat': 'Three predeclared seeds and validation-selected checkpoints. Bootstrap intervals are conditional on trained weights; multiple comparisons are not corrected.'}
    for key in ('DA_rmse_K', 'neural_correction_rmse_K'):
        values = [s[key] for s in per_seed]
        summary[key] = {'mean': float(np.mean(values)), 'sample_std': float(np.std(values, ddof=1))}
    np.savez_compressed(out / 'paired_errors.npz', times=np.array(times), methods=np.array(methods),
                        squared_error_sums=squared, absolute_error_sums=absolute, weight_sums=weight_sums)
    atomic_json(out / 'summary.json', summary)
    print('ALL TEST EVALUATION COMPLETE', json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['freeze', 'prepare', 'evaluate', 'all'], default='all')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/single_da_03_t2m_test_2022')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = freeze_protocol(args.output)
    status = {'stage': 'protocol_frozen', 'protocol_sha256': digest(args.output / 'protocol.json')}
    atomic_json(args.output / 'status.json', status)
    try:
        if args.phase in ('prepare', 'all'):
            status['stage'] = 'preparing_test_data'
            atomic_json(args.output / 'status.json', status)
            prepare(protocol, args.output, args.workers)
        if args.phase in ('evaluate', 'all'):
            status['stage'] = 'evaluating_frozen_models'
            atomic_json(args.output / 'status.json', status)
            evaluate(protocol, args.output)
        status['stage'] = 'complete' if args.phase in ('evaluate', 'all') else args.phase + '_complete'
        atomic_json(args.output / 'status.json', status)
    except Exception as error:
        status['failed'] = repr(error)
        atomic_json(args.output / 'status.json', status)
        raise


if __name__ == '__main__':
    main()

