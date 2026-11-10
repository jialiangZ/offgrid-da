"""Paired validation-only ablation, regional/station diagnostics and block bootstrap."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from offgrid_da.data.arches_training import LazyArchesDataset, atomic_json
from offgrid_da.data.collate import single_cycle_collate
from offgrid_da.data.alignment import sample_background_at_obs
from train_single_cycle import CachedSamples, make_model, pack

METHODS = ['background', 'neural_correction', 'DA']


def circular_block_bootstrap(day_sse, day_weights, repetitions=2000, seed=20261101):
    """Resample the same circular consecutive-day blocks for all methods."""
    day_sse = np.asarray(day_sse, dtype=np.float64)
    day_weights = np.asarray(day_weights, dtype=np.float64)
    if day_sse.ndim != 2 or len(day_sse) != len(day_weights) or not (day_weights > 0).all():
        raise ValueError('Invalid daily sufficient statistics')
    n = len(day_weights)
    result = {}
    for length in (7, 14, 30):
        rng = np.random.default_rng(seed + length)
        starts = rng.integers(n, size=(repetitions, math.ceil(n / length)))
        indices = ((starts[:, :, None] + np.arange(length)) % n).reshape(repetitions, -1)[:, :n]
        values = np.sqrt(day_sse[indices].sum(axis=1) / day_weights[indices].sum(axis=1)[:, None])
        result[str(length)] = {}
        for j, name in ((0, 'DA_minus_background'), (1, 'DA_minus_neural_correction')):
            differences = values[:, 2] - values[:, j]
            result[str(length)][name] = {'lower_95_K': float(np.quantile(differences, .025)),
                                        'upper_95_K': float(np.quantile(differences, .975))}
    return result


def regions(latitude, longitude):
    lat, lon = np.meshgrid(latitude, longitude, indexing='ij')
    return {
        'global': np.ones(lat.shape, dtype=bool),
        'tropics': np.abs(lat) <= 30,
        'north_extratropics': lat > 30,
        'south_extratropics': lat < -30,
        'Europe_rectangle': (lat >= 35) & (lat <= 70) & ((lon >= 350) | (lon <= 40)),
        'North_America_rectangle': (lat >= 25) & (lat <= 60) & (lon >= 230) & (lon <= 300),
        'East_Asia_rectangle': (lat >= 20) & (lat <= 55) & (lon >= 100) & (lon <= 145),
    }


def aggregate(sse, weights):
    return np.sqrt(sse.sum(axis=0) / weights.sum(axis=0)[None, :])


def main():
    run = ROOT / 'outputs/single_da_02_t2m_dim32_main'
    out = run / 'diagnostics'
    out.mkdir(exist_ok=True)
    cfg = json.loads((run / 'config.json').read_text())
    frozen = json.loads((run / 'implementation.json').read_text())
    for name, digest in frozen.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen implementation changed: {name}')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda:0'
    if not torch.cuda.is_available():
        raise RuntimeError('Assign a GPU explicitly')
    dataset = LazyArchesDataset(ROOT, 'val')
    cached = CachedSamples(dataset)
    loader = DataLoader(cached, batch_size=1, shuffle=False, num_workers=2,
                        multiprocessing_context='spawn', pin_memory=True, collate_fn=single_cycle_collate)
    mean = dataset.norm_stats.mean('background', '2m_temperature')
    std = dataset.norm_stats.std('background', '2m_temperature')
    models, provenance = {}, {}
    for kind, label in [('da', 'DA'), ('neural_correction', 'neural_correction')]:
        path = run / kind / 'best.pt'
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if ckpt['signature']['data']['val'] != dataset.signature:
            raise ValueError('Validation data fingerprint changed')
        model = make_model(kind, cfg['model']).to(device)
        model.load_state_dict(ckpt['model'])
        models[label] = model.eval()
        provenance[label] = {'checkpoint': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'selected_step': ckpt['step']}
    n = len(dataset)
    fixed = np.linspace(0, n - 1, cfg['trial_validation_cycles'], dtype=int).tolist()
    fixed_set = set(fixed)
    latitude = np.linspace(90, -90, 121)
    longitude = np.arange(240) * 1.5
    region_masks = regions(latitude, longitude)
    names = list(region_masks)
    spatial_weights = np.maximum(np.cos(np.deg2rad(latitude)), 0)[:, None] * np.ones((1, 240))
    region_weights = np.stack([spatial_weights * region_masks[name] for name in names])
    denominators = region_weights.sum(axis=(1, 2))
    sse = np.zeros((n, len(METHODS), len(names)), dtype=np.float64)
    weights = np.repeat(denominators[None, :], n, axis=0)
    station_methods = METHODS + ['ERA5_reference']
    station_sse = np.zeros((n, len(station_methods)), dtype=np.float64)
    station_count = np.zeros(n, dtype=np.int64)
    ablation = []
    times = []
    mse_map = np.zeros((2, 121, 240), dtype=np.float64)
    no_obs_exact = 0
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            x, obs, _ = pack(batch, device)
            reference = batch['reference'][0, 0].numpy().astype(np.float64)
            fields = {'background': batch['background'][0, 0].numpy().astype(np.float64)}
            for label, model in models.items():
                prediction = model(x, obs)['analysis']
                if not torch.isfinite(prediction).all():
                    raise ValueError('Nonfinite analysis')
                fields[label] = (prediction[0, 0].float() * std + mean).cpu().numpy().astype(np.float64)
            errors = np.stack([(fields[name] - reference) ** 2 for name in METHODS])
            sse[i] = np.einsum('mhw,rhw->mr', errors, region_weights)
            mse_map[0] += errors[2] - errors[0]
            mse_map[1] += errors[2] - errors[1]
            timestamp = pd.Timestamp(batch['valid_times'][0])
            if timestamp.year != 2021:
                raise ValueError('Analysis must use validation year only')
            times.append(timestamp.isoformat())
            mask = batch['heldout_obs_mask'][0]
            loc = batch['heldout_obs_locations'][0][mask].numpy()
            values = batch['heldout_obs_values'][0][mask].numpy().astype(np.float64)
            station_count[i] = len(values)
            for j, name in enumerate(station_methods):
                field = reference if name == 'ERA5_reference' else fields[name]
                sampled = sample_background_at_obs(field, latitude, longitude, loc[:, 0], loc[:, 1])
                if not np.isfinite(sampled).all():
                    raise ValueError('Nonfinite held-out station diagnostic')
                station_sse[i, j] = np.square(sampled - values).sum()
            if i in fixed_set:
                empty = {k: v.clone() for k, v in obs.items()}
                empty['observation_mask'].zero_()
                empty_prediction = models['DA'](x, empty)['analysis']
                if not torch.equal(empty_prediction, x):
                    raise ValueError('No-observation identity failed')
                no_obs_exact += 1
                empty_field = (empty_prediction[0, 0].float() * std + mean).cpu().numpy().astype(np.float64)
                global_weight = denominators[0]
                row = {'index': i, 'valid_time': timestamp.isoformat(),
                       'background_sse': float(sse[i, 0, 0]), 'DA_real_sse': float(sse[i, 2, 0]),
                       'neural_correction_sse': float(sse[i, 1, 0]),
                       'DA_masked_sse': float((np.square(empty_field - reference) * spatial_weights).sum()),
                       'weight': float(global_weight), 'DA_shuffled_sse': []}
                for repetition in range(3):
                    changed = {k: v.clone() for k, v in obs.items()}
                    rng = np.random.default_rng(20261101 + i * 10 + repetition)
                    for variable in torch.unique(obs['obs_variables'][obs['observation_mask']]):
                        indices = torch.where(obs['observation_mask'][0] & (obs['obs_variables'][0] == variable))[0]
                        permutation = torch.as_tensor(rng.permutation(len(indices)), device=x.device)
                        changed['obs_values_normalized'][0, indices] = obs['obs_values_normalized'][0, indices[permutation]]
                    shuffled = models['DA'](x, changed)['analysis']
                    if not torch.isfinite(shuffled).all():
                        raise ValueError('Nonfinite shuffled-observation output')
                    field = (shuffled[0, 0].float() * std + mean).cpu().numpy().astype(np.float64)
                    row['DA_shuffled_sse'].append(float((np.square(field - reference) * spatial_weights).sum()))
                ablation.append(row)
            if (i + 1) % 50 == 0:
                progress = {'completed': i + 1, 'total': n, 'ablation_cycles': len(ablation)}
                atomic_json(out / 'analysis_progress.json', progress)
                print(progress, flush=True)
    if n != 1460 or len(ablation) != len(fixed):
        raise ValueError('Incomplete validation analysis')
    pooled = aggregate(sse, weights)
    for j, kind in ((1, 'neural_correction'), (2, 'da')):
        expected = json.loads((run / kind / 'result.json').read_text())['selected_validation']['rmse_K']
        if abs(pooled[j, 0] - expected) > 2e-5:
            raise ValueError('Reevaluation does not reproduce selected checkpoint score')
    months = np.array([pd.Timestamp(t).month for t in times])
    monthly = {}
    for month in range(1, 13):
        selected = months == month
        values = aggregate(sse[selected], weights[selected])[:, 0]
        monthly[str(month)] = {name: float(values[j]) for j, name in enumerate(METHODS)}
    regional = {region: {name: float(pooled[j, r]) for j, name in enumerate(METHODS)}
                for r, region in enumerate(names)}
    total_station_count = int(station_count.sum())
    stations = {name: float(math.sqrt(station_sse[:, j].sum() / total_station_count))
                for j, name in enumerate(station_methods)}
    # Every day is one complete four-cycle block. Resample consecutive days jointly.
    expected_times = pd.date_range('2021-01-01', '2022-01-01', freq='6h', inclusive='left')
    if not pd.DatetimeIndex(times).equals(expected_times):
        raise ValueError('Daily-block statistics require complete sorted six-hour validation times')
    daily_sse = sse[:, :, 0].reshape(365, 4, len(METHODS)).sum(axis=1)
    daily_weights = weights[:, 0].reshape(365, 4).sum(axis=1)
    intervals = circular_block_bootstrap(daily_sse, daily_weights)
    point_differences = {'DA_minus_background_K': float(pooled[2, 0] - pooled[0, 0]),
                         'DA_minus_neural_correction_K': float(pooled[2, 0] - pooled[1, 0])}
    ablation_weight = sum(row['weight'] for row in ablation)
    ablation_scores = {name: math.sqrt(sum(row[key] for row in ablation) / ablation_weight)
                       for name, key in [('background', 'background_sse'), ('DA_real', 'DA_real_sse'),
                                         ('DA_masked', 'DA_masked_sse'), ('neural_correction', 'neural_correction_sse')]}
    ablation_scores['DA_shuffled_repeats'] = [math.sqrt(sum(row['DA_shuffled_sse'][j] for row in ablation) / ablation_weight) for j in range(3)]
    report = {'year': 2021, 'cycles': n, 'test_set_used': False, 'provenance': provenance,
              'cache_signature': dataset.signature, 'global_rmse_K': regional['global'],
              'monthly_rmse_K': monthly, 'regional_rmse_K': regional,
              'region_definitions': {'tropics': '|latitude| <= 30', 'extratropics': 'latitude > 30 or latitude < -30',
                                     'Europe_rectangle': '35..70N, 10W..40E', 'North_America_rectangle': '25..60N, 130W..60W',
                                     'East_Asia_rectangle': '20..55N, 100..145E; geographic rectangles include ocean'},
              'heldout_station_rmse_K': stations, 'heldout_station_reports': total_station_count,
              'station_caveat': 'Point-to-grid, timing and observation differences include about 2.5 K representativeness error; not a pure forecast-error estimate.',
              'ablation': {'indices': fixed, 'cycles': len(fixed), 'rmse_K': ablation_scores,
                           'masked_exact_normalized_identity_cycles': no_obs_exact,
                           'shuffle': 'three fixed-seed within-time/within-variable value permutations; preserve locations and metadata',
                           'caveat': 'Masked identity is enforced by architecture. Shuffling is an information-dependence diagnostic and changes physical value-location consistency.'},
              'paired_point_differences': point_differences, 'paired_daily_block_bootstrap': intervals,
              'bootstrap': {'repetitions': 2000, 'confidence': .95, 'block_lengths_days': [7, 14, 30],
                            'caveat': 'Conditional descriptive intervals on the validation set used to select checkpoints; not an independent confirmatory significance test.'},
              'cycle_win_fraction_DA_vs_background': float(np.mean(sse[:, 2, 0] < sse[:, 0, 0])),
              'cycle_win_fraction_DA_vs_neural_correction': float(np.mean(sse[:, 2, 0] < sse[:, 1, 0]))}
    atomic_json(out / 'validation_analysis.json', report)
    atomic_json(out / 'ablation_per_cycle.json', ablation)
    np.savez_compressed(out / 'paired_errors.npz', times=np.array(times), methods=np.array(METHODS),
                        regions=np.array(names), squared_error_sums=sse, weight_sums=weights,
                        station_squared_error_sums=station_sse, station_counts=station_count,
                        mean_mse_difference_maps=mse_map / n, latitude=latitude, longitude=longitude)
    plot_diagnostics(out, report, mse_map / n)
    print('ANALYSIS COMPLETE', json.dumps({'global': report['global_rmse_K'], 'ablation': ablation_scores,
                                          'station': stations, 'paired': point_differences}), flush=True)


def plot_diagnostics(out, report, maps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    months = np.arange(1, 13)
    for name, color in [('DA', '#2563eb'), ('neural_correction', '#e87920')]:
        diff = [(report['monthly_rmse_K'][str(m)][name] - report['monthly_rmse_K'][str(m)]['background']) * 1000 for m in months]
        ax[0].plot(months, diff, '-o', label=name, color=color)
    ax[0].axhline(0, color='gray', linewidth=1)
    ax[0].set(xlabel='Month in 2021', ylabel='RMSE minus background (mK)', title='Monthly validation improvement; negative is better')
    ax[0].legend()
    labels = list(report['regional_rmse_K'])
    xpos = np.arange(len(labels))
    for shift, name, color in [(-.18, 'DA', '#2563eb'), (.18, 'neural_correction', '#e87920')]:
        values = [(report['regional_rmse_K'][r][name] - report['regional_rmse_K'][r]['background']) * 1000 for r in labels]
        ax[1].bar(xpos + shift, values, width=.35, color=color, label=name)
    ax[1].axhline(0, color='gray', linewidth=1)
    ax[1].set_xticks(xpos, labels=[x.replace('_', '\n') for x in labels], fontsize=8)
    ax[1].set(ylabel='RMSE minus background (mK)', title='Predefined geographic regions')
    fig.savefig(out / 'monthly_regional_diagnostics.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for i, title in enumerate(['DA minus background', 'DA minus Neural Correction']):
        scale = float(np.quantile(np.abs(maps[i]), .99)) or 1e-8
        im = axes[i].imshow(maps[i], extent=(0, 360, -90, 90), origin='upper', aspect='auto',
                            cmap='RdBu_r', vmin=-scale, vmax=scale)
        axes[i].set(title=title + ': annual mean squared-error difference', xlabel='Longitude (degrees E)', ylabel='Latitude')
        fig.colorbar(im, ax=axes[i], label='K squared (blue is better)', shrink=.8)
    fig.savefig(out / 'mse_difference_maps.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()

