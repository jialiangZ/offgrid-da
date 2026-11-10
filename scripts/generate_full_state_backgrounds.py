"""Generate canonical physical full states for the predeclared train/validation input ablation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.main_hydra import collate_fn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from offgrid_da.data.forecast_contract import archesweather_t2m_background
from offgrid_da.data.full_state_data import CHANNELS, SURFACE, LEVEL, PRESSURE, digest


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=0, help='Diagnostic only; no complete status for a limited run')
    args = parser.parse_args()
    experiment = ROOT / 'experiments/archesweather-reproduction'
    geo = experiment / 'geoarches'
    freeze = json.loads((ROOT / 'outputs/forecast_freeze/archesweather_250k_20261031/manifest.json').read_text())
    ckpt, config, stats_path = [Path(freeze[k][p]) for k, p in [('checkpoint', 'path'), ('config', 'source'), ('normalization', 'source')]]
    for path, key in [(ckpt, 'checkpoint'), (config, 'config'), (stats_path, 'normalization')]:
        if digest(path) != freeze[key]['sha256']:
            raise ValueError('Frozen forecast file changed')
    head = subprocess.check_output(['git', '-C', str(geo), 'rev-parse', 'HEAD']).decode().strip()
    patch = hashlib.sha256(subprocess.check_output(['git', '-C', str(geo), 'diff', '--binary', 'HEAD'])).hexdigest()
    if head != freeze['geoarches']['head'] or patch != freeze['geoarches']['worktree_patch_sha256']:
        raise ValueError('Frozen forecast implementation changed')
    header = {'schema': 'physical_full_state_v1', 'checkpoint_sha256': freeze['checkpoint']['sha256'],
              'config_sha256': freeze['config']['sha256'], 'forecast_statistics_sha256': freeze['normalization']['sha256'],
              'forecast_head': head, 'forecast_worktree_sha256': patch, 'generator_sha256': digest(__file__),
              'channel_names': CHANNELS, 'surface_variables': SURFACE, 'level_variables': LEVEL,
              'pressure_hpa': PRESSURE, 'surface_units': ['m/s', 'm/s', 'K', 'Pa'],
              'level_units': ['m2/s2', 'm/s', 'm/s', 'K', 'kg/kg', 'Pa/s'],
              'valid_interval': ['2018-01-01T00:00:00Z', '2022-01-01T00:00:00Z'],
              'lead_hours': 6, 'init_history': 'ERA5 t-6h,t -> forecast t+6h',
              'latitude': '90 to -90', 'longitude': '0 to 358.5', 'precision': 'float32; TF32 disabled',
              'expected_cycles': 5844, 'test_2022_used': False}
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'generation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output / 'manifest.json'
    if path.exists():
        metadata = json.loads(path.read_text())
        if {key: metadata[key] for key in header} != header:
            raise ValueError('Cannot mix generation implementations or schemas')
    else:
        metadata = dict(header, samples=[], complete=False)
    done = {s['valid_utc']: s for s in metadata['samples']}
    if len(done) != len(metadata['samples']):
        raise ValueError('Duplicate completed times')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(args.device)
    cfg = OmegaConf.load(config)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.module.module, cfg.module)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert state['global_step'] == 250000
    model.load_state_dict(state['state_dict'], strict=True)
    del state
    model.eval().to(args.device)
    stats = torch.load(stats_path, map_location='cpu', weights_only=True)
    data = Era5Forecast(path=str(experiment / 'data/era5_240/full'), domain='all', multistep=0,
                        lead_time_hours=6, load_prev=True, norm_scheme='train_stats', norm_stats_path=str(stats_path))
    mapping = {data.id2pt[i + 1][2]: i for i in range(len(data))}
    times = np.arange(np.datetime64('2018-01-01T00', 'ns'), np.datetime64('2022-01-01T00', 'ns'), np.timedelta64(6, 'h'))
    if args.limit:
        times = times[:args.limit]
    for valid in times:
        iso = np.datetime_as_string(valid, unit='s') + 'Z'
        if iso in done:
            sample = done[iso]
            if digest(args.output / sample['file']) != sample['sha256']:
                raise ValueError('Completed background changed')
            continue
        init = valid - np.timedelta64(6, 'h')
        i = mapping[init]
        assert data.id2pt[i][2] == init - np.timedelta64(6, 'h')
        batch = collate_fn([data[i]])
        assert int(batch['timestamp'][0]) == int(init.astype('datetime64[s]').astype('int64'))
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.inference_mode():
            prediction = model.forward(batch)
            converted = archesweather_t2m_background(prediction, stats, [np.datetime_as_string(init, unit='s') + 'Z'])
            physical = data.denormalize(prediction)
            surface = physical['surface'].float().roll(-120, -1)
            level = physical['level'].float().roll(-120, -1)
            torch.testing.assert_close(surface[:, 2], converted['background'], rtol=1e-5, atol=1e-4)
            if not torch.isfinite(surface).all() or not torch.isfinite(level).all():
                raise ValueError('Nonfinite full forecast')
        name = 'state_' + np.datetime_as_string(valid, unit='h').replace('-', '') + '.npz'
        temporary = args.output / (name + '.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, surface=surface[0].cpu().numpy(), level=level[0].cpu().numpy(),
                                latitude=np.linspace(90, -90, 121), longitude=np.arange(240) * 1.5,
                                pressure_hpa=np.array(PRESSURE))
        os.replace(temporary, args.output / name)
        sample = {'valid_utc': iso, 'init_utc': np.datetime_as_string(init, unit='s') + 'Z',
                  'previous_utc': np.datetime_as_string(init - np.timedelta64(6, 'h'), unit='s') + 'Z',
                  'file': name, 'sha256': digest(args.output / name)}
        metadata['samples'].append(sample)
        atomic_json(path, metadata)
        print(f'{len(metadata["samples"])}/5844 {iso}', flush=True)
    metadata['samples'].sort(key=lambda x: x['valid_utc'])
    actual = [s['valid_utc'] for s in metadata['samples']]
    expected = [np.datetime_as_string(t, unit='s') + 'Z' for t in np.arange(np.datetime64('2018-01-01T00', 'ns'), np.datetime64('2022-01-01T00', 'ns'), np.timedelta64(6, 'h'))]
    metadata['complete'] = actual == expected
    atomic_json(path, metadata)
    print('GENERATION', 'COMPLETE' if metadata['complete'] else 'DIAGNOSTIC PARTIAL', len(actual), flush=True)


if __name__ == '__main__':
    main()

