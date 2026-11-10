"""Frozen train/validation pipeline for the matched T2M-only/full-state input experiment."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from offgrid_da.data.arches_training import atomic_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/da/full_state_input_v2.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/single_da_05_input82_ablation')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen_path = args.output / 'config.json'
    if frozen_path.exists() and frozen_path.read_bytes() != args.config.read_bytes():
        raise ValueError('Frozen v2 configuration changed')
    if not frozen_path.exists():
        frozen_path.write_bytes(args.config.read_bytes())
    cfg = json.loads(frozen_path.read_text())
    original = json.loads((ROOT / 'outputs/single_da_02_t2m_dim32_main/implementation.json').read_text())
    added = ['src/offgrid_da/models/full_state_t2m.py', 'src/offgrid_da/data/full_state_data.py',
             'scripts/generate_full_state_backgrounds.py', 'scripts/prepare_full_state_input.py',
             'scripts/train_full_state_input.py', 'scripts/run_full_state_input_v2.py']
    fingerprints = {**original, **{relative: digest(ROOT / relative) for relative in added}}
    implementation = args.output / 'implementation.json'
    if implementation.exists() and json.loads(implementation.read_text()) != fingerprints:
        raise ValueError('Frozen v2 implementation changed')
    atomic_json(implementation, fingerprints)
    for relative, expected in original.items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Original experiment was modified')
    for relative in fingerprints:
        dest = args.output / 'source_snapshot' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes((ROOT / relative).read_bytes())
    state = {'stage': 'configured', 'stages': {}, 'started_unix': time.time(),
             'test_2022_used': False, 'test_2022_already_consumed_by_v1': True}

    def check_code():
        for relative, expected in fingerprints.items():
            if digest(ROOT / relative) != expected:
                raise ValueError(f'Implementation changed: {relative}')

    def run(stage, command, gpu=None):
        check_code()
        state['stage'] = stage
        state['stages'][stage] = 'running'
        atomic_json(args.output / 'status.json', state)
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['PYTHONPATH'] = str(ROOT / 'experiments/archesweather-reproduction/geoarches') + ':' + str(ROOT / 'src')
        print('START', stage, flush=True)
        with (args.output / f'{stage}.log').open('ab', buffering=0) as log:
            subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        state['stages'][stage] = 'complete'
        atomic_json(args.output / 'status.json', state)

    def wait_for_gpus():
        while True:
            lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                             '--format=csv,noheader,nounits']).decode().splitlines()
            info = {int(p[0]): (int(p[1]), int(p[2])) for p in [line.split(',') for line in lines]}
            if all(info[gpu][0] < 1024 and info[gpu][1] < 10 for gpu in cfg['arm_gpus'].values()):
                return
            state['stage'] = 'waiting_for_gpus'
            atomic_json(args.output / 'status.json', state)
            time.sleep(30)

    try:
        geo_python = ROOT / 'experiments/archesweather-reproduction/geoarches/.venv/bin/python'
        run('generation', [str(geo_python), '-u', str(ROOT / 'scripts/generate_full_state_backgrounds.py'),
                           '--output', str(args.output / 'backgrounds')], cfg['generation_gpu'])
        run('preparation', [sys.executable, '-u', str(ROOT / 'scripts/prepare_full_state_input.py'),
                            '--experiment', str(args.output), '--workers', str(cfg['prepare_workers'])])
        configs = {}
        for seed in cfg['seeds']:
            for mode in cfg['input_modes']:
                directory = args.output / f'seed{seed}' / mode
                directory.mkdir(parents=True, exist_ok=True)
                config = dict(cfg, seed=seed, name=f'full_state_v2_seed{seed}_{mode}',
                              input_mode=mode, experiment_root=str(args.output))
                path = directory / 'config.json'
                if path.exists() and json.loads(path.read_text()) != config:
                    raise ValueError('Run config changed')
                atomic_json(path, config)
                configs[seed, mode] = path
        for mode in cfg['input_modes']:
            directory = args.output / 'trial' / mode
            command = [sys.executable, '-u', str(ROOT / 'scripts/train_full_state_input.py'),
                       '--config', str(configs[cfg['seeds'][0], mode]),
                       '--preparation', str(args.output / f'preparation_{mode}.json'),
                       '--output', str(directory), '--kind', 'da', '--trial']
            if not (directory / 'result.json').exists():
                if (directory / 'last.pt').exists():
                    command += ['--resume', str(directory / 'last.pt')]
                run(f'trial_{mode}', command, cfg['generation_gpu'])
            result = json.loads((directory / 'result.json').read_text())
            if not result['checks']['empty_exact_identity'] or not result['checks']['checkpoint_reload_exact']:
                raise ValueError('Input-arm trial failed')
        comparisons = []
        for seed in cfg['seeds']:
            wait_for_gpus()
            processes = []
            for mode in cfg['input_modes']:
                directory = args.output / f'seed{seed}' / mode
                if (directory / 'result.json').exists():
                    continue
                check_code()
                command = [sys.executable, '-u', str(ROOT / 'scripts/train_full_state_input.py'),
                           '--config', str(configs[seed, mode]),
                           '--preparation', str(args.output / f'preparation_{mode}.json'),
                           '--output', str(directory), '--kind', 'da']
                if (directory / 'last.pt').exists():
                    command += ['--resume', str(directory / 'last.pt')]
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(cfg['arm_gpus'][mode])
                env['PYTHONUNBUFFERED'] = '1'
                log = (directory / 'training.log').open('ab', buffering=0)
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                processes.append((mode, process, log))
            state['stage'] = f'formal_seed{seed}'
            state['children'] = {mode: process.pid for mode, process, _ in processes}
            atomic_json(args.output / 'status.json', state)
            failed = []
            for mode, process, log in processes:
                code = process.wait()
                log.close()
                if code:
                    failed.append((mode, code))
            if failed:
                raise RuntimeError(f'Formal input arms failed: {failed}')
            results = {mode: json.loads((args.output / f'seed{seed}' / mode / 'result.json').read_text())
                       for mode in cfg['input_modes']}
            masked, full = [results[mode] for mode in cfg['input_modes']]
            if masked['initial_model_sha256'] != full['initial_model_sha256'] or masked['parameters'] != full['parameters']:
                raise ValueError('Input arms did not have identical architecture and initialization')
            if abs(masked['selected_validation']['baseline0_rmse_K'] - full['selected_validation']['baseline0_rmse_K']) > 1e-10:
                raise ValueError('Input arms used different T2M backgrounds')
            if any(r['selected_validation']['cycles'] != 1460 for r in results.values()):
                raise ValueError('Incomplete validation coverage')
            comparisons.append({'seed': seed, 'parameters': full['parameters'],
                                'baseline0_rmse_K': full['selected_validation']['baseline0_rmse_K'],
                                't2m_only_rmse_K': masked['selected_validation']['rmse_K'],
                                'full82_rmse_K': full['selected_validation']['rmse_K'],
                                'full_minus_t2m_K': full['selected_validation']['rmse_K'] - masked['selected_validation']['rmse_K']})
            atomic_json(args.output / 'partial_comparison.json', comparisons)
            state['stages'][f'formal_seed{seed}'] = 'complete'
        import numpy as np
        result = {'per_seed': comparisons, 'test_2022_used': False,
                  'note': 'Matched architecture and initialization; T2M-only masks 81 of 82 slots. Validation comparison, not a new independent test.',
                  't2m_only_mean_rmse_K': float(np.mean([r['t2m_only_rmse_K'] for r in comparisons])),
                  'full82_mean_rmse_K': float(np.mean([r['full82_rmse_K'] for r in comparisons]))}
        atomic_json(args.output / 'comparison.json', result)
        state['stage'] = 'complete'
        state['finished_unix'] = time.time()
        atomic_json(args.output / 'status.json', state)
        print('ALL COMPLETE', json.dumps(result), flush=True)
    except Exception as error:
        state['failed'] = repr(error)
        atomic_json(args.output / 'status.json', state)
        raise


if __name__ == '__main__':
    main()

