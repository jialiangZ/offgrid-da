"""Sequential paired seed-43/44 replications of the frozen seed-42 recipe."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from offgrid_da.data.arches_training import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    original = ROOT / 'outputs/single_da_02_t2m_dim32_main'
    out = original / 'diagnostics/replicates'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = json.loads((original / 'config.json').read_text())
    fingerprints = json.loads((original / 'implementation.json').read_text())
    prep = original / 'preparation.json'
    if not json.loads(prep.read_text())['complete']:
        raise ValueError('Incomplete data preparation')
    plan = {'seeds': [42, 43, 44], 'new_seeds': [43, 44], 'gpu': 0,
            'validation_year': 2021, 'test_set_used': False, 'same_preparation': str(prep),
            'source_seed': 42, 'configuration_differences': ['name', 'seed'],
            'trial_weights_used': False, 'jobs': []}
    for seed in (43, 44):
        directory = out / f'seed{seed}'
        directory.mkdir(exist_ok=True)
        replica_cfg = dict(cfg, seed=seed, name=f'single_cycle_v1_seed{seed}')
        config_path = directory / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != replica_cfg:
            raise ValueError('Replication configuration changed')
        atomic_json(config_path, replica_cfg)
        for kind in ('da', 'neural_correction'):
            plan['jobs'].append({'seed': seed, 'kind': kind, 'config': str(config_path),
                                 'output': str(directory / kind)})
    atomic_json(out / 'plan.json', plan)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    if not (original / 'diagnostics/validation_analysis.json').is_file():
        raise ValueError('Finish the single-seed ablation/diagnostics before replication')
    status = {'started_unix': time.time(), 'completed_jobs': [], 'test_set_used': False}
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['PYTHONUNBUFFERED'] = '1'
    try:
        for job in plan['jobs']:
            for relative, expected in fingerprints.items():
                if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Original experiment implementation changed: {relative}')
            directory = Path(job['output'])
            directory.mkdir(exist_ok=True)
            status['current_job'] = job
            result_path = directory / 'result.json'
            if not result_path.exists():
                command = [sys.executable, '-u', str(ROOT / 'scripts/train_single_cycle.py'),
                           '--config', job['config'], '--preparation', str(prep),
                           '--output', str(directory), '--kind', job['kind']]
                if (directory / 'last.pt').exists():
                    command += ['--resume', str(directory / 'last.pt')]
                print('START', job['seed'], job['kind'], flush=True)
                with (directory / 'training.log').open('ab', buffering=0) as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT)
                    status['child_pid'] = process.pid
                    atomic_json(out / 'status.json', status)
                    code = process.wait()
                if code:
                    raise RuntimeError(f"Seed {job['seed']} {job['kind']} exited {code}")
            result = json.loads(result_path.read_text())
            expected_config = hashlib.sha256(Path(job['config']).read_bytes()).hexdigest()
            if result['signature']['config_sha256'] != expected_config or result['trial']:
                raise ValueError('Unexpected replication result signature')
            if result['selected_validation']['cycles'] != 1460:
                raise ValueError('Replication validation coverage is incomplete')
            status['completed_jobs'].append({'seed': job['seed'], 'kind': job['kind'],
                                             'best_step': result['best_step'],
                                             'rmse_K': result['selected_validation']['rmse_K']})
            status.pop('child_pid', None)
            atomic_json(out / 'status.json', status)
        values = []
        for seed in (42, 43, 44):
            directory = original if seed == 42 else out / f'seed{seed}'
            da = json.loads((directory / 'da/result.json').read_text())
            nc = json.loads((directory / 'neural_correction/result.json').read_text())
            base = da['selected_validation']['baseline0_rmse_K']
            if abs(base - nc['selected_validation']['baseline0_rmse_K']) > 1e-10:
                raise ValueError('Paired controls use different background samples')
            values.append({'seed': seed, 'background_rmse_K': base,
                           'DA_rmse_K': da['selected_validation']['rmse_K'],
                           'neural_correction_rmse_K': nc['selected_validation']['rmse_K']})
        summary = {'per_seed': values, 'test_set_used': False, 'validation_cycles_per_seed': 1460,
                   'caveat': 'Three seeds; best checkpoints selected on the same validation year. Descriptive seed variation is not independent test evidence.'}
        for key in ('DA_rmse_K', 'neural_correction_rmse_K'):
            numbers = [v[key] for v in values]
            summary[key] = {'mean': float(np.mean(numbers)), 'sample_std': float(np.std(numbers, ddof=1))}
        summary['paired_DA_minus_NC_K'] = [v['DA_rmse_K'] - v['neural_correction_rmse_K'] for v in values]
        summary['paired_DA_minus_background_K'] = [v['DA_rmse_K'] - v['background_rmse_K'] for v in values]
        atomic_json(out / 'summary.json', summary)
        status['current_job'] = 'complete'
        status['finished_unix'] = time.time()
        atomic_json(out / 'status.json', status)
        print('ALL REPLICATES COMPLETE', json.dumps(summary), flush=True)
    except Exception as error:
        status['failed'] = repr(error)
        atomic_json(out / 'status.json', status)
        raise


if __name__ == '__main__':
    main()

