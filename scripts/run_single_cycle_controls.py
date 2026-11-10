"""Resume-safe, GPU-aware supplemental validation experiment queue."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
from offgrid_da.data.arches_training import atomic_json


def main():
    out = ROOT/'outputs/single_da_04_t2m_controls_ablation_oi'
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source = ROOT/'outputs/single_da_05_input82_ablation'
    cfg = json.loads((source/'config.json').read_text())
    cfg['seeds'] = [42,43]  # User requested two seeds for supplemental experiments.
    files = ['scripts/single_cycle_controls.py','scripts/train_full_state_correction.py',
             'scripts/evaluate_full_state_controls.py','scripts/run_single_cycle_controls.py']
    frozen = json.loads((source/'implementation.json').read_text())
    frozen.update({f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    impl = out/'implementation.json'
    if impl.exists() and json.loads(impl.read_text()) != frozen:
        raise ValueError('Supplemental experiment implementation changed')
    atomic_json(impl,frozen)
    for f in frozen:
        destination = out/'source_snapshot'/f
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():
            destination.write_bytes((ROOT/f).read_bytes())
    atomic_json(out/'protocol.json',{'seeds':[42,43],'modes':['t2m_masked','full82'],
        'split':'2021 validation','test_2022_used':False,'gpus':[0,3],
        'idle_rule':'User authorized sharing GPU0/3; memory <4096 MiB and utilization <90%; launch immediately',
        'correction':'Same initialized convolution branch; full template retains unused attention parameters. No observation dependence.',
        'training_recipe':cfg,'oi':'Fixed exponential covariance: sigma_b=1 K, sigma_o=2.5 K, length=500 km, radius=1000 km, K=16',
        'shuffle':'Within-cycle valid values only; locations, masks and metadata retained; three deterministic permutations'})
    jobs = []
    for mode in cfg['input_modes']:
        for seed in cfg['seeds']:
            directory = out/f'seed{seed}'/mode
            directory.mkdir(parents=True,exist_ok=True)
            config = dict(cfg,seeds=[42,43,44],seed=seed,name=f'controls_v2_{mode}_{seed}',input_mode=mode,experiment_root=str(source))
            config_path = directory/'config.json'
            if config_path.exists() and json.loads(config_path.read_text()) != config:
                raise ValueError('Configuration changed')
            atomic_json(config_path,config)
            for trial in ([True,False] if seed == 42 else [False]):
                target = directory/'trial' if trial else directory/'formal'
                prerequisite = directory/'trial/result.json' if not trial and seed == 42 else out/'seed42'/mode/'trial/result.json' if seed !=42 else None
                command = [sys.executable,'-u',str(ROOT/'scripts/train_full_state_correction.py'),
                    '--config',str(config_path),'--preparation',str(source/f'preparation_{mode}.json'),
                    '--output',str(target),'--kind','neural_correction'] + (['--trial'] if trial else [])
                jobs.append((f'NC_{mode}_{seed}_'+('trial' if trial else 'formal'),target,command,prerequisite,True))
    jobs.append(('local_oi',out/'local_oi',[sys.executable,'-u',str(ROOT/'scripts/evaluate_full_state_controls.py'),
        '--output',str(out/'local_oi'),'--oi-only'],None,False))
    for seed in cfg['seeds']:
        for mode in cfg['input_modes']:
            target = out/f'ablation_seed{seed}'/mode
            jobs.append((f'ablation_{seed}_{mode}',target,[sys.executable,'-u',str(ROOT/'scripts/evaluate_full_state_controls.py'),
                '--seed',str(seed),'--mode',mode,'--output',str(target)],source/f'seed{seed}'/mode/'result.json',False))
    active, stable = {}, {0:0,3:0}
    while True:
        for gpu,(job,process,log) in list(active.items()):
            code = process.poll()
            if code is not None:
                log.close()
                if code or not (job[1]/'result.json').is_file():
                    atomic_json(out/'failure.json',{'job':job[0],'returncode':code})
                    raise RuntimeError(f'Failed {job[0]}; inspect its run.log')
                result = json.loads((job[1]/'result.json').read_text())
                if job[4] and not result['checks']['checkpoint_reload_exact']:
                    raise RuntimeError('Correction checkpoint reload check failed')
                del active[gpu]
                stable[gpu] = 0
        pending = [j for j in jobs if not (j[1]/'result.json').is_file() and j[0] not in [v[0][0] for v in active.values()]]
        if not pending and not active:
            atomic_json(out/'summary.json',{'complete':True,'results':{j[0]:json.loads((j[1]/'result.json').read_text()) for j in jobs},'test_2022_used':False})
            atomic_json(out/'status.json',{'stage':'complete','finished_unix':time.time()})
            return
        lines = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().splitlines()
        info = {int(x[0]):(int(x[1]),int(x[2])) for x in [l.split(',') for l in lines]}
        for gpu in stable:
            stable[gpu] = stable[gpu]+1 if gpu not in active and info[gpu][0]<4096 and info[gpu][1]<90 else 0
            if stable[gpu]<1:
                continue
            ready = next((j for j in pending if j[3] is None or j[3].is_file()),None)
            if ready is None:
                continue
            for f,expected in frozen.items():
                if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=expected:
                    raise ValueError(f'Frozen source changed: {f}')
            ready[1].mkdir(parents=True,exist_ok=True)
            pidfile = ready[1]/'process.json'
            if pidfile.exists():
                previous = json.loads(pidfile.read_text())
                try:
                    os.kill(previous['pid'],0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError('Previous child still alive; refuse duplicate launch')
            command = list(ready[2])
            if ready[4] and (ready[1]/'last.pt').exists():
                command += ['--resume',str(ready[1]/'last.pt')]
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1')
            log = (ready[1]/'run.log').open('ab',buffering=0)
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[gpu] = (ready,process,log)
            atomic_json(pidfile,{'pid':process.pid,'gpu':gpu,'job':ready[0]})
            pending.remove(ready)
            print('START',ready[0],'GPU',gpu,'PID',process.pid,flush=True)
        atomic_json(out/'status.json',{'stage':'running_or_waiting','active':{str(g):{'job':j[0],'pid':p.pid} for g,(j,p,_) in active.items()},
            'pending':[j[0] for j in pending],'idle_checks':stable,'updated_unix':time.time()})
        time.sleep(30)

if __name__ == '__main__':
    main()
