"""Frozen, restartable train-only scaling -> diagnosis -> matched trials/training."""
import concurrent.futures,fcntl,hashlib,json,os,subprocess,sys,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from offgrid_da.data.arches_training import atomic_json
OUT=ROOT/'outputs/single_da_09_full82_loss_balanced'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources=['scripts/'+s+'.py' for s in ['error_balanced','prepare_error_balanced','train_error_balanced',
        'diagnose_frozen_full_update','verify_error_balanced','error_balanced_trainer','run_error_balanced_pipeline','run_full_update_v1','train_single_cycle']]
    sources+=['src/offgrid_da/data/'+s+'.py' for s in ['metar_multivar','multivar_dataset','full_state_data']]
    sources+=['src/offgrid_da/models/'+s+'.py' for s in ['analysis','observation_encoder']]
    fingerprints={s:digest(ROOT/s) for s in sources}
    manifest=OUT/'implementation.json'
    if manifest.exists() and json.loads(manifest.read_text())!=fingerprints:raise RuntimeError('Frozen source changed')
    atomic_json(manifest,fingerprints)
    for name in sources:
        p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.write_bytes((ROOT/name).read_bytes())
    state={'stage':'configured','completed':[],'children':{},'test_2022_used':False}
    mutex=threading.RLock()
    def status(stage):
        with mutex:
            state['stage']=stage;atomic_json(OUT/'status.json',state)
    def check():
        for name,h in fingerprints.items():
            if digest(ROOT/name)!=h:raise RuntimeError(f'Frozen source changed: {name}')
    def run(name,cmd,gpu=None):
        check();env=dict(os.environ,PYTHONUNBUFFERED='1')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with (OUT/f'{name}.log').open('ab',buffering=0) as log:
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            with mutex:
                state['children'][name]={'pid':process.pid,'gpu':gpu};atomic_json(OUT/'status.json',state)
            code=process.wait()
        if code:raise RuntimeError(f'{name} failed ({code}); inspect {name}.log')
        with mutex:
            state['completed'].append(name);state['children'].pop(name,None);atomic_json(OUT/'status.json',state)
    def available(count):
        streak={}
        while True:
            lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().splitlines()
            for line in lines:
                gpu,mem,util=[int(x) for x in line.split(',')]
                streak[gpu]=streak.get(gpu,0)+1 if mem<1536 and util<10 else 0
            good=sorted(g for g,n in streak.items() if n>=3)
            if len(good)>=count:return good[:count]
            state['idle_checks']=streak;atomic_json(OUT/'status.json',state);time.sleep(30)
    try:
        status('training_error_statistics')
        if not (OUT/'loss_statistics.json').exists():run('statistics',[sys.executable,'-u',str(ROOT/'scripts/prepare_error_balanced.py')])
        stats=json.loads((OUT/'loss_statistics.json').read_text());assert stats['complete'] and stats['cycles']==4384
        if not (OUT/'preparation.json').exists():raise RuntimeError('Incomplete statistics preparation')
        status('waiting_for_acceptance_gpu')
        if not (OUT/'acceptance.json').exists():
            gpu=available(1)[0];run('acceptance',[sys.executable,'-u',str(ROOT/'scripts/verify_error_balanced.py')],gpu)
        assert json.loads((OUT/'acceptance.json').read_text())['passed']
        status('frozen_model_diagnostics')
        pending=[s for s in [42,43] if not (OUT/f'frozen_diagnostics/seed{s}/result.json').exists()]
        if pending:
            gpus=available(len(pending))
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                fs=[pool.submit(run,f'diagnostics_{s}',[sys.executable,'-u',str(ROOT/'scripts/diagnose_frozen_full_update.py'),'--seed',str(s)],g) for s,g in zip(pending,gpus)]
                for f in fs:f.result()
        for seed in [42,43]:assert json.loads((OUT/f'frozen_diagnostics/seed{seed}/result.json').read_text())['cycles']==1460
        def paired(seed,gpu,trial):
            processes=[]
            for kind in ['da','neural_correction']:
                target=OUT/f'seed{seed}'/kind/('trial' if trial else 'formal');target.mkdir(parents=True,exist_ok=True)
                if (target/'result.json').exists():continue
                cfg=json.loads((ROOT/f'outputs/single_da_08_full82_large/seed{seed}/{kind}/formal/config.json').read_text())
                cfg.update(name=f'error_balanced_{seed}_{kind}',loss_statistics_sha256=digest(OUT/'loss_statistics.json'),
                    metric='dimensionless training-error-balanced RMSE',trial_steps=256,trial_validation_cycles=64,loader_workers=2)
                config=target/'config.json'
                if config.exists() and json.loads(config.read_text())!=cfg:raise RuntimeError('Configuration changed')
                atomic_json(config,cfg)
                cmd=[sys.executable,'-u',str(ROOT/'scripts/train_error_balanced.py'),'--config',str(config),
                    '--preparation',str(OUT/'preparation.json'),'--output',str(target),'--kind',kind]+(['--trial'] if trial else [])
                if (target/'last.pt').exists():cmd+=['--resume',str(target/'last.pt')]
                processes.append((f'{seed}_{kind}_'+('trial' if trial else 'formal'),cmd))
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                fs=[pool.submit(run,name,cmd,gpu) for name,cmd in processes]
                for f in fs:f.result()
            results=[json.loads((OUT/f'seed{seed}'/k/('trial' if trial else 'formal')/'result.json').read_text()) for k in ['da','neural_correction']]
            assert results[0]['convolution_sha256']==results[1]['convolution_sha256']
            for result in results:assert result['checks']['checkpoint_reload_exact']
            assert results[0]['checks']['empty_exact_identity']
            if not trial:
                assert all(r['selected_validation']['cycles']==1460 for r in results)
        for trial in [True,False]:
            status('waiting_for_trial_gpus' if trial else 'waiting_for_formal_gpus')
            gpus=available(2);status('trials' if trial else 'formal')
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                fs=[pool.submit(paired,s,g,trial) for s,g in zip([42,43],gpus)]
                for f in fs:f.result()
        atomic_json(OUT/'summary.json',{'complete':True,'test_2022_used':False,'results':{
            f'{s}_{k}':json.loads((OUT/f'seed{s}'/k/'formal/result.json').read_text()) for s in [42,43] for k in ['da','neural_correction']}})
        status('complete')
    except Exception as error:
        state['error']=repr(error);status('failed');raise
if __name__=='__main__':main()
