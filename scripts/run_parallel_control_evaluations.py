"""Run outstanding validation evaluations alongside training, without duplicate dispatch."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from offgrid_da.data.arches_training import atomic_json

def main():
 out=ROOT/'outputs/single_da_04_t2m_controls_ablation_oi'
 lock=(out/'parallel_evaluation.lock').open('a')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 pid=int((out/'queue.pid').read_text())
 if b'run_single_cycle_controls.py' not in Path(f'/proc/{pid}/cmdline').read_bytes():
  raise RuntimeError('Incorrect dispatcher PID')
 fingerprints=json.loads((out/'implementation.json').read_text())
 for name,expected in fingerprints.items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
   raise RuntimeError(f'Frozen implementation changed: {name}')
 status={'stage':'running','dispatcher_pid':pid,'jobs':{},'started_unix':time.time()}
 atomic_json(out/'parallel_evaluation_status.json',status)
 os.kill(pid,signal.SIGSTOP)
 try:
  jobs=[('local_oi',out/'local_oi',['--oi-only'])]
  for seed in [42,43]:
   for mode in ['t2m_masked','full82']:
    jobs.append((f'ablation_{seed}_{mode}',out/f'ablation_seed{seed}'/mode,['--seed',str(seed),'--mode',mode]))
  groups={0:jobs[::2],3:jobs[1::2]}
  def worker(gpu,items):
   for name,target,args in items:
    if (target/'result.json').exists():
     continue
    target.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u',str(ROOT/'scripts/evaluate_full_state_controls.py'),'--output',str(target)]+args
    with (target/'parallel_run.log').open('ab',buffering=0) as log:
     process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
     atomic_json(target/'parallel_process.json',{'pid':process.pid,'gpu':gpu,'job':name})
     print('START',name,'GPU',gpu,'PID',process.pid,flush=True)
     code=process.wait()
    if code: raise RuntimeError(f'{name} failed with {code}')
    result=json.loads((target/'result.json').read_text())
    if result['cycles']!=1460: raise RuntimeError('Incomplete validation coverage')
    print('COMPLETE',name,flush=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   futures=[pool.submit(worker,gpu,items) for gpu,items in groups.items()]
   for f in futures: f.result()
  status['stage']='complete'
 except Exception as error:
  status['stage']='failed'; status['error']=repr(error)
  raise
 finally:
  status['finished_unix']=time.time()
  atomic_json(out/'parallel_evaluation_status.json',status)
  os.kill(pid,signal.SIGCONT)

if __name__=='__main__': main()
