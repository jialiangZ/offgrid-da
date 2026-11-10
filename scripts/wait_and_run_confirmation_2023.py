"""Frozen confirmation queue. Wait for predecessor, data, and GPUs; never fallback."""
import os,sys,json,time,hashlib,subprocess,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from offgrid_da.data.arches_training import atomic_json

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=ROOT/'outputs/cycling_t2m_06_confirm_2023_pending';out.mkdir(exist_ok=True)
 lock=(out/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 frozen=out/'confirmation_protocol.json'
 if not frozen.exists():
  raise RuntimeError('Freeze confirmation protocol before scheduling')
 protocol=json.loads(frozen.read_text())
 def check():
  for relative,expected in protocol['frozen_files'].items():
   if digest(ROOT/relative)!=expected:raise RuntimeError('Frozen file changed: '+relative)
 def status(stage,**details):atomic_json(out/'status.json',dict(stage=stage,updated_unix=time.time(),**details))
 check()
 while True:
  previous=ROOT/'outputs/cycling_t2m_05_15day/status.json'
  state=json.loads(previous.read_text()) if previous.exists() else {}
  if 'failed' in state.get('stage',''):
   status('blocked_predecessor_failed',predecessor=state);return
  if state.get('stage')=='complete':break
  status('waiting_for_predecessor',predecessor=state);time.sleep(60)
 while True:
  check()
  # Metadata-only coverage check; do not inspect reference values or scores.
  import xarray as xr
  import numpy as np
  directory=ROOT/'experiments/archesweather-reproduction/data/era5_240/full'
  files=sorted(directory.glob('*2023*.nc'))
  times=set()
  for file in files:
   with xr.open_dataset(file) as ds:times.update(np.asarray(ds.time.values).astype('datetime64[s]').astype('int64').tolist())
  expected=set()
  for window in protocol['windows']:
   begin=np.datetime64(window,'s')-np.timedelta64(6,'h')
   expected.update(np.arange(begin,np.datetime64(window,'s')+np.timedelta64(15*24+6,'h'),np.timedelta64(6,'h')).astype('int64').tolist())
  missing=expected-times
  if missing:
   status('waiting_for_2023_data',missing_timestamps=len(missing),required_directory=str(directory),note='No fallback to 2021/2022; full-state ERA5 2023 netCDF shards required. METAR is read from existing public reader.')
   time.sleep(60);continue
  lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().splitlines()
  info={int(a[0]):(int(a[1]),int(a[2])) for a in [l.split(',') for l in lines]}
  if not all(info[g][0]<4096 and info[g][1]<90 for g in [0,3]):
   status('waiting_for_gpus',gpu_state=info);time.sleep(60);continue
  break
 check()
 children=[]
 for seed,gpu in [(42,0),(43,3)]:
  if (out/f'formal/seed{seed}/result.json').exists():continue
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='src:scripts:experiments/archesweather-reproduction/geoarches')
  log=(out/f'seed{seed}.log').open('ab',buffering=0)
  child=subprocess.Popen([str(ROOT/'experiments/archesweather-reproduction/geoarches/.venv/bin/python'),'-u',str(ROOT/'scripts/run_cycling_confirmation_2023.py'),'--seed',str(seed),'--steps','60'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  children.append((seed,child,log))
 status('running_confirmation',children={str(s):p.pid for s,p,l in children})
 failed=[]
 for seed,child,log in children:
  code=child.wait();log.close()
  if code:failed.append([seed,code])
 if failed:status('confirmation_failed',errors=failed);return
 check();status('aggregating')
 try:subprocess.run([sys.executable,str(ROOT/'scripts/summarize_cycling_confirmation_2023.py')],cwd=ROOT,check=True)
 except Exception as e:status('aggregation_failed',error=repr(e));raise
if __name__=='__main__':main()
