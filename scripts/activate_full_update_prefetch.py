"""Wait for both periodic recovery checkpoints before switching loader processes."""
from pathlib import Path
import json,os,signal,subprocess,time,sys,fcntl
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from offgrid_da.data.arches_training import atomic_json

def main():
 out=ROOT/'outputs/single_da_08_full82_large'
 lock=(out/'prefetch_switch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  state=json.loads((out/'status.json').read_text())
  if state.get('stage')!='formal' or state.get('seed')!=42:raise RuntimeError('Unexpected stage; refusing interruption')
  recovered={}
  for kind in ['da','neural_correction']:
   p=out/'seed42'/kind/'formal'/'last.pt'
   if p.exists():
    try:
     ck=torch.load(p,map_location='cpu',weights_only=False)
     if ck['step']>=1000:recovered[kind]=ck['step']
    except Exception:pass
  atomic_json(out/'prefetch_switch.json',{'stage':'waiting_for_periodic_checkpoints','saved_steps':recovered})
  if len(recovered)==2:break
  time.sleep(30)
 parent=int((out/'pipeline.pid').read_text())
 if b'run_full_update_large_v1.py' not in Path(f'/proc/{parent}/cmdline').read_bytes():raise RuntimeError('Wrong parent PID')
 os.kill(parent,signal.SIGSTOP)
 for child in state['pids']:
  try:
   cmd=Path(f'/proc/{child}/cmdline').read_bytes()
   if b'run_full_update_large_v1.py' not in cmd:raise RuntimeError('Wrong child PID')
   os.kill(child,signal.SIGTERM)
  except FileNotFoundError:pass
 os.kill(parent,signal.SIGTERM);os.kill(parent,signal.SIGCONT)
 with (out/'pipeline.log').open('ab',buffering=0) as log:
  p=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/run_full_update_large_v1.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'pipeline.pid').write_text(str(p.pid))
 atomic_json(out/'prefetch_switch.json',{'stage':'activated','saved_steps':recovered,'new_pipeline_pid':p.pid,'workers':2,'prefetch_factor':2})
if __name__=='__main__':main()
