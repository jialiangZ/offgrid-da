"""Two DA seeds in parallel; matched NC shares the respective GPU."""
import concurrent.futures,fcntl,json,os,signal,subprocess,sys,time,threading
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from offgrid_da.data.arches_training import atomic_json
OUT=ROOT/'outputs/single_da_08_full82_large'
STATE={'stage':'waiting_for_recovery_checkpoint','seed_gpus':{'42':0,'43':1},'jobs':{}}
MUTEX=threading.Lock()
def update(name,value):
 with MUTEX:
  STATE['jobs'][name]=value;atomic_json(OUT/'parallel_seed_status.json',STATE)

def worker(seed,gpu):
 for trial in [True,False]:
  children=[]
  for kind in ['da','neural_correction']:
   target=OUT/f'seed{seed}'/kind/('trial' if trial else 'formal');target.mkdir(parents=True,exist_ok=True)
   name=f'{seed}_{kind}_'+('trial' if trial else 'formal')
   if (target/'result.json').exists():update(name,{'stage':'complete','gpu':gpu});continue
   cfg=json.loads((OUT/'seed42'/kind/'formal/config.json').read_text())
   cfg.update(seed=seed,name=f'full_update_large_{seed}_{kind}')
   config=target/'config.json'
   if config.exists() and json.loads(config.read_text())!=cfg:raise RuntimeError('Configuration mismatch')
   atomic_json(config,cfg)
   cmd=[sys.executable,'-u',str(ROOT/'scripts/train_full_update_prefetch.py'),'--config',str(config),'--preparation',str(OUT/'preparation.json'),'--output',str(target),'--kind',kind]
   if trial:cmd+=['--trial']
   checkpoint=target/'last.pt'
   if checkpoint.exists():cmd+=['--resume',str(checkpoint)]
   log=(target/'run.log').open('ab',buffering=0)
   process=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   children.append((name,process,log,target))
   update(name,{'stage':'running','pid':process.pid,'gpu':gpu,'resume':str(checkpoint) if checkpoint.exists() else None})
  for name,process,log,target in children:
   code=process.wait();log.close()
   if code:raise RuntimeError(f'{name} failed with {code}')
   result=json.loads((target/'result.json').read_text())
   assert result['checks']['checkpoint_reload_exact']
   if result['kind']=='da':assert result['checks']['empty_exact_identity']
   update(name,{'stage':'complete','gpu':gpu,'steps':result['steps']})

def main():
 lock=(OUT/'parallel_seed.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  saved={}
  for kind in ['da','neural_correction']:
   p=OUT/'seed42'/kind/'formal/last.pt'
   if p.exists():
    try:
     ck=torch.load(p,map_location='cpu',weights_only=False)
     if ck['step']>=1000:saved[kind]=ck['step']
    except Exception:pass
  STATE['saved_steps']=saved;atomic_json(OUT/'parallel_seed_status.json',STATE)
  if len(saved)==2:break
  time.sleep(30)
 previous=json.loads((OUT/'status.json').read_text());parent=int((OUT/'pipeline.pid').read_text())
 if b'run_full_update_large_v1.py' not in Path(f'/proc/{parent}/cmdline').read_bytes():raise RuntimeError('Wrong prior dispatcher')
 os.kill(parent,signal.SIGSTOP)
 for pid in previous['pids']:
  try:
   if b'run_full_update_large_v1.py' not in Path(f'/proc/{pid}/cmdline').read_bytes():raise RuntimeError('Wrong prior child')
   os.kill(pid,signal.SIGTERM)
  except FileNotFoundError:pass
 os.kill(parent,signal.SIGTERM);os.kill(parent,signal.SIGCONT)
 for attempt in range(60):
  lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits']).decode().splitlines()
  info={int(l.split(',')[0]):int(l.split(',')[1]) for l in lines}
  if info[0]<1024 and info[1]<1024:break
  time.sleep(2)
 else:raise RuntimeError('Waiting for GPU release timed out; no new jobs launched')
 STATE['stage']='parallel_seeds';atomic_json(OUT/'status.json',{'stage':'parallel_seeds','details':'parallel_seed_status.json'})
 (OUT/'pipeline.pid').write_text(str(os.getpid()))
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(worker,seed,gpu) for seed,gpu in [(42,0),(43,1)]]
  for future in futures:future.result()
 STATE['stage']='complete';atomic_json(OUT/'parallel_seed_status.json',STATE);atomic_json(OUT/'status.json',{'stage':'complete'})
if __name__=='__main__':
 try:main()
 except Exception as error:
  STATE['stage']='failed';STATE['error']=repr(error);atomic_json(OUT/'parallel_seed_status.json',STATE);raise
