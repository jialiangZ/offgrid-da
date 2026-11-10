"""GPU2/3 workflow: await trained full82 models, freeze, gate, pilot, expanded evaluation."""
import os,sys,json,time,subprocess,hashlib,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from offgrid_da.data.arches_training import atomic_json
out=ROOT/'outputs/cycling_full82_01_small_superseded'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out.mkdir(exist_ok=True);lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def status(stage,**kwargs):atomic_json(out/'status.json',dict(stage=stage,updated_unix=time.time(),**kwargs))
 def gpu_wait(gpus):
  while True:
   lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().splitlines()
   info={int(a[0]):(int(a[1]),int(a[2])) for a in [l.split(',') for l in lines]}
   if all(info[g][0]<1024 and info[g][1]<10 for g in gpus):return
   status('waiting_for_gpus',GPUs=gpus,occupancy=info);time.sleep(30)
 def check_frozen():
  frozen=json.loads((out/'frozen.json').read_text())
  for name,expected in frozen['files'].items():
   if digest(ROOT/name)!=expected:raise RuntimeError('Frozen input changed: '+name)
 def run(stage,command,gpu=None):
  check_frozen();status(stage)
  env=dict(os.environ,PYTHONPATH='src:scripts:experiments/archesweather-reproduction/geoarches')
  if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
  with (out/(stage+'.log')).open('ab',buffering=0) as log:
   subprocess.run(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
 def pair(stage,extra):
  gpu_wait([2,3]);check_frozen();children=[]
  for seed,gpu in [(42,2),(43,3)]:
   if (out/stage/f'seed{seed}/result.json').exists():continue
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='src:scripts:experiments/archesweather-reproduction/geoarches')
   log=(out/f'{stage}_seed{seed}.log').open('ab',buffering=0)
   p=subprocess.Popen([str(ROOT/'experiments/archesweather-reproduction/geoarches/.venv/bin/python'),'-u',str(ROOT/'scripts/run_full_update_cycling.py'),'--seed',str(seed),*extra],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   children.append((seed,p,log))
  status(stage,children={str(s):p.pid for s,p,l in children});failed=[]
  for seed,p,log in children:
   code=p.wait();log.close()
   if code:failed.append([seed,code])
  if failed:raise RuntimeError(f'{stage} failed: {failed}')
  run(stage+'_aggregation',[sys.executable,str(ROOT/'scripts/summarize_full_update_cycling.py'),stage])
 try:
  status('contract_tests')
  with (out/'contract_tests.log').open('ab',buffering=0) as log:
   subprocess.run([sys.executable,'-m','pytest','-q','tests/test_full_update_cycling_contract.py','tests/test_metar_multivar.py'],cwd=ROOT,env=dict(os.environ,PYTHONPATH='src:scripts'),stdout=log,stderr=subprocess.STDOUT,check=True)
  upstream=ROOT/'outputs/single_da_07_full82_small_stopped'
  while True:
   state=json.loads((upstream/'status.json').read_text()) if (upstream/'status.json').exists() else {}
   if 'failed' in state.get('stage',''):raise RuntimeError('Upstream training failed')
   if state.get('stage')=='complete':break
   status('waiting_for_single_cycle_models',upstream=state,required=['seed42 DA/NC best','seed43 DA/NC best']);time.sleep(30)
  # Verify selected checkpoints rather than trusting a completion flag.
  import torch
  from run_full_update_v1 import Data
  signatures={s:Data(ROOT,s).signature for s in ['train','val']}
  files={}
  for seed in [42,43]:
   for kind in ['da','neural_correction']:
    directory=upstream/f'seed{seed}'/kind/'formal'
    result=json.loads((directory/'result.json').read_text());ck=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    if result['selected_validation']['cycles']!=1460 or ck['step']!=result['best_step']:raise RuntimeError('Incomplete or incorrectly selected model')
    if ck['signature']['data']!=signatures or ck['signature']['config_sha256']!=digest(directory/'config.json'):raise RuntimeError('Checkpoint provenance mismatch')
    if not result['checks']['checkpoint_reload_exact']:raise RuntimeError('Upstream reload gate failed')
    if kind=='da' and not result['checks']['empty_exact_identity']:raise RuntimeError('Upstream noobs gate failed')
    for name in ['best.pt','config.json','result.json']:files[str((directory/name).relative_to(ROOT))]=digest(directory/name)
  # Freeze all imported project code and model/data normalization inputs after upstream completion.
  for folder in [ROOT/'src/offgrid_da/models',ROOT/'src/offgrid_da/data',ROOT/'src/offgrid_da/utils']:
   for file in folder.glob('*.py'):files[str(file.relative_to(ROOT))]=digest(file)
  for relative in ['scripts/run_full_update_v1.py','scripts/train_single_cycle.py','scripts/analyze_single_cycle.py','scripts/run_full_update_cycling.py','scripts/check_full_update_cycling.py','scripts/summarize_full_update_cycling.py','scripts/run_full_update_cycling_pipeline.py','outputs/single_da_05_input82_ablation/normalization.json','outputs/normalization/arches_train_2018_2020.json','configs/data/wb2_metar.yaml']:
   files[relative]=digest(ROOT/relative)
  frozen={'files':files,'data_signatures':signatures,'seeds':[42,43],'GPUs':[2,3],'alpha':1,'output_channels':82,'observations':['temperature','dewpoint','U10','V10','MSLP'],'primary_metric':'group-balanced normalized RMSE using frozen training background scales','selection':'existing single-cycle validation best; no cycling reselection','stages':['real_training_batch_gate','two_step_smoke','four_season_5day_pilot','24_window_5day_development'],'dewpoint_direct_station_score':False,'test_2022_used':False,'physical_units':'T/Td K, wind m/s, MSLP Pa; upper channels native units; never label normalized score kelvin'}
  if (out/'frozen.json').exists() and json.loads((out/'frozen.json').read_text())!=frozen:raise RuntimeError('Frozen cycling protocol changed')
  atomic_json(out/'frozen.json',frozen)
  for relative in files:
   if relative.endswith('.py'):
    target=out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/relative).read_bytes())
  gpu_wait([2]);run('real_batch_gate',[sys.executable,str(ROOT/'scripts/check_full_update_cycling.py')],2)
  geo=ROOT/'experiments/archesweather-reproduction/geoarches/.venv/bin/python'
  gpu_wait([2]);run('smoke',[str(geo),'-u',str(ROOT/'scripts/run_full_update_cycling.py'),'--seed','42','--steps','2','--smoke'],2)
  smoke=json.loads((out/'smoke/seed42/result.json').read_text())
  if not smoke['complete'] or not smoke['masked_cycle_exact'] or smoke['rows']!=8:raise RuntimeError('Two-step smoke failed')
  pair('pilot',['--pilot'])
  pair('formal',[])
  status('complete',reports=['pilot_report.md','formal_report.md'],all82_scored=True)
 except Exception as e:
  status('failed',error=repr(e));traceback.print_exc();raise
if __name__=='__main__':main()
