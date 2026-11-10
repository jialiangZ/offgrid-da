"""Run two seeds and automatically audit/aggregate scaling pilot."""
from pathlib import Path
import json,os,sys,subprocess,hashlib,fcntl
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from offgrid_da.data.arches_training import atomic_json

def rows(path):return [json.loads(x) for x in path.read_text().splitlines()]
def main():
 out=ROOT/'outputs/cycling_t2m_02_alpha_scaling';out.mkdir(exist_ok=True)
 lock=(out/'dispatcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 script=ROOT/'scripts/run_cycling_scaling.py';sha=hashlib.sha256(script.read_bytes()).hexdigest()
 # alpha=1 must reproduce original pilot; alpha=0 and masked paths checked inside each run.
 original=rows(ROOT/'outputs/cycling_t2m_01_pilot_4win_5d/smoke/seed42/metrics.jsonl')
 smoke=rows(out/'smoke/seed42/metrics.jsonl')
 original={(x['step'],x['method']):x for x in original}
 for row in smoke:
  method={'NC_1':'NC','DA_1':'DA','free':'free','masked_DA':'masked_DA'}.get(row['method'])
  if method:
   for phase in ['background','analysis']:
    key=phase+'_rmse_K_or_native_units'
    np.testing.assert_allclose(row[key],original[row['step'],method][key],rtol=0,atol=1e-10)
 atomic_json(out/'protocol.json',{'seeds':[42,43],'gpus':[0,3],'alphas':[0,.25,.5,1],'windows':['2021-01-15T00','2021-04-15T00','2021-07-15T00','2021-10-15T00'],'steps':20,'methods':['free','NC','DA','masked_DA'],'smoke_alpha1_reproduces_original':True,'primary_checks':'T2M benefit versus all 81 other channel changes; no tuning or new test claim','script_sha256':sha})
 (out/'source_snapshot.py').write_bytes(script.read_bytes())
 processes=[]
 for seed,gpu in [(42,0),(43,3)]:
  directory=out/'formal'/f'seed{seed}'
  if (directory/'result.json').exists():continue
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='src:scripts:experiments/archesweather-reproduction/geoarches')
  log=(out/f'seed{seed}.log').open('ab',buffering=0)
  process=subprocess.Popen([str(ROOT/'experiments/archesweather-reproduction/geoarches/.venv/bin/python'),'-u',str(script),'--seed',str(seed)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  processes.append((seed,process,log))
 atomic_json(out/'status.json',{'stage':'running','children':{str(s):p.pid for s,p,l in processes}})
 errors=[]
 for seed,p,log in processes:
  code=p.wait();log.close()
  if code:errors.append([seed,code])
 if errors:
  atomic_json(out/'status.json',{'stage':'failed','errors':errors});raise RuntimeError(errors)
 summary={};report=['# Cycling increment scaling pilot','Development windows only. Negative delta means improvement.','', '|Seed|Method|T2M RMSE K|Change vs free K|Other channels worse /81|Worst pooled relative degradation %|','|---|---|---:|---:|---:|---:|']
 for seed in [42,43]:
  records=rows(out/f'formal/seed{seed}/metrics.jsonl');assert len(records)==800
  assert len({(v['window'],v['step'],v['method']) for v in records})==800
  groups={}
  for row in records:groups.setdefault(row['method'],[]).append(row)
  scores={m:np.sqrt(np.mean(np.array([x['analysis_rmse_K_or_native_units'] for x in group])**2,axis=0)) for m,group in groups.items()}
  free=scores['free'];detail={}
  for m,score in scores.items():
   assert np.isfinite(score).all()
   relative=(score-free)/free*100;other=np.delete(relative,2)
   detail[m]={'T2M_rmse_K':float(score[2]),'T2M_minus_free_K':float(score[2]-free[2]),'nonT2M_worse_count':int((other>0).sum()),'worst_nonT2M_degradation_percent':float(other.max()),'per_channel_relative_change_percent':relative.tolist()}
   report.append(f'|{seed}|{m}|{score[2]:.6f}|{score[2]-free[2]:+.6f}|{int((other>0).sum())}|{other.max():+.4f}|')
  # Original alpha1 trajectory must remain reproducible for every scored cycle.
  old=rows(ROOT/f'outputs/cycling_t2m_01_pilot_4win_5d/formal/seed{seed}/metrics.jsonl')
  lookup={(v['window'],v['step'],v['method']):v for v in old}
  for v in records:
   original_method={'DA_1':'DA','NC_1':'NC','free':'free'}.get(v['method'])
   if original_method:
    for phase in ['background','analysis']:
     key=phase+'_rmse_K_or_native_units'
     np.testing.assert_allclose(v[key],lookup[v['window'],v['step'],original_method][key],rtol=0,atol=1e-10)
  # Existing NC/DA cross-variable diagnostic, including per-lead and per-window values.
  diagnostic={}
  for method in ['NC','DA']:
   group=[v for v in old if v['method']==method]
   values=np.sqrt(np.mean(np.array([v['analysis_rmse_K_or_native_units'] for v in group])**2,axis=0))
   rel=(values-free)/free*100
   diagnostic[method]={'nonT2M_worse_count':int((np.delete(rel,2)>0).sum()),'per_channel_relative_change_percent':rel.tolist()}
  summary[str(seed)]={'scaling':detail,'original_cross_variable_diagnostic':diagnostic}
 atomic_json(out/'summary.json',{'complete':True,'seeds':summary,'alpha1_exact_reproduction':True,'note':'Validation development only; channel counts are correlated and not significance tests; alpha0 exact equals free.'})
 (out/'scaling_report.md').write_text('\n'.join(report))
 atomic_json(out/'status.json',{'stage':'complete','audit':'800 records per seed; alpha1 reproduces original; finite metrics'})
if __name__=='__main__':main()
