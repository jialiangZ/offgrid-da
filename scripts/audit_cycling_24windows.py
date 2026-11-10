import json,time,os,sys,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from offgrid_da.data.arches_training import atomic_json
out=ROOT/'outputs/cycling_t2m_03_24win_raw'
launch=json.loads((out/'launch.json').read_text())
while True:
 status={};failed=False
 for seed,entry in launch.items():
  directory=out/'formal'/f'seed{seed}'
  path=directory/'result.json';metrics=directory/'metrics.jsonl'
  if path.exists():status[seed]='complete'
  else:
   try:os.kill(entry['pid'],0);status[seed]='running'
   except ProcessLookupError:status[seed]='failed';failed=True
 atomic_json(out/'status.json',{'stage':'failed' if failed else 'running','seeds':status})
 if failed:raise RuntimeError('A cycling process exited without result')
 if all(v=='complete' for v in status.values()):break
 time.sleep(30)
summary={}
for seed in launch:
 directory=out/'formal'/f'seed{seed}'
 rows=[json.loads(l) for l in (directory/'metrics.jsonl').read_text().splitlines()]
 assert len(rows)==1920 and len({(v['window'],v['step'],v['method']) for v in rows})==1920
 methods={}
 for method in ['free','NC','DA','masked_DA']:
  group=[v for v in rows if v['method']==method];assert len(group)==480
  methods[method]={}
  for phase in ['background','analysis']:
   a=np.array([v[phase+'_rmse_K_or_native_units'] for v in group]);assert np.isfinite(a).all()
   methods[method][phase+'_per_channel_rmse']=np.sqrt(np.mean(a*a,axis=0)).tolist()
 assert methods['free']==methods['masked_DA']
 summary[seed]=methods
atomic_json(out/'summary.json',{'complete':True,'per_seed':summary,'cycles_per_seed':480,'note':'2021 development only; RMS pooling of per-cycle weighted errors, physical units per channel.'})
atomic_json(out/'status.json',{'stage':'complete','seeds':{k:'complete' for k in launch},'coverage_and_finite_audit':True})
