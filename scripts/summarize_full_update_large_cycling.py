import sys,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from offgrid_da.data.arches_training import atomic_json
from offgrid_da.data.full_state_data import CHANNELS
out=ROOT/'outputs/cycling_full82_02_large';stage=sys.argv[1]
std=np.array(json.loads((ROOT/'outputs/single_da_05_input82_ablation/normalization.json').read_text())['input_std'])
cw=np.array([.1]*4+[.1/13]*78)
summary={};report=['# Full82 multivariate Cycling evaluation',f'Stage: {stage}; 2021 development only. Compare matched full-state NC and DA; never mix physical units. Dewpoint is input only and not directly station-scored without a validated observation operator.','']
for seed in [42,43]:
 rows=[json.loads(l) for l in (out/stage/f'seed{seed}/metrics.jsonl').read_text().splitlines()]
 count=320 if stage=='pilot' else 1920
 assert len(rows)==count and len({(x['window'],x['step'],x['method']) for x in rows})==count
 def aggregate(group):
  result={}
  for phase in ['background','analysis']:
   fields=np.array([x[phase+'_rmse_K_or_native_units'] for x in group]);assert np.isfinite(fields).all()
   physical=np.sqrt(np.mean(fields**2,axis=0));score=float(np.sqrt(np.sum((physical/std)**2*cw)))
   result[phase]={'physical_rmse_by_channel':dict(zip(CHANNELS,physical.tolist())),'group_balanced_normalized_rmse':score,'regions':{reg:np.sqrt(np.mean(np.array([x['regional_'+phase][reg] for x in group])**2,axis=0)).tolist() for reg in group[0]['regional_'+phase]}}
  result['heldout_by_type']={}
  for variable in ['0','1','2','4']:
   n=sum(x['heldout_by_type'][variable]['count'] for x in group)
   result['heldout_by_type'][variable]={'reports':n,**{phase+'_rmse':math.sqrt(sum(x['heldout_by_type'][variable][phase+'_sse'] for x in group)/n) if n else None for phase in ['background','analysis','reference']}}
  return result
 methods={}
 for method in ['free','NC','DA','masked_DA']:
  group=[x for x in rows if x['method']==method]
  methods[method]={'overall':aggregate(group),'by_lead_hours':{str(step*6):aggregate([x for x in group if x['step']==step]) for step in range(1,21)},'by_window':{w:aggregate([x for x in group if x['window']==w]) for w in sorted({x['window'] for x in group})}}
 assert methods['free']==methods['masked_DA']
 # Free model is identical to prior T2M experiment; this checks the replay and reference contract.
 old=ROOT/('outputs/cycling_t2m_01_pilot_4win_5d' if stage=='pilot' else 'outputs/cycling_t2m_03_24win_raw')/'formal'/f'seed{seed}'/'metrics.jsonl'
 oldrows=[json.loads(l) for l in old.read_text().splitlines() if json.loads(l)['method']=='free']
 lookup={(x['window'],x['step']):x for x in oldrows}
 for row in rows:
  if row['method']=='free':np.testing.assert_allclose(row['analysis_rmse_K_or_native_units'],lookup[row['window'],row['step']]['analysis_rmse_K_or_native_units'],rtol=0,atol=1e-10)
 summary[str(seed)]=methods
 report.extend([f'## Seed {seed}','|Method|Normalized background RMSE|Normalized analysis RMSE|T2M analysis K|','|---|---:|---:|---:|'])
 for method in ['free','NC','DA']:
  d=methods[method]['overall'];report.append(f"|{method}|{d['background']['group_balanced_normalized_rmse']:.8f}|{d['analysis']['group_balanced_normalized_rmse']:.8f}|{d['analysis']['physical_rmse_by_channel']['2m_temperature']:.6f}|")
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for method in ['free','NC','DA']:
  values=[methods[method]['by_lead_hours'][str(step*6)]['analysis'] for step in range(1,21)]
  axes[0].plot(np.arange(1,21)/4,[v['group_balanced_normalized_rmse'] for v in values],label=method)
  axes[1].plot(np.arange(1,21)/4,[v['physical_rmse_by_channel']['2m_temperature'] for v in values],label=method)
 axes[0].set(ylabel='Group-balanced normalized RMSE',xlabel='Days');axes[1].set(ylabel='T2M RMSE K',xlabel='Days')
 for ax in axes:ax.legend()
 fig.tight_layout();fig.savefig(out/f'{stage}_seed{seed}_curves.png',dpi=170);plt.close(fig)
atomic_json(out/f'{stage}_summary.json',{'complete':True,'per_seed':summary,'free_reproduces_previous':True,'all82_scored':True,'dewpoint_direct_score':False})
(out/f'{stage}_report.md').write_text('\n'.join(report))
