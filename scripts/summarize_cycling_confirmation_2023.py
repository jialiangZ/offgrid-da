import json,sys,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from offgrid_da.data.arches_training import atomic_json
out=ROOT/'outputs/cycling_t2m_06_confirm_2023_pending'
summary={};report=['# 15-day cycling evaluation','Predeclared 2023 assessment; no model reselection. Overlapping windows are not independent. Month groups refer to initialization month. All scores pooled over equal-grid times; station scores weighted by report counts. Approximately 2.5 K representativeness scale affects station interpretation. No independent-test claim.','']
for seed in [42,43]:
 rows=[json.loads(x) for x in (out/f'formal/seed{seed}/metrics.jsonl').read_text().splitlines()]
 assert len(rows)==5760 and len({(v['window'],v['step'],v['method']) for v in rows})==5760
 for v in rows:
  for phase in ['background','analysis']:
   key=phase+'_rmse_K_or_native_units'
   np.testing.assert_allclose(v['regional_'+phase]['global'],v[key],atol=1e-10,rtol=0)
 def aggregate(group):
  n=sum(x['heldout_count'] for x in group)
  result={'cycles':len(group),'station_reports':n}
  for phase in ['background','analysis']:
   result['regional_'+phase]={reg:np.sqrt(np.mean(np.array([x['regional_'+phase][reg] for x in group])**2,axis=0)).tolist() for reg in group[0]['regional_'+phase]}
   result['heldout_'+phase+'_rmse_K']=math.sqrt(sum(x['heldout_'+phase+'_sse'] for x in group)/n) if n else None
  result['heldout_reference_rmse_K']=math.sqrt(sum(x['heldout_reference_sse'] for x in group)/n) if n else None
  return result
 result={}
 for method in ['free','NC','DA','masked_DA']:
  group=[x for x in rows if x['method']==method]
  result[method]={'overall':aggregate(group),'by_initialization_month':{str(m):aggregate([x for x in group if int(x['window'][5:7])==m]) for m in range(1,13)},'by_lead_hours':{str(t*6):aggregate([x for x in group if x['step']==t]) for t in range(1,61)},'by_window':{w:aggregate([x for x in group if x['window']==w]) for w in sorted({x['window'] for x in group})}}
 assert result['free']==result['masked_DA']
 summary[str(seed)]=result
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for method in ['free','NC','DA']:
  d=result[method]
  for phase,style in [('background','--'),('analysis','-')]:
   axes[0].plot(range(6,361,6),[d['by_lead_hours'][str(t*6)]['regional_'+phase]['global'][2] for t in range(1,61)],ls=style,label=method+' '+phase)
  axes[1].plot(range(1,13),[d['by_initialization_month'][str(m)]['regional_analysis']['global'][2] for m in range(1,13)],label=method)
 axes[0].set(xlabel='Elapsed hours',ylabel='T2M RMSE K',title='Background vs analysis');axes[1].set(xlabel='Initialization month',ylabel='T2M RMSE K',title='Two 15-day windows per month')
 for ax in axes:ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(out/f'seed{seed}_curves.png',dpi=170);plt.close(fig)
 report.extend([f'## Seed {seed}','|Method|Global T2M background K|Global T2M analysis K|Heldout background K|Heldout analysis K|Reports|','|---|---:|---:|---:|---:|---:|'])
 for method in ['free','NC','DA']:
  d=result[method]['overall'];report.append(f"|{method}|{d['regional_background']['global'][2]:.6f}|{d['regional_analysis']['global'][2]:.6f}|{d['heldout_background_rmse_K']:.6f}|{d['heldout_analysis_rmse_K']:.6f}|{d['station_reports']}|")
 report.extend(['','|Region|Free analysis K|NC analysis K|DA analysis K|','|---|---:|---:|---:|'])
 for reg in result['DA']['overall']['regional_analysis']:
  scores=[result[m]['overall']['regional_analysis'][reg][2] for m in ['free','NC','DA']]
  report.append('|'+reg+'|'+'|'.join(f'{v:.6f}' for v in scores)+'|')
 report.append(f'![Curves](seed{seed}_curves.png)')
atomic_json(out/'summary.json',{'complete':True,'per_seed':summary,'all_82_channels':True,'predeclared_confirmation_year':2023,'station_caveat':'Heldout temperature reports, temporal/point-grid/representativeness differences; ~2.5 K scale is not an exact noise floor.'})
(out/'evaluation_report.md').write_text('\n'.join(report))
atomic_json(out/'status.json',{'stage':'complete','coverage':5760,'predeclared_protocol_used':True,'regional_global_consistency':True})
