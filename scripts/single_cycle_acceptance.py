"""Single-cycle DA baseline and bounded training acceptance."""
import argparse,json,sys,random,hashlib
from pathlib import Path
import numpy as np,torch
root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root/'src'))
from offgrid_da.data.arches_dataset import ArchesSingleCycleDADataset,ArchesBackgroundReader
from offgrid_da.data.analysis import ERA5Reader
from offgrid_da.data.collate import single_cycle_collate
from offgrid_da.models import CrossAttentionAnalysisModel
from offgrid_da.evaluation.metrics import _area_weighted_error_sums
from offgrid_da.utils.config import load_config
p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['baseline','train'],required=True); p.add_argument('--steps',type=int,default=30); p.add_argument('--resume',type=Path); p.add_argument('--output',type=Path,required=True); p.add_argument('--train-end',default='2018-01-02'); p.add_argument('--val-end',default='2021-01-02'); p.add_argument('--sample-limit',type=int,default=2); p.add_argument('--batch-size',type=int,default=1); p.add_argument('--val-every',type=int,default=5); p.add_argument('--save-every',type=int,default=10); args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
norm=root/'outputs/normalization/arches_train_2018_2020.json'; manifests={s:root/f'outputs/arches_backgrounds_{s}_{y}/manifest.json' for s,y in [('train','2018_2020'),('val','2021')]}
if args.mode=='baseline':
 reader=ArchesBackgroundReader(manifests['val']); era=ERA5Reader(load_config()['data']['era5']['url']); total=np.zeros(3,dtype=np.float64); rows=[]
 for i,(valid,(init,_,_)) in enumerate(sorted(reader.rows.items())):
  bg=reader.read_background(init,6,['2m_temperature']); ref=era.read_analysis(valid,['2m_temperature']); sums=np.array(_area_weighted_error_sums(bg,ref,reader.grid()['latitude'])); total+=sums; rows.append({'valid_time':str(valid),'rmse_K':float(np.sqrt(sums[0]/sums[2]))})
  if (i+1)%100==0: print('baseline',i+1,len(reader.rows),flush=True)
 report={'cycles':len(rows),'pooled_area_weighted_rmse_K':float(np.sqrt(total[0]/total[2])),'pooled_area_weighted_mae_K':float(total[1]/total[2]),'mean_cycle_rmse_K':float(np.mean([x['rmse_K'] for x in rows])),'reference':'ERA5','weight':'cos(latitude)','per_cycle':rows}
 (args.output/'baseline0.json').write_text(json.dumps(report,indent=2)+'\n'); print('BASELINE COMPLETE',report['pooled_area_weighted_rmse_K'],flush=True); sys.exit(0)
import swanlab
torch.manual_seed(42); np.random.seed(42); random.seed(42); torch.set_num_threads(4); device='cuda' if torch.cuda.is_available() else 'cpu'
sets={s:ArchesSingleCycleDADataset(manifests[s],start=f'{y}-01-01',end=args.train_end if s=='train' else args.val_end,normalization_path=norm) for s,y in [('train',2018),('val',2021)]}; samples={s:[d[i] for i in range(len(d) if args.sample_limit==0 else min(args.sample_limit,len(d)))] for s,d in sets.items()}
for s,items in samples.items():
 for sample in items:
  assert set(sample['context_station_ids']).isdisjoint(sample['heldout_station_ids'])
  assert torch.isfinite(sample['background_normalized']).all() and torch.isfinite(sample['reference_normalized']).all()
  mean=sets[s].norm_stats.mean('background','2m_temperature'); std=sets[s].norm_stats.std('background','2m_temperature'); torch.testing.assert_close(sample['reference_normalized'],(sample['reference']-mean)/std)
def pack(items):
 b=single_cycle_collate(items); assert all(int(b['observation_mask'][i].sum())==len(x['obs_values']) for i,x in enumerate(items)); keys=['obs_values_normalized','obs_locations','obs_time_offsets','obs_variables','obs_sources','obs_quality','observation_mask']; return b['background_normalized'].to(device),{k:b[k].to(device) for k in keys},b['reference_normalized'].to(device)
config={'dim':8,'heads':2,'depth':1,'max_neighbors':8,'query_chunk':512}; model=CrossAttentionAnalysisModel(**config).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4); start=0
if args.resume:
 state=torch.load(args.resume,map_location=device,weights_only=False); assert state['norm_sha256']==hashlib.sha256(norm.read_bytes()).hexdigest(); model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); start=state['step']; torch.set_rng_state(state['torch_rng'].cpu())
run=swanlab.init(project='offgrid-da',experiment_name='single-cycle-acceptance',mode='offline',logdir=str(args.output/'swanlogs'),config=config); history=[]
context={x for items in samples.values() for sample in items for x in sample['context_station_ids']}
heldout={x for items in samples.values() for sample in items for x in sample['heldout_station_ids']}
assert context.isdisjoint(heldout)
weights=torch.cos(torch.deg2rad(torch.linspace(90,-90,121,device=device))).clamp_min(0)[None,None,:,None]
def weighted_mse(a,b):
 return ((a-b).square()*weights).sum()/weights.expand_as(a).sum()
def validate():
 model.eval(); total=0.; count=0
 with torch.no_grad():
  for start in range(0,len(samples['val']),args.batch_size):
   items=samples['val'][start:start+args.batch_size]; x,o,y=pack(items); total+=float(weighted_mse(model(x,o)['analysis'],y))*len(items); count+=len(items)
 return total/count
def save(step):
 path=args.output/f'checkpoint_{step}.pt'; torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'model_config':config,'norm_sha256':hashlib.sha256(norm.read_bytes()).hexdigest(),'torch_rng':torch.get_rng_state()},path); return path
for step in range(start,args.steps):
 model.train(); x,o,y=pack([samples['train'][(step*args.batch_size+j)%len(samples['train'])] for j in range(args.batch_size)]); opt.zero_grad(); loss=weighted_mse(model(x,o)['analysis'],y); assert torch.isfinite(loss); loss.backward(); assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()); opt.step(); item={'step':step+1,'train_loss':float(loss)}
 if (step+1)%args.val_every==0: item['val_loss']=validate(); item['val_rmse_K']=item['val_loss']**0.5*sets['val'].norm_stats.std('background','2m_temperature'); print(item,flush=True)
 swanlab.log(item,step=step+1); history.append(item)
 if (step+1)%args.save_every==0: save(step+1)
path=save(args.steps); model.eval(); x,o,y=pack(samples['val'][:args.batch_size])
with torch.no_grad(): expected=model(x,o)['analysis']
other=CrossAttentionAnalysisModel(**config).to(device); state=torch.load(path,map_location=device,weights_only=False); other.load_state_dict(state['model']); other.eval()
with torch.no_grad(): torch.testing.assert_close(other(x,o)['analysis'],expected,rtol=0,atol=0)
(args.output/'acceptance.json').write_text(json.dumps({'scope':'configured single-cycle run','train_cycles':len(samples['train']),'validation_cycles':len(samples['val']),'device':device,'history':history,'checkpoint_reload_exact':True,'station_disjoint':True,'common_normalization':True},indent=2)+'\n'); swanlab.finish(); print('TRAIN ACCEPTANCE COMPLETE',flush=True)
