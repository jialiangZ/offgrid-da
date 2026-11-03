"""Generate resumable six-hour ArchesWeather T2M backgrounds over [start,end)."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np,torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.main_hydra import collate_fn
root=Path('/home/jlzhang/Documents/offgrid-da'); sys.path.insert(0,str(root/'src'))
from offgrid_da.data.forecast_contract import archesweather_t2m_background
geo=root/'experiments/archesweather-reproduction/geoarches'; exp=geo.parent
p=argparse.ArgumentParser(); p.add_argument('--start',required=True); p.add_argument('--end',required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cpu'); args=p.parse_args()
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''): h.update(b)
 return h.hexdigest()
def save_json(path,obj):
 tmp=path.with_suffix('.json.tmp'); tmp.write_text(json.dumps(obj,indent=2)+'\n'); os.replace(tmp,path)
f=json.loads((root/'outputs/forecast_freeze/archesweather_250k_20261031/manifest.json').read_text()); ck=Path(f['checkpoint']['path']); config=Path(f['config']['source']); sp=Path(f['normalization']['source'])
for path,key in [(ck,'checkpoint'),(config,'config'),(sp,'normalization')]:
 if sha(path)!=f[key]['sha256']: raise RuntimeError('frozen hash mismatch: '+str(path))
start=np.datetime64(args.start,'h'); end=np.datetime64(args.end,'h')
if start>=end or int(start.astype('int64'))%6 or int(end.astype('int64'))%6: raise ValueError('start/end must be aligned to 6h UTC')
args.output.mkdir(parents=True,exist_ok=True); mp=args.output/'manifest.json'
metadata={'checkpoint_sha256':f['checkpoint']['sha256'],'config_sha256':f['config']['sha256'],'statistics_sha256':f['normalization']['sha256'],'start_valid_utc':str(start)+'Z','end_valid_exclusive_utc':str(end)+'Z','device':args.device,'surface_variables':['10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature','mean_sea_level_pressure'],'surface_units':['m/s','m/s','K','Pa'],'scope':'single-cycle backgrounds from ERA5 initial states','samples':[]}
if mp.exists():
 old=json.loads(mp.read_text())
 for key in ('checkpoint_sha256','config_sha256','statistics_sha256','start_valid_utc','end_valid_exclusive_utc'):
  if old[key]!=metadata[key]: raise ValueError('resume metadata mismatch: '+key)
 metadata=old
 for sample in metadata['samples']: sample.setdefault('device',metadata.get('device','unknown'))
metadata.setdefault('device_history',[])
if args.device not in metadata['device_history']: metadata['device_history'].append(args.device)
completed={x['valid_utc']:x for x in metadata['samples']}
torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; cfg=OmegaConf.load(config); OmegaConf.resolve(cfg); model=instantiate(cfg.module.module,cfg.module); state=torch.load(ck,map_location='cpu',weights_only=False)
assert state['global_step']==250000; model.load_state_dict(state['state_dict'],strict=True); del state; model.eval().to(args.device); stats=torch.load(sp,map_location='cpu',weights_only=True)
ds=Era5Forecast(path=str(exp/'data/era5_240/full'),domain='all',multistep=0,lead_time_hours=6,load_prev=True,norm_scheme='train_stats',norm_stats_path=str(sp)); mapping={ds.id2pt[i+1][2]:i for i in range(len(ds))}
valids=np.arange(start,end,np.timedelta64(6,'h')).astype('datetime64[ns]')
for valid in valids:
 init=valid-np.timedelta64(6,'h'); key=np.datetime_as_string(valid,unit='s')+'Z'
 if key in completed:
  item=completed[key]; path=args.output/item['file']
  if not path.exists() or sha(path)!=item['sha256']: raise RuntimeError('resume file failed checksum: '+str(path))
  continue
 if init not in mapping: raise RuntimeError('input time unavailable: '+str(init))
 idx=mapping[init]; assert ds.id2pt[idx][2]==init-np.timedelta64(6,'h')
 batch=collate_fn([ds[idx]]); assert int(batch['timestamp'][0])==int(init.astype('datetime64[s]').astype('int64'))
 batch={k:v.to(args.device) for k,v in batch.items()}
 with torch.inference_mode(): pred=model.forward(batch)
 converted=archesweather_t2m_background(pred,stats,[np.datetime_as_string(init,unit='s')+'Z']); physical=ds.denormalize(pred)['surface'].float().roll(-120,-1).cpu().numpy()[0]
 if not np.isfinite(physical).all(): raise RuntimeError('nonfinite output')
 name='background_'+np.datetime_as_string(init,unit='h').replace('-','')+'.npz'; tmp=args.output/(name+'.tmp')
 with tmp.open('wb') as stream: np.savez_compressed(stream,surface=physical,latitude=np.linspace(90,-90,121),longitude=np.arange(240)*1.5)
 os.replace(tmp,args.output/name)
 item={'init_utc':np.datetime_as_string(init,unit='s')+'Z','previous_utc':np.datetime_as_string(init-np.timedelta64(6,'h'),unit='s')+'Z','valid_utc':key,'file':name,'sha256':sha(args.output/name),'device':args.device,'precision':'float32'}
 metadata['samples'].append(item); save_json(mp,metadata); print(f'{len(metadata["samples"])}/{len(valids)} {key}',flush=True)
print('COMPLETE',flush=True)
