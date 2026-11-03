import hashlib,json,sys
from pathlib import Path
import numpy as np, torch, xarray as xr
from hydra.utils import instantiate
from omegaconf import OmegaConf
from geoarches.dataloaders.era5 import Era5Forecast, surface_variables, level_variables, pressure_levels
from geoarches.main_hydra import collate_fn
root=Path('/home/jlzhang/Documents/offgrid-da'); sys.path.insert(0,str(root/'src'))
from offgrid_da.data.forecast_contract import archesweather_t2m_background
geo=root/'experiments/archesweather-reproduction/geoarches'; exp=geo.parent
freeze=root/'outputs/forecast_freeze/archesweather_250k_20261031/manifest.json'
m=json.loads(freeze.read_text()); ck=Path(m['checkpoint']['path']); config=Path(m['config']['source']); statsfile=Path(m['normalization']['source'])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''): h.update(b)
 return h.hexdigest()
for p,k in [(ck,'checkpoint'),(config,'config'),(statsfile,'normalization')]:
 if sha(p)!=m[k]['sha256']: raise RuntimeError('frozen hash mismatch: '+str(p))
torch.set_num_threads(4); torch.set_grad_enabled(False)
cfg=OmegaConf.load(config); OmegaConf.resolve(cfg); model=instantiate(cfg.module.module,cfg.module)
state=torch.load(ck,map_location='cpu',weights_only=False)
assert state['global_step']==250000
model.load_state_dict(state['state_dict'],strict=True); del state; model.eval()
stats=torch.load(statsfile,map_location='cpu',weights_only=True)
ds=Era5Forecast(path=str(exp/'data/era5_240/full'),domain='test',multistep=0,lead_time_hours=6,load_prev=True,norm_scheme='train_stats',norm_stats_path=str(statsfile))
mapping={ds.id2pt[i+1][2]:i for i in range(len(ds))}
init_times=np.arange(np.datetime64('2020-01-02T00','ns'),np.datetime64('2020-01-02T18','ns'),np.timedelta64(6,'h'))
out=root/'outputs/forecast_contract/arches250k_real_20201031'; out.mkdir(parents=True,exist_ok=False)
results=[]
for t in init_times:
 i=mapping[t]; assert ds.id2pt[i][2]==t-np.timedelta64(6,'h')
 batch=collate_fn([ds[i]]); assert int(batch['timestamp'][0])==int(t.astype('datetime64[s]').astype('int64'))
 with torch.inference_mode(): pred=model.forward(batch)
 assert all(torch.isfinite(pred[k]).all() for k in ('surface','level'))
 iso=np.datetime_as_string(t,unit='s')+'Z'; conv=archesweather_t2m_background(pred,stats,[iso])
 physical=ds.denormalize(pred); surf=physical['surface'].float().roll(-120,-1); lev=physical['level'].float().roll(-120,-1)
 assert torch.isfinite(surf).all() and torch.isfinite(lev).all()
 torch.testing.assert_close(surf[:,2],conv['background'],rtol=1e-5,atol=1e-4)
 name='background_'+np.datetime_as_string(t,unit='h').replace('-','').replace('T','T')+'.npz'
 np.savez_compressed(out/name,surface=surf[0].numpy(),level=lev[0].numpy(),latitude=np.linspace(90,-90,121),longitude=np.arange(240)*1.5,pressure_hpa=np.array(pressure_levels))
 results.append({'init_utc':iso,'previous_utc':np.datetime_as_string(t-np.timedelta64(6,'h'),unit='s')+'Z','valid_utc':conv['valid_times'][0].isoformat(),'file':name,'sha256':sha(out/name),'t2m_min_K':float(surf[:,2].min()),'t2m_max_K':float(surf[:,2].max())})
 print(json.dumps(results[-1]),flush=True)
report={'checkpoint_sha256':m['checkpoint']['sha256'],'config_sha256':m['config']['sha256'],'statistics_sha256':m['normalization']['sha256'],'input_data':str(exp/'data/era5_240/full'),'device':'cpu float32','surface_variables':surface_variables,'surface_units':['m/s','m/s','K','Pa'],'level_variables':level_variables,'level_units':['m2/s2','m/s','m/s','K','kg/kg','Pa/s'],'surface_shape':[4,1,121,240],'level_shape':[6,13,121,240],'grid':'latitude 90 to -90; longitude 0 to 358.5; 1.5 degrees','pressure_hpa':pressure_levels,'checks':['frozen hashes','strict checkpoint load','input timestamp and previous index','finite full output','T2M adapter agrees with denormalization','six-hour valid time'],'samples':results}
(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print('COMPLETE',out,flush=True)
