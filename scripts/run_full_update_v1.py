"""Complete-state single-cycle experiment, independent of frozen earlier sources."""
import argparse,json,sys,hashlib,math,os,subprocess,time,fcntl
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
import xarray as xr
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from offgrid_da.data.multivar_dataset import MultivarMetarDataset,multivar_collate
from offgrid_da.data.full_state_data import SURFACE,LEVEL,PRESSURE,CHANNELS
from offgrid_da.models import CrossAttentionAnalysisModel
from offgrid_da.data.arches_training import atomic_json
import train_single_cycle as trainer

@lru_cache(maxsize=4)
def opened(path):return xr.open_dataset(path)
class Data(MultivarMetarDataset):
 def __init__(self,root,split):
  super().__init__(root,split)
  self.signature=hashlib.sha256((self.signature+Path(__file__).read_text()).encode()).hexdigest()
 def __getitem__(self,index):
  s=super().__getitem__(index);t=self.valid_time(index)
  path=ROOT/f'experiments/archesweather-reproduction/data/era5_240/full/era5_240_{t.year}_{t.hour}h.nc'
  ds=opened(str(path)).sel(time=np.datetime64(t)).sortby('latitude',ascending=False).sortby('longitude')
  np.testing.assert_allclose(ds.latitude,np.linspace(90,-90,121));np.testing.assert_allclose(ds.longitude,np.arange(240)*1.5)
  a=[ds[v].transpose('latitude','longitude').values for v in SURFACE]
  for v in LEVEL:
   for p in PRESSURE:a.append(ds[v].sel(level=p).transpose('latitude','longitude').values)
  target=torch.tensor(np.stack(a),dtype=torch.float32)
  torch.testing.assert_close(target[2:3],s['reference'],rtol=1e-5,atol=1e-3)
  mean=self.base.mean;std=self.base.std
  s['reference']=target;s['reference_normalized']=(target-mean)/std
  s['background_normalized']=s['grid_state'];s['background']=s['grid_state']*std+mean
  assert torch.isfinite(s['reference_normalized']).all()
  return s

def channel_weights(x):
 return torch.tensor([.1]*4+[.1/13]*78,device=x.device)[None,:,None,None]
def loss(pred,target):
 w=trainer.weights_for(pred)*channel_weights(pred)
 return ((pred-target).square()*w).sum()/w.expand_as(pred).sum()
def evaluate(model,loader,device,std):
 if model:model.eval()
 errors=torch.zeros(82,dtype=torch.float64,device=device);base=errors.clone();den=0.;n=0
 with torch.inference_mode():
  for batch in loader:
   x,obs,y=trainer.pack(batch,device);p=x if model is None else model(x,obs)['analysis']
   assert torch.isfinite(p).all()
   w=trainer.weights_for(x).double()
   errors+=((p.double()-y.double()).square()*w).sum((0,2,3));base+=((x.double()-y.double()).square()*w).sum((0,2,3))
   den+=float(w.expand(x.shape[0],1,*x.shape[-2:]).sum());n+=len(x)
 scale=Data(ROOT,'train').base.std[:,0,0].double().to(device)
 cw=channel_weights(x).flatten().double()
 return {'rmse_K':float((errors/den*cw).sum().sqrt()),'baseline0_rmse_K':float((base/den*cw).sum().sqrt()),'cycles':n,
 'selection_metric':'dimensionless group-balanced normalized RMSE; rmse_K legacy key is NOT kelvin',
 'physical_rmse_by_channel':dict(zip(CHANNELS,((errors/den).sqrt()*scale).cpu().tolist()))}
def run_train():
 trainer.LazyArchesDataset=Data;trainer.CachedSamples=lambda d:d;trainer.single_cycle_collate=multivar_collate
 def factory(kind,cfg):
  model=CrossAttentionAnalysisModel(channels=82,**cfg)
  return model if kind=='da' else trainer.BackgroundCorrection(model)
 trainer.make_model=factory;trainer.training_loss=loss;trainer.evaluate=evaluate
 trainer.main()
def pipeline():
 out=ROOT/'outputs/single_da_07_full82_small_stopped';out.mkdir(exist_ok=True)
 lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 datasets={s:Data(ROOT,s) for s in ['train','val']}
 prep={'complete':False,'signatures':{s:d.signature for s,d in datasets.items()}}
 atomic_json(out/'preparation.json',prep)
 for split,d in datasets.items():
  for i in range(len(d)):
   d[i]
   if i%100==0:atomic_json(out/'status.json',{'stage':'full_target_audit','split':split,'index':i})
 prep['complete']=True;atomic_json(out/'preparation.json',prep)
 cfg=json.loads((ROOT/'outputs/single_da_05_input82_ablation/config.json').read_text());cfg['model'].pop('input_channels');cfg['model']['max_neighbors']=80
 cfg['model']['query_chunk']=256;cfg['loader_workers']=0
 for seed in [42,43]:
  for trial in [True,False]:
   children=[]
   for gpu,kind in [(0,'da'),(1,'neural_correction')]:
    while True:
     info=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().split(',')
     if int(info[0])<1024 and int(info[1])<10:break
     time.sleep(30)
    target=out/f'seed{seed}'/kind/('trial' if trial else 'formal');target.mkdir(parents=True,exist_ok=True)
    if (target/'result.json').exists():continue
    c=dict(cfg,seed=seed,name=f'full_update_{seed}_{kind}',selection_metric='normalized group-balanced RMSE',output_channels=82)
    atomic_json(target/'config.json',c)
    command=[sys.executable,'-u',__file__,'--train','--config',str(target/'config.json'),'--preparation',str(out/'preparation.json'),'--output',str(target),'--kind',kind]+(['--trial'] if trial else [])
    if (target/'last.pt').exists():command+=['--resume',str(target/'last.pt')]
    log=(target/'run.log').open('ab');p=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
    children.append((p,log))
   atomic_json(out/'status.json',{'stage':'trial' if trial else 'formal','seed':seed,'pids':[p.pid for p,_ in children]})
   for p,log in children:
    code=p.wait();log.close()
    if code:raise RuntimeError('Child failed; see run.log')
 atomic_json(out/'status.json',{'stage':'complete'})
if __name__=='__main__':
 if '--train' in sys.argv:sys.argv.remove('--train');run_train()
 else:pipeline()
