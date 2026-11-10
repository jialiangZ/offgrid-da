import sys,time,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from cycling_prefetch import CyclingInputs,make_loader
from run_full_update_v1 import Data
from offgrid_da.data.multivar_dataset import multivar_collate
from train_single_cycle import OBSERVATION_KEYS
from offgrid_da.data.arches_training import atomic_json

def compare(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:compare(a[k],b[k])
 else:assert a==b

def main():
 torch.set_num_threads(4)
 freeze=json.loads((ROOT/'outputs/forecast_freeze/archesweather_250k_20261031/manifest.json').read_text())
 windows=['2021-01-05T00','2021-04-05T00'];steps=12
 serial=CyclingInputs(ROOT,windows,steps,freeze['normalization']['source'])
 reference=[];latency=[];old_latency=[]
 old=Data(ROOT,'val')
 for i in range(len(serial)):
  t=time.perf_counter();item=serial[i];latency.append(time.perf_counter()-t);reference.append(item)
  if i in [0,1,12,13]:
   valid=np.datetime64(item['valid_utc'],'ns');idx=int((valid-np.datetime64('2021-01-01T00','ns'))/np.timedelta64(6,'h'))
   t=time.perf_counter();original=multivar_collate([old[idx]]);old_latency.append(time.perf_counter()-t)
   for key in OBSERVATION_KEYS+('heldout_obs_values','heldout_obs_variables','heldout_obs_locations','heldout_obs_mask'):
    compare(item['sample'][key],original[key])
 waits=[];loader=make_loader(CyclingInputs(ROOT,windows,steps,freeze['normalization']['source']),2);iterator=iter(loader)
 for expected in reference:
  t=time.perf_counter();actual=next(iterator);waits.append(time.perf_counter()-t);compare(actual,expected)
 try:next(iterator);raise AssertionError('Unexpected extra sample')
 except StopIteration:pass
 del iterator,loader
 # Check actual GPU reconstruction and batched regional reductions independently
 # of DA weights. This is an engineering input/reduction test, not an experiment.
 from tensordict import TensorDict
 from geoarches.main_hydra import collate_fn
 from analyze_single_cycle import regions
 first=reference[0]
 original=collate_fn([serial._weather[serial._mapping[np.datetime64(first['valid_utc'],'ns')]]])['state'].to('cuda:0')
 rebuilt=TensorDict({'surface':first['reference_surface'].cuda(),'level':first['reference_level'].cuda()},batch_size=[1])
 for key in ['surface','level']:assert torch.equal(original[key],rebuilt[key])
 field=serial._weather.denormalize(rebuilt)
 canonical=torch.cat([field['surface'][:,:,0],field['level'].flatten(1,2)],1).roll(-120,-1)
 error=(canonical-canonical*.999).double().square()
 weight=torch.cos(torch.deg2rad(torch.linspace(90,-90,121,device='cuda'))).clamp_min(0)[None,:,None].double()
 masks={name:torch.tensor(mask,device='cuda') for name,mask in regions(np.linspace(90,-90,121),np.arange(240)*1.5).items()}
 stacked=torch.stack([(weight*mask).reshape(-1) for mask in masks.values()])
 fast=((error.flatten(2)[0] @ stacked.T)/stacked.sum(-1)).sqrt().T
 for i,mask in enumerate(masks.values()):
  expected=((error*weight*mask).sum((-2,-1))/(weight*mask).sum()).sqrt()[0]
  torch.testing.assert_close(fast[i],expected,atol=1e-10,rtol=1e-12)
 out=ROOT/'outputs/cycling_full82_02_large';out.mkdir(exist_ok=True)
 atomic_json(out/'prefetch_verification.json',{'passed':True,'cycles':len(reference),'all_tensors_and_order_exact':True,'old_sample_observations_exact':True,'GPU_reference_reconstruction_exact':True,'batched_regional_reduction_matches':True,'workers':2,'prefetch_factor':2,'serial_cpu_seconds_after_warmup':float(np.mean(latency[4:])),'old_full_sample_cpu_seconds':float(np.mean(old_latency)),'two_worker_wait_seconds_after_warmup':float(np.mean(waits[4:])),'note':'CPU input benchmark only; no end-to-end GPU speedup claim. Recursive forecast/analysis states never enter loader.'})
 print((out/'prefetch_verification.json').read_text())
if __name__=='__main__':main()
