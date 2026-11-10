"""Frozen seasonal five-day cycling pilot; ERA5 used only for initialization/scoring."""
import argparse, json, hashlib, subprocess, sys, os, time
from pathlib import Path
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.main_hydra import collate_fn
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from offgrid_da.data.full_state_data import FullStateInputDataset,full_state_collate,CHANNELS,digest
from offgrid_da.models import CrossAttentionAnalysisModel
from run_full_update_v1 import Data
from train_single_cycle import BackgroundCorrection
from offgrid_da.data.arches_training import atomic_json
from single_cycle_controls import FullStateCorrection
from train_single_cycle import pack,OBSERVATION_KEYS
from cycling_prefetch import CyclingInputs,make_loader
from tensordict import TensorDict
from analyze_single_cycle import regions
from offgrid_da.data.alignment import sample_background_at_obs

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=20);p.add_argument('--smoke',action='store_true');p.add_argument('--pilot',action='store_true');a=p.parse_args()
 frozen=json.loads((ROOT/'outputs/cycling_full82_02_large/frozen.json').read_text())
 for path,expected in frozen['files'].items():
  if digest(ROOT/path)!=expected:raise ValueError('Frozen cycling source/input changed: '+path)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 out=ROOT/'outputs/cycling_full82_02_large'/('smoke' if a.smoke else 'pilot' if a.pilot else 'formal')/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
 freeze=json.loads((ROOT/'outputs/forecast_freeze/archesweather_250k_20261031/manifest.json').read_text())
 for key,field in [('checkpoint','path'),('config','source'),('normalization','source')]:
  if digest(freeze[key][field])!=freeze[key]['sha256']:raise ValueError('Frozen forecast changed')
 geo=ROOT/'experiments/archesweather-reproduction/geoarches'
 if subprocess.check_output(['git','-C',str(geo),'rev-parse','HEAD']).decode().strip()!=freeze['geoarches']['head']:raise ValueError('Forecast HEAD changed')
 if hashlib.sha256(subprocess.check_output(['git','-C',str(geo),'diff','--binary','HEAD'])).hexdigest()!=freeze['geoarches']['worktree_patch_sha256']:raise ValueError('Forecast patch changed')
 cfg=OmegaConf.load(freeze['config']['source']);OmegaConf.resolve(cfg)
 forecast=instantiate(cfg.module.module,cfg.module)
 forecast.load_state_dict(torch.load(freeze['checkpoint']['path'],map_location='cpu',weights_only=False)['state_dict']);forecast.cuda().eval()
 data=Era5Forecast(path=str(ROOT/'experiments/archesweather-reproduction/data/era5_240/full'),domain='all',multistep=0,lead_time_hours=6,load_prev=True,norm_scheme='train_stats',norm_stats_path=freeze['normalization']['source'])
 mapping={data.id2pt[i+1][2]:i for i in range(len(data))}
 ds=Data(ROOT,'val')
 da_cfg=json.loads((ROOT/f'outputs/single_da_08_full82_large/seed{a.seed}/da/formal/config.json').read_text())
 models={};provenance={}
 for label,kind in [('DA','da'),('NC','neural_correction')]:
  directory=ROOT/f'outputs/single_da_08_full82_large/seed{a.seed}/{kind}/formal'
  path=directory/'best.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
  result=json.loads((directory/'result.json').read_text())
  if ck['step']!=result['best_step'] or ck['signature']['data']['val']!=ds.signature:raise ValueError('Checkpoint selection/data mismatch')
  if ck['signature']['config_sha256']!=digest(directory/'config.json'):raise ValueError('Checkpoint config mismatch')
  settings=json.loads((directory/'config.json').read_text())
  assert settings['model']['dim']==128 and settings['model']['depth']==4 and settings['model']['heads']==8, 'Expected large full82 model'
  template=CrossAttentionAnalysisModel(channels=82,**settings['model'])
  model=template if kind=='da' else BackgroundCorrection(template)
  model.cuda().eval();model.load_state_dict(ck['model'])
  models[label]=model
  provenance[label]={'sha256':digest(path),'step':ck['step'],'path':str(path)}
 mean=ds.base.mean.cuda();std=ds.base.std.cuda();tstd=float(std[2]);w=torch.cos(torch.deg2rad(torch.linspace(90,-90,121,device='cuda'))).clamp_min(0)[None,:,None].double()
 def flat(native):
  physical=data.denormalize(native)
  return torch.cat([physical['surface'][:,:,0],physical['level'].flatten(1,2)],1).roll(-120,-1)
 def score(field,reference):
  delta=(field-reference).double();return ((delta.square()*w).sum((-2,-1))/(w.sum()*240)).sqrt()[0].cpu().tolist()
 windows=['2021-01-05T00', '2021-01-20T00', '2021-02-05T00', '2021-02-20T00', '2021-03-05T00', '2021-03-20T00', '2021-04-05T00', '2021-04-20T00', '2021-05-05T00', '2021-05-20T00', '2021-06-05T00', '2021-06-20T00', '2021-07-05T00', '2021-07-20T00', '2021-08-05T00', '2021-08-20T00', '2021-09-05T00', '2021-09-20T00', '2021-10-05T00', '2021-10-20T00', '2021-11-05T00', '2021-11-20T00', '2021-12-05T00', '2021-12-20T00']
 if a.pilot:windows=['2021-01-15T00','2021-04-15T00','2021-07-15T00','2021-10-15T00']
 if a.smoke:windows=windows[:1]
 protocol={'seed':a.seed,'windows_initial_utc':windows,'steps':a.steps,'lead_hours':6,'methods':['free','NC','DA','masked_DA'],'history':'previous analysis plus current analysis; initial pair from ERA5 only','checkpoints':provenance,'forecast_sha256':freeze['checkpoint']['sha256'],'channels':CHANNELS,'data_signature':ds.signature,'script_sha256':digest(__file__),'validation_development_only':True,'test_2022_used':False}
 atomic_json(out/'protocol.json',protocol)
 region_masks={name:torch.tensor(mask,device='cuda') for name,mask in regions(np.linspace(90,-90,121),np.arange(240)*1.5).items()}
 region_names=list(region_masks)
 region_weights=torch.stack([(w*mask).reshape(-1) for mask in region_masks.values()])
 region_denominators=region_weights.sum(-1)
 def regional(field,reference):
  err=(field-reference).double().square().flatten(2)[0]
  values=((err @ region_weights.T)/region_denominators).sqrt().T.cpu().tolist()
  return dict(zip(region_names,values))
 cpu_loader=make_loader(CyclingInputs(ROOT,windows,a.steps,freeze['normalization']['source']),workers=2)
 input_iterator=iter(cpu_loader)
 records=[]
 with torch.inference_mode(),(out/'metrics.jsonl').open('w') as stream:
  for window_index,start in enumerate(windows):
   init=np.datetime64(start,'ns');batch=collate_fn([data[mapping[init]]]);batch={k:v.cuda() for k,v in batch.items()}
   histories={name:(batch['prev_state'].clone(),batch['state'].clone()) for name in ['free','NC','DA','masked_DA']}
   for step in range(1,a.steps+1):
    valid=init+np.timedelta64(step*6,'h');current=valid-np.timedelta64(6,'h')
    wait_start=time.perf_counter();payload=next(input_iterator);input_wait=time.perf_counter()-wait_start
    if payload['window_index']!=window_index or payload['step']!=step or np.datetime64(payload['valid_utc'],'ns')!=valid:raise ValueError('Prefetch time/order mismatch')
    reference_native=TensorDict({'surface':payload['reference_surface'].to('cuda:0',non_blocking=True),'level':payload['reference_level'].to('cuda:0',non_blocking=True)},batch_size=[1])
    reference=flat(reference_native)
    sample=payload['sample']
    if int(sample['observation_mask'][0].sum())!=len(sample['context_station_ids'][0]):raise ValueError('Prefetch mask mismatch')
    obs={k:sample[k].to('cuda:0',non_blocking=True) for k in OBSERVATION_KEYS}
    backgrounds={};analyses={}
    for method,(previous,state) in histories.items():
     feed={'prev_state':previous,'state':state,'timestamp':batch['timestamp']+int((step-1)*21600),'lead_time_hours':batch['lead_time_hours']}
     pred=forecast.forward(feed);physical=flat(pred)
     if not torch.isfinite(physical).all():raise ValueError('Nonfinite cycling forecast')
     normalized=(physical-mean)/std;bg=normalized
     local={k:v.clone() for k,v in obs.items()};local['grid_state']=normalized
     if method=='masked_DA':local['observation_mask'].zero_()
     analysis=bg if method=='free' else models['NC' if method=='NC' else 'DA'](bg,local)['analysis']
     if method=='masked_DA' and not torch.equal(analysis,bg):raise ValueError('No observation identity failed')
     updated=physical+(analysis-bg)*std
     if not torch.isfinite(updated).all():raise ValueError('Nonfinite analysis')
     native=pred.clone()
     if not torch.equal(analysis,bg):
      physical_native=data.denormalize(pred)
      physical_native['surface']=updated[:,:4].roll(120,-1).unsqueeze(2)
      physical_native['level']=updated[:,4:].reshape(1,6,13,121,240).roll(120,-1)
      native=data.normalize(physical_native)
      torch.testing.assert_close(flat(native),updated,atol=1e-3,rtol=1e-5)
     else:
      assert torch.equal(native['surface'],pred['surface']) and torch.equal(native['level'],pred['level'])
     histories[method]=(state.clone(),native)
     backgrounds[method]=physical;analyses[method]=updated
     row={'window':start,'step':step,'valid_utc':str(valid),'method':method,'input_wait_seconds':input_wait,'background_rmse_K_or_native_units':score(physical,reference),'analysis_rmse_K_or_native_units':score(updated,reference),'max_t2m_increment_K':float((updated[:,2]-physical[:,2]).abs().max()),'normalized_increment_rms_by_channel':(analysis-bg).double().square().mean((0,2,3)).sqrt().cpu().tolist()}
     row.update(regional_background=regional(physical,reference),regional_analysis=regional(updated,reference))
     row['heldout_by_type']={}
     for variable,channel in [(0,2),(1,0),(2,1),(4,3)]:
      m=sample['heldout_obs_mask'][0] & (sample['heldout_obs_variables'][0]==variable)
      loc=sample['heldout_obs_locations'][0][m].numpy();val=sample['heldout_obs_values'][0][m].numpy().astype(np.float64)
      def station_channel(field):
       sampled=sample_background_at_obs(field[0,channel].cpu().numpy(),np.linspace(90,-90,121),np.arange(240)*1.5,loc[:,0],loc[:,1])
       if not np.isfinite(sampled).all():raise ValueError('Nonfinite multivariate station sample')
       return float(np.square(sampled.astype(np.float64)-val).sum())
      row['heldout_by_type'][str(variable)]={'count':len(val),'background_sse':station_channel(physical),'analysis_sse':station_channel(updated),'reference_sse':station_channel(reference)}
     stream.write(json.dumps(row)+'\n');stream.flush();records.append(row)
    torch.testing.assert_close(backgrounds['free'],backgrounds['masked_DA'],rtol=0,atol=0)
    torch.testing.assert_close(analyses['free'],analyses['masked_DA'],rtol=0,atol=0)
    print('CYCLE',start,step,'DA_T2M',records[-2]['analysis_rmse_K_or_native_units'][2],flush=True)
 try:next(input_iterator);raise AssertionError('Unexpected extra prefetched cycle')
 except StopIteration:pass
 del input_iterator,cpu_loader
 atomic_json(out/'result.json',{'complete':True,'cycles':len(windows)*a.steps,'methods':4,'rows':len(records),'masked_cycle_exact':True,'full82_updates':True,'prefetch_workers':2,'prefetch_factor':2,'immutable_inputs_only':True,'note':'Per-cycle per-channel RMSE in native units; do not average unlike variable units.'})
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
