"""Frozen seasonal five-day cycling pilot; ERA5 used only for initialization/scoring."""
import argparse, json, hashlib, subprocess, sys, os
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
from offgrid_da.models.full_state_t2m import FullStateT2MAnalysis
from offgrid_da.data.arches_training import atomic_json
from single_cycle_controls import FullStateCorrection
from train_single_cycle import pack

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=20);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 out=ROOT/'outputs/cycling_t2m_01_pilot_4win_5d'/('smoke' if a.smoke else 'formal')/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
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
 ds=FullStateInputDataset(ROOT,'val',ROOT/'outputs/single_da_05_input82_ablation','full82')
 da_cfg=json.loads((ROOT/f'outputs/single_da_05_input82_ablation/seed{a.seed}/full82/config.json').read_text())
 models={};provenance={}
 for label,cls,path in [('DA',FullStateT2MAnalysis,ROOT/f'outputs/single_da_05_input82_ablation/seed{a.seed}/full82/best.pt'),('NC',FullStateCorrection,ROOT/f'outputs/single_da_04_t2m_controls_ablation_oi/seed{a.seed}/full82/formal/best.pt')]:
  ck=torch.load(path,map_location='cpu',weights_only=False)
  if ck['signature']['data']['val']!=ds.signature:raise ValueError('DA data changed')
  model=cls(**da_cfg['model']).cuda().eval();model.load_state_dict(ck['model']);models[label]=model
  provenance[label]={'sha256':digest(path),'step':ck['step'],'path':str(path)}
 mean=ds.mean.cuda();std=ds.std.cuda();tstd=float(std[2]);w=torch.cos(torch.deg2rad(torch.linspace(90,-90,121,device='cuda'))).clamp_min(0)[None,:,None].double()
 def flat(native):
  physical=data.denormalize(native)
  return torch.cat([physical['surface'][:,:,0],physical['level'].flatten(1,2)],1).roll(-120,-1)
 def score(field,reference):
  delta=(field-reference).double();return ((delta.square()*w).sum((-2,-1))/(w.sum()*240)).sqrt()[0].cpu().tolist()
 windows=['2021-01-15T00','2021-04-15T00','2021-07-15T00','2021-10-15T00']
 if a.smoke:windows=windows[:1]
 protocol={'seed':a.seed,'windows_initial_utc':windows,'steps':a.steps,'lead_hours':6,'methods':['free','NC','DA','masked_DA'],'history':'previous analysis plus current analysis; initial pair from ERA5 only','checkpoints':provenance,'forecast_sha256':freeze['checkpoint']['sha256'],'channels':CHANNELS,'data_signature':ds.signature,'script_sha256':digest(__file__),'validation_development_only':True,'test_2022_used':False}
 atomic_json(out/'protocol.json',protocol)
 records=[]
 with torch.inference_mode(),(out/'metrics.jsonl').open('w') as stream:
  for start in windows:
   init=np.datetime64(start,'ns');batch=collate_fn([data[mapping[init]]]);batch={k:v.cuda() for k,v in batch.items()}
   histories={name:(batch['prev_state'].clone(),batch['state'].clone()) for name in ['free','NC','DA','masked_DA']}
   for step in range(1,a.steps+1):
    valid=init+np.timedelta64(step*6,'h');current=valid-np.timedelta64(6,'h')
    reference_batch=collate_fn([data[mapping[valid]]]);reference=flat(reference_batch['state'].cuda())
    idx=int((valid-np.datetime64('2021-01-01T00','ns'))/np.timedelta64(6,'h'))
    sample=full_state_collate([ds[idx]]);_,obs,_=pack(sample,'cuda:0')
    backgrounds={};analyses={}
    for method,(previous,state) in histories.items():
     feed={'prev_state':previous,'state':state,'timestamp':batch['timestamp']+int((step-1)*21600),'lead_time_hours':batch['lead_time_hours']}
     pred=forecast.forward(feed);physical=flat(pred)
     if not torch.isfinite(physical).all():raise ValueError('Nonfinite cycling forecast')
     normalized=(physical-mean)/std;bg=normalized[:,2:3]
     local={k:v.clone() for k,v in obs.items()};local['grid_state']=normalized
     if method=='masked_DA':local['observation_mask'].zero_()
     analysis=bg if method=='free' else models['NC' if method=='NC' else 'DA'](bg,local)['analysis']
     if method=='masked_DA' and not torch.equal(analysis,bg):raise ValueError('No observation identity failed')
     updated=physical.clone();updated[:,2:3]=physical[:,2:3]+(analysis-bg)*tstd
     if not torch.isfinite(updated).all():raise ValueError('Nonfinite analysis')
     # Preserve all non-T2M native channels bit-for-bit. Only surface T2M is renormalized.
     native=pred.clone()
     if method not in ['free','masked_DA']:
      physical_native=data.denormalize(pred)
      physical_native['surface']=physical_native['surface'].clone()
      physical_native['surface'][:,2,0]=updated[:,2].roll(120,-1)
      normed=data.normalize(physical_native)
      native['surface']=pred['surface'].clone();native['surface'][:,2]=normed['surface'][:,2]
     assert torch.equal(native['level'],pred['level'])
     for ch in [0,1,3]:assert torch.equal(native['surface'][:,ch],pred['surface'][:,ch])
     assert torch.equal(native['surface'],pred['surface']) if method in ['free','masked_DA'] else True
     histories[method]=(state.clone(),native)
     backgrounds[method]=physical;analyses[method]=updated
     row={'window':start,'step':step,'valid_utc':str(valid),'method':method,'background_rmse_K_or_native_units':score(physical,reference),'analysis_rmse_K_or_native_units':score(updated,reference),'max_t2m_increment_K':float((updated[:,2]-physical[:,2]).abs().max())}
     stream.write(json.dumps(row)+'\n');stream.flush();records.append(row)
    torch.testing.assert_close(backgrounds['free'],backgrounds['masked_DA'],rtol=0,atol=0)
    torch.testing.assert_close(analyses['free'],analyses['masked_DA'],rtol=0,atol=0)
    print('CYCLE',start,step,'DA_T2M',records[-2]['analysis_rmse_K_or_native_units'][2],flush=True)
 atomic_json(out/'result.json',{'complete':True,'cycles':len(windows)*a.steps,'methods':4,'rows':len(records),'masked_cycle_exact':True,'non_T2M_update_identity':True,'note':'Per-cycle per-channel RMSE in native units; do not average unlike variable units.'})
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
