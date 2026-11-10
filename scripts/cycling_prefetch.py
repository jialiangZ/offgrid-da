"""CPU-only ordered reference/observation prefetch; never prefetch recursive states."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

def identity(value):return value

class CyclingInputs(Dataset):
 def __init__(self,root,windows,steps,stats_path):
  self.root=Path(root);self.windows=list(windows);self.steps=steps;self.stats_path=str(stats_path)
  self._weather=None;self._observations=None;self._mapping=None
 def __len__(self):return len(self.windows)*self.steps
 def _initialize(self):
  if self._weather is not None:return
  from geoarches.dataloaders.era5 import Era5Forecast
  from run_full_update_v1 import Data
  torch.set_num_threads(1)
  self._weather=Era5Forecast(path=str(self.root/'experiments/archesweather-reproduction/data/era5_240/full'),domain='all',multistep=0,lead_time_hours=6,load_prev=True,norm_scheme='train_stats',norm_stats_path=self.stats_path)
  self._mapping={self._weather.id2pt[i+1][2]:i for i in range(len(self._weather))}
  self._observations=Data(self.root,'val')
 def __getitem__(self,index):
  self._initialize()
  from geoarches.main_hydra import collate_fn
  from offgrid_da.data.collate import _collate_observation_group
  window_index,offset=divmod(index,self.steps);step=offset+1
  valid=np.datetime64(self.windows[window_index],'ns')+np.timedelta64(step*6,'h')
  reference=collate_fn([self._weather[self._mapping[valid]]])['state']
  obs_index=int((valid-np.datetime64('2021-01-01T00','ns'))/np.timedelta64(6,'h'))
  tokens=self._observations.observation_sample(obs_index)
  assert not (set(tokens['context_station_ids'])&set(tokens['heldout_station_ids']))
  sample={**_collate_observation_group([tokens],'obs'),**_collate_observation_group([tokens],'heldout_obs')}
  sample['context_station_ids']=[tokens['context_station_ids']];sample['heldout_station_ids']=[tokens['heldout_station_ids']]
  return {'window_index':window_index,'step':step,'valid_utc':str(valid),'reference_surface':reference['surface'],'reference_level':reference['level'],'sample':sample}

def make_loader(dataset,workers=2):
 options=dict(batch_size=None,shuffle=False,num_workers=workers,collate_fn=identity,pin_memory=True)
 if workers:options.update(multiprocessing_context='spawn',prefetch_factor=2,persistent_workers=True)
 return DataLoader(dataset,**options)
