"""Reuse frozen observation QC/packing; no future reference fields enter models."""
import numpy as np
import pandas as pd
import torch
from offgrid_da.data.arches_training import LazyArchesDataset
from offgrid_da.data.observations import metar_to_observation_set
from offgrid_da.data.collate import single_cycle_collate
TEMPLATE=None

def confirmation_sample(full_dataset, valid):
 global TEMPLATE
 if TEMPLATE is None:TEMPLATE=LazyArchesDataset(__import__('pathlib').Path(__file__).resolve().parents[1],'val')
 t=TEMPLATE;valid=pd.Timestamp(valid)
 assert valid.year==2023
 observations=metar_to_observation_set(t.metar,t.registry,t.variables,window_start=valid-pd.Timedelta(minutes=t.obs_window_minutes),window_end=valid,received_cutoff=valid+pd.Timedelta(minutes=t.latency_minutes),filters=t.qc_filters)
 context=[r for r in observations.records if r.station_id in t.context_stations]
 heldout=[r for r in observations.records if r.station_id in t.heldout_stations]
 assert not (set(r.station_id for r in context)&set(r.station_id for r in heldout))
 sample={'background':torch.zeros(1,121,240),'reference':torch.zeros(1,121,240),'background_normalized':torch.zeros(1,121,240),'reference_normalized':torch.zeros(1,121,240),'valid_time':valid,'sample_id':str(valid),'grid_latitude':torch.linspace(90,-90,121),'grid_longitude':torch.arange(240)*1.5,'context_station_ids':[r.station_id for r in context],'heldout_station_ids':[r.station_id for r in heldout]}
 for prefix,records in [('obs',context),('heldout_obs',heldout)]:
  packed=t._pack_observations(records,valid)
  for name,values in packed.items():sample[prefix+'_'+name]=torch.from_numpy(values)
  sample[prefix+'_values_normalized']=torch.from_numpy(t._normalize_observations(packed))
 return single_cycle_collate([sample])
