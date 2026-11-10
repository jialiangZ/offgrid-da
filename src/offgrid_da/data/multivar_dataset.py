"""Opt-in METAR multivariate view of existing single-cycle full-state inputs."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import pandas as pd
import torch
from .full_state_data import FullStateInputDataset,full_state_collate,digest
from .metar import MetarReader,metar_hour_url
from .metar_multivar import build_records,pack_records,normalization,signature,POLICY,SPECS
from ..utils.io import gs_to_https


class MultivarMetarDataset(torch.utils.data.Dataset):
    """Keep existing forecast/target contract; replace T-only tokens with five types.

    This adapter still has a T2M target. It is the observation interface for a
    future 82-output trainer, not an implementation of full-state updates.
    """
    def __init__(self,root,split,experiment=None,input_mode='full82',observation_cache=None):
        self.root=Path(root)
        self.base=FullStateInputDataset(root,split,experiment or self.root/'outputs/single_da_05_input82_ablation',input_mode)
        self.norm_stats=self.base.norm_stats
        self.context_stations=self.base.context_stations;self.heldout_stations=self.base.heldout_stations
        self.split=split
        stats_path=self.base.experiment/'normalization.json'
        self.scales=normalization(json.loads(stats_path.read_text()))
        self.reader=MetarReader('gs://weatherbench2/datasets/metar','metar-timeObs-by-hour_2001-07-01T00_2024-01-01T00',str(self.root/'outputs/cache'))
        self.token_signature=hashlib.sha256(json.dumps({'transform':signature(),'adapter':digest(__file__),
            'normalization_sha256':digest(stats_path),'source':[self.reader.base_url,self.reader.layout]},sort_keys=True).encode()).hexdigest()
        self.signature=hashlib.sha256((self.base.signature+self.token_signature).encode()).hexdigest()
        self.cache=Path(observation_cache or self.root/'outputs/metar_multivar_cache')/self.token_signature
        self.cache.mkdir(parents=True,exist_ok=True)

    def __len__(self):return len(self.base)

    def valid_time(self,index):
        return pd.Timestamp(self.base.records[index]['valid_utc']).tz_convert('UTC').tz_localize(None)

    def observation_sample(self,index):
        valid=self.valid_time(index);path=self.cache/(valid.strftime('%Y%m%dT%H')+'.pt')
        stamps=pd.date_range((valid-pd.Timedelta(minutes=105)).floor('h'),valid.floor('h'),freq='h')
        def source_files():
            return [self.root/'outputs/cache'/hashlib.sha1(gs_to_https(metar_hour_url(self.reader.base_url,self.reader.layout,t)).encode()).hexdigest() for t in stamps]
        sources=source_files()
        if path.exists():
            sample=torch.load(path,map_location='cpu',weights_only=False)
            if sample['token_signature']!=self.token_signature or pd.Timestamp(sample['valid_time'])!=valid:
                raise ValueError('Token provenance mismatch')
            if len(sample['raw_sha256'])!=len(sources) or not all(p.exists() and digest(p)==expected for p,expected in zip(sources,sample['raw_sha256'])):
                raise ValueError('Raw observation source changed')
            return sample
        raw=self.reader.read_window(valid-pd.Timedelta(minutes=60),valid)
        records,report=build_records(raw,valid)
        sample=pack_records(records,valid,self.scales)
        sample.update(qc_report=report,valid_time=valid,token_signature=self.token_signature,
                      raw_sha256=[digest(p) for p in sources])
        for prefix in ['obs','heldout_obs']:
            for suffix in ['values','values_normalized','locations','time_offsets','error','error_normalized']:
                if not torch.isfinite(sample[prefix+'_'+suffix]).all():raise ValueError('Nonfinite observation')
        fd,name=tempfile.mkstemp(prefix=path.name,dir=self.cache)
        try:
            with os.fdopen(fd,'wb') as stream:torch.save(sample,stream)
            os.replace(name,path)
        finally:
            if os.path.exists(name):os.unlink(name)
        return sample

    def __getitem__(self,index):
        sample=self.base[index]
        # Remove old observation-derived fields before merging new aligned tokens.
        for key in list(sample):
            if key.startswith(('obs_','heldout_obs_')) or key in ['context_station_ids','heldout_station_ids','qc_report']:
                del sample[key]
        sample.update(self.observation_sample(index))
        sample['multivar_signature']=self.signature
        return sample


multivar_collate=full_state_collate
