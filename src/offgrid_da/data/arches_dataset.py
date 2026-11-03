"""Single-cycle samples backed by a verified ArchesWeather forecast manifest."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .dataset import SingleCycleDADataset
from .analysis import ERA5Reader
from .metar import MetarReader
from .qc import QCFilters
from .stations import context_and_heldout_sets
from .variables import DEFAULT_REGISTRY
from ..utils.config import load_config
from ..utils.normalization import NormStats

class ArchesBackgroundReader:
    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        self.data = json.loads(self.manifest.read_text())
        self.rows = {}
        for sample in self.data['samples']:
            init = pd.Timestamp(sample['init_utc']).tz_convert('UTC').tz_localize(None)
            valid = pd.Timestamp(sample['valid_utc']).tz_convert('UTC').tz_localize(None)
            if valid - init != pd.Timedelta(hours=6) or valid in self.rows:
                raise ValueError('Invalid or duplicate forecast valid time')
            path = (self.manifest.parent / sample['file']).resolve()
            if path.parent != self.manifest.parent or not path.is_file():
                raise ValueError('Invalid forecast file path')
            self.rows[valid] = (init, path, sample['sha256'])
    def grid(self):
        return {'latitude': np.linspace(90,-90,121), 'longitude':np.arange(240)*1.5}
    def read_background(self, init, lead, variables):
        if list(variables) != ['2m_temperature'] or int(lead) != 6: raise ValueError('Only 6h T2M is supported')
        valid=pd.Timestamp(init)+pd.Timedelta(hours=6)
        expected_init,path,digest=self.rows[valid]
        if pd.Timestamp(init)!=expected_init: raise ValueError('Forecast initialization mismatch')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest: raise ValueError('Forecast hash mismatch')
        with np.load(path,allow_pickle=False) as data:
            np.testing.assert_allclose(data['latitude'],self.grid()['latitude'])
            np.testing.assert_allclose(data['longitude'],self.grid()['longitude'])
            surface=data['surface']
            if surface.shape!=(4,1,121,240) or not np.isfinite(surface).all(): raise ValueError('Invalid forecast surface')
            return np.array(surface[2],dtype=np.float32,copy=True)

class ArchesSingleCycleDADataset(SingleCycleDADataset):
    """Load physical-unit T2M, METAR and ERA5 using a fixed forecast manifest.

    Normalization intentionally disabled until ArchesWeather training statistics exist.
    """
    def __init__(self, manifest, config=None, start=None, end=None, station_split_name='default', normalization_path=None, allow_diagnostic_normalization=False):
        self.cfg=config or load_config(); self.registry=DEFAULT_REGISTRY; self.norm_stats=None
        if normalization_path is not None:
            norm_path=Path(normalization_path); metadata=json.loads(norm_path.with_suffix('.meta.json').read_text())
            if metadata.get('source') != 'ArchesWeather 250k': raise ValueError('wrong normalization source')
            if not metadata.get('complete') and not allow_diagnostic_normalization: raise ValueError('incomplete normalization is diagnostic only')
            if metadata.get('validation_and_test_used') is not False: raise ValueError('normalization leaks validation or test data')
            self.norm_stats=NormStats.load(norm_path)
        time=self.cfg['time']; data=self.cfg['data']; self.variables=list(self.cfg['variables']['target_variables'])
        if self.variables != ['2m_temperature']: raise ValueError('Only T2M is supported')
        self.lead_hours=6; self.analysis_interval_hours=6
        self.obs_window_minutes=int(time['observation_window_minutes']); self.latency_minutes=int(time['analysis_latency_minutes'])
        self.representativeness_std=self.cfg.get('observation_error',{}).get('representativeness_std',{})
        self.pangu=ArchesBackgroundReader(manifest)  # inherited reader interface; contents are ArchesWeather
        self.era5=ERA5Reader(data['era5']['url']); self.metar=MetarReader(data['metar']['url'],data['metar']['layout'],data['cache_dir'])
        rows=[]
        for valid,(init,path,digest) in sorted(self.pangu.rows.items()):
            if start is not None and valid < pd.Timestamp(start): continue
            if end is not None and valid >= pd.Timestamp(end): continue
            if not self.era5.has_time(valid): raise ValueError(f'ERA5 reference unavailable: {valid}')
            rows.append({'valid_time':valid,'init_time':init,'lead_time_hours':6,'observation_window_end':valid,'sample_id':f'arches_{valid:%Y%m%dT%H}'})
        if not rows: raise ValueError('No forecast samples in requested interval')
        self.index=pd.DataFrame(rows); self.station_split_name=station_split_name
        station=self.cfg.get('station_split',{}); all_stations=self._collect_station_names()
        self.context_stations,self.heldout_stations=context_and_heldout_sets(all_stations,split_name=station_split_name,heldout_fraction=float(station.get('heldout_fraction',0.2)),seed=int(station.get('seed',42)))
        self.qc_filters=QCFilters()

    def _normalize_grid(self, field, group):
        # Analysis outputs and all supervised targets share background coordinates.
        return super()._normalize_grid(field, 'background')
