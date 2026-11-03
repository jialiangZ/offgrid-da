"""Compute ArchesWeather T2M statistics for the configured training period."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root/'src'))
from offgrid_da.data.arches_dataset import ArchesBackgroundReader
from offgrid_da.data.analysis import ERA5Reader
from offgrid_da.utils.normalization import NormStats
from offgrid_da.utils.config import load_config
p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--allow-partial-diagnostic',action='store_true'); a=p.parse_args()
cfg=load_config(); train=cfg['splits']['train']; start=pd.Timestamp(train['start']); end=pd.Timestamp(train['end']); reader=ArchesBackgroundReader(a.manifest)
times=sorted(t for t in reader.rows if start<=t<end); expected=pd.date_range(start,end,freq='6h',inclusive='left'); missing=expected.difference(pd.DatetimeIndex(times))
if missing.size and not a.allow_partial_diagnostic: raise RuntimeError(f'incomplete training data: {len(times)}/{len(expected)}; missing {len(missing)}')
if not times: raise RuntimeError('no training samples')
era=ERA5Reader(cfg['data']['era5']['url']); stats=NormStats(); var='2m_temperature'
for i,t in enumerate(times):
 init,_,_=reader.rows[t]; bg=reader.read_background(init,6,[var]); ref=era.read_analysis(t,[var]); stats.update('background',var,bg[0]); stats.update('reference',var,ref[0])
 if (i+1)%100==0: print(f'{i+1}/{len(times)}',flush=True)
a.output.parent.mkdir(parents=True,exist_ok=True); stats.save(a.output)
meta={'source':'ArchesWeather 250k','forecast_manifest':str(a.manifest.resolve()),'forecast_manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'train_start':str(start),'train_end_exclusive':str(end),'cycles':len(times),'expected_cycles':len(expected),'missing_cycles':len(missing),'complete':len(missing)==0,'diagnostic_only':bool(missing.size),'aggregation':'unweighted finite grid points, sample standard deviation','validation_and_test_used':False}
a.output.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps({'metadata':meta,'stats':stats.to_dict()},indent=2))
