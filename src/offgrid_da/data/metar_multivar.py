"""Five scalar METAR observation types, without changing frozen experiments."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from .qc import QCFilters, apply_time_window, apply_received_cutoff, apply_location_qc, apply_variable_qc, dedup_reports
from .stations import station_role
from .variables import DD_QUALITY_CODE

# Preserve existing variable IDs 0,1,2. Pressure is MSLP, never altimeter.
SPECS = {
    0: {'name':'2m_temperature','unit':'K','background_channel':2,'instrument':1.,'representativeness':2.5},
    1: {'name':'10m_u_component_of_wind','unit':'m/s','background_channel':0,'instrument':1.5,'representativeness':3.},
    2: {'name':'10m_v_component_of_wind','unit':'m/s','background_channel':1,'instrument':1.5,'representativeness':3.},
    3: {'name':'2m_dewpoint_temperature','unit':'K','background_channel':2,'instrument':1.,'representativeness':2.5},
    4: {'name':'mean_sea_level_pressure','unit':'Pa','background_channel':3,'instrument':100.,'representativeness':150.},
}
POLICY = {'window_minutes':60,'latency_minutes':30,'station_split_seed':42,'heldout_fraction':.2,
          'temperature_range_K':[180.,340.],'dewpoint_range_K':[150.,340.],
          'wind_speed_range_ms':[0.,100.],'pressure_range_Pa':[80000.,110000.],
          'dewpoint_temperature_tolerance_K':1.,'altimeter_assimilated':False,
          'humidity_type':'dewpoint only; no RH or q duplication',
          'error_policy':'Initial instrument/representativeness assumptions, not calibrated errors',
          'dewpoint_normalization':'Use training-background T2M mean/std as a shared temperature coordinate, not measured dewpoint moments'}


def signature():
    sources=[Path(__file__),Path(__file__).with_name('qc.py'),Path(__file__).with_name('stations.py'),
             Path(__file__).with_name('variables.py'),Path(__file__).with_name('metar.py')]
    return hashlib.sha256(json.dumps({'specs':SPECS,'policy':POLICY,
        'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},sort_keys=True).encode()).hexdigest()


def build_records(raw, valid_time):
    """QC each variable independently, derive paired u/v, deduplicate per station/type.

    A missing pressure does not remove a valid temperature. Calm wind has u=v=0
    even when direction is missing; non-calm missing/variable direction is rejected.
    """
    valid_time=pd.Timestamp(valid_time)
    filters=QCFilters()
    df=raw.reset_index(drop=True).copy()
    df=apply_time_window(df,valid_time-pd.Timedelta(minutes=60),valid_time)
    df=apply_received_cutoff(df,valid_time+pd.Timedelta(minutes=30))
    df=apply_location_qc(df,filters)
    good_name=df.stationName.notna() & ~df.stationName.astype(str).str.strip().isin(['','nan','None'])
    df=df.loc[good_name].copy()
    frames=[];counts={}
    def append(frame, variable, value, quality):
        part=frame[['stationName','latitude','longitude','elevation','timeObs','timeReceived']].copy()
        part['variable']=variable;part['value']=value;part['quality']=quality
        part=dedup_reports(part,SPECS[variable]['name'],filters)
        counts[str(variable)]=len(part);frames.append(part)
    for variable,column,limits in [(0,'temperature',POLICY['temperature_range_K']),
                                   (3,'dewpoint',POLICY['dewpoint_range_K']),
                                   (4,'seaLevelPress',POLICY['pressure_range_Pa'])]:
        frame=apply_variable_qc(df,column,column+'DD',limits,filters)
        if variable==3:
            temp=pd.to_numeric(frame.temperature,errors='coerce')
            temperature_valid=np.isfinite(temp) & temp.between(*POLICY['temperature_range_K']) & frame.temperatureDD.isin(filters.allow_dd_flags)
            frame=frame.loc[~temperature_valid | (frame.dewpoint<=temp+POLICY['dewpoint_temperature_tolerance_K'])]
        append(frame,variable,frame[column].to_numpy(dtype=float),frame[column+'DD'].astype(str).str.strip().map(DD_QUALITY_CODE).to_numpy())
    wind=apply_variable_qc(df,'windSpeed','windSpeedDD',POLICY['wind_speed_range_ms'],filters).copy()
    direction=pd.to_numeric(wind.windDir,errors='coerce')
    direction_good=np.isfinite(direction) & direction.between(0,360) & wind.windDirDD.astype(str).str.strip().isin(filters.allow_dd_flags)
    wind=wind.loc[(wind.windSpeed==0)|direction_good].copy()
    angle=np.deg2rad(pd.to_numeric(wind.windDir,errors='coerce').fillna(0).to_numpy())
    speed=wind.windSpeed.to_numpy(dtype=float)
    qspeed=wind.windSpeedDD.astype(str).str.strip().map(DD_QUALITY_CODE).to_numpy()
    qdir=wind.windDirDD.astype(str).str.strip().map(DD_QUALITY_CODE).fillna(0).to_numpy()
    quality=np.where(speed==0,qspeed,np.minimum(qspeed,qdir))
    append(wind,1,-speed*np.sin(angle),quality)
    append(wind,2,-speed*np.cos(angle),quality)
    result=pd.concat(frames,ignore_index=True)
    result['longitude']=result.longitude%360
    result['role']=[station_role(s,'default',POLICY['heldout_fraction'],POLICY['station_split_seed']) for s in result.stationName]
    result=result.sort_values(['variable','stationName'],kind='stable').reset_index(drop=True)
    if len(result) and (not np.isfinite(result.value).all() or result.duplicated(['stationName','variable']).any()):
        raise ValueError('Invalid multivariate records')
    return result,{'raw_reports':len(raw),'after_location_time_received':len(df),'tokens_by_variable':counts,
                   'context_tokens':int((result.role=='context').sum()),'heldout_tokens':int((result.role=='heldout').sum())}


def normalization(full_state_statistics):
    # Four observed quantities map to train background scales; Td uses T's scale.
    mean=np.array(full_state_statistics['input_mean'],dtype=float)
    std=np.array(full_state_statistics['input_std'],dtype=float)
    if full_state_statistics['train_cycles']!=4384:
        raise ValueError('Complete 2018-2020 training normalization required')
    result={v:(float(mean[s['background_channel']]),float(std[s['background_channel']])) for v,s in SPECS.items()}
    if any(not np.isfinite(m) or not np.isfinite(s) or s<=0 for m,s in result.values()):
        raise ValueError('Invalid observation scales')
    return result


def pack_records(records, valid_time, scales):
    result={}
    for role,prefix,role_id in [('context','obs',0),('heldout','heldout_obs',1)]:
        frame=records.loc[records.role==role]
        ids=frame.variable.to_numpy(dtype=np.int64)
        values=frame.value.to_numpy(dtype=np.float32)
        means=np.array([scales[int(v)][0] for v in ids],dtype=np.float32)
        std=np.array([scales[int(v)][1] for v in ids],dtype=np.float32)
        instrument=np.array([SPECS[int(v)]['instrument'] for v in ids],dtype=np.float32)
        representation=np.array([SPECS[int(v)]['representativeness'] for v in ids],dtype=np.float32)
        error=np.hypot(instrument,representation)
        arrays={'values':values,'values_normalized':(values-means)/std,
            'variables':ids,'locations':frame[['latitude','longitude','elevation']].to_numpy(dtype=np.float32),
            'time_offsets':((pd.to_datetime(frame.timeObs)-pd.Timestamp(valid_time)).dt.total_seconds()/60).to_numpy(dtype=np.float32),
            'sources':np.zeros(len(frame),dtype=np.int64),'quality':frame.quality.to_numpy(dtype=np.int64),
            'error':error,'error_normalized':error/std,'error_instrument':instrument,
            'error_representativeness':representation,'station_role':np.full(len(frame),role_id,dtype=np.int64)}
        for name,array in arrays.items():result[prefix+'_'+name]=torch.from_numpy(array.copy())
        result['context_station_ids' if role=='context' else 'heldout_station_ids']=frame.stationName.astype(str).tolist()
    if set(result['context_station_ids']) & set(result['heldout_station_ids']):
        raise ValueError('Station leakage')
    return result
