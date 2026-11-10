import numpy as np
import pandas as pd
import torch
from offgrid_da.data.metar_multivar import build_records,pack_records,normalization,SPECS
from offgrid_da.data.collate import _collate_observation_group
from offgrid_da.models.observation_encoder import ObservationEncoder


def frame():
    row={'stationName':'TEST','latitude':30.,'longitude':-10.,'elevation':50.,
         'timeObs':pd.Timestamp('2021-01-01T00'),'timeReceived':pd.Timestamp('2021-01-01T00:05'),
         'temperature':280.,'dewpoint':275.,'windSpeed':10.,'windDir':90.,'seaLevelPress':101300.,'altimeter':102000.}
    row.update({v+'DD':'S' for v in ['temperature','dewpoint','windSpeed','windDir','seaLevelPress','altimeter']})
    return pd.DataFrame([row])


def test_five_types_si_units_and_wind_from_convention():
    records,_=build_records(frame(),'2021-01-01')
    values=records.set_index('variable').value
    assert set(values.index)==set(SPECS)
    assert np.isclose(values[1],-10) and abs(values[2])<1e-10
    assert values[4]==101300 and values[3]==275
    assert (records.longitude==350).all()


def test_missing_pressure_does_not_remove_other_variables_or_use_altimeter():
    df=frame();df['seaLevelPress']=np.nan;df['seaLevelPressDD']='Z'
    records,_=build_records(df,'2021-01-01')
    assert set(records.variable)=={0,1,2,3}


def test_calm_with_missing_direction_and_noncalm_variable_direction():
    df=frame();df['windDir']=np.nan;df['windDirDD']='Z'
    records,_=build_records(df,'2021-01-01');assert not records.variable.isin([1,2]).any()
    df['windSpeed']=0.
    records,_=build_records(df,'2021-01-01');assert (records.loc[records.variable.isin([1,2]),'value']==0).all()


def test_time_latency_qc_and_latest_valid_report_per_variable():
    old=frame();old['timeObs']=pd.Timestamp('2020-12-31T23:30')
    new=frame();new['seaLevelPress']=np.nan;new['seaLevelPressDD']='Z'
    bad=frame();bad['temperature']=999.;bad['timeObs']=pd.Timestamp('2020-12-31T23:59')
    late=frame();late['timeReceived']=pd.Timestamp('2021-01-01T00:31');late['stationName']='LATE'
    boundary=frame();boundary['timeObs']=pd.Timestamp('2020-12-31T23');boundary['stationName']='BOUNDARY'
    records,_=build_records(pd.concat([old,new,bad,late,boundary]),'2021-01-01')
    assert len(records)==5 and set(records.stationName)=={'TEST'}
    # Invalid temperature must not reject that report's valid pressure.
    assert records.loc[records.variable==4,'timeObs'].iloc[0]==pd.Timestamp('2020-12-31T23:59')
    assert records.loc[records.variable==0,'value'].iloc[0]==280.


def test_dewpoint_consistency_and_dedup_are_variable_specific():
    df=frame();df['dewpoint']=290.
    records,_=build_records(df,'2021-01-01')
    assert 3 not in set(records.variable) and 0 in set(records.variable)
    df['temperature']=np.nan;df['temperatureDD']='Z'
    records,_=build_records(df,'2021-01-01')
    assert 3 in set(records.variable)


def test_shared_training_scales_station_isolation_and_padding_encoder_backward():
    stats={'input_mean':[2.,3.,280.,100000.],'input_std':[4.,5.,20.,1000.],'train_cycles':4384}
    scales=normalization(stats);assert scales[0]==scales[3]==(280.,20.)
    df=pd.concat([frame().assign(stationName=f'S{i}') for i in range(20)],ignore_index=True)
    records,_=build_records(df,'2021-01-01');assert records.groupby('stationName').role.nunique().eq(1).all()
    a=pack_records(records,'2021-01-01',scales)
    b=pack_records(records.iloc[:0],'2021-01-01',scales)
    assert not set(a['context_station_ids'])&set(a['heldout_station_ids'])
    batch=_collate_observation_group([a,b],'obs')
    assert batch['observation_mask'][0].sum()==len(a['obs_values'])
    assert not batch['observation_mask'][1].any()
    encoder=ObservationEncoder(dim=8,use_innovation=False)
    output=encoder(batch)
    assert torch.count_nonzero(output['tokens'][1])==0
    output['tokens'].square().sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters())
    for variable in SPECS:
        assert encoder.variable.weight.grad[variable].abs().sum()>0
