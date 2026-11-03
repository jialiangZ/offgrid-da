from __future__ import annotations
import calendar
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"era5_240"/"full"
OUTPUT=ROOT/"stats"/"era5_norm_stats_1998_2017_lead6h.pt"
HOURS=(0,6,12,18)
CHUNK=16
LEVEL=("geopotential","u_component_of_wind","v_component_of_wind","temperature","specific_humidity","vertical_velocity")
SURFACE=("10m_u_component_of_wind","10m_v_component_of_wind","2m_temperature","mean_sea_level_pressure")
def acc(): return {"sum":None,"sumsq":None,"count":0}
def update(a,x,axes):
    x=np.asarray(x)
    if not np.isfinite(x).all(): raise ValueError("Non-finite training value")
    x=x.astype(np.float64,copy=False)
    s=x.sum(axis=axes,dtype=np.float64); q=np.square(x,dtype=np.float64).sum(axis=axes,dtype=np.float64)
    if a["sum"] is None: a["sum"]=np.zeros_like(s); a["sumsq"]=np.zeros_like(q)
    a["sum"]+=s; a["sumsq"]+=q; a["count"]+=x.shape[0]*x.shape[-2]*x.shape[-1]
def moments(a):
    m=a["sum"]/a["count"]
    return m,np.sqrt(np.maximum(a["sumsq"]/a["count"]-m*m,0))
state={n:acc() for n in LEVEL+SURFACE}
delta={n:acc() for n in LEVEL+SURFACE}
for year in range(1998,2018):
    days=366 if calendar.isleap(year) else 365
    print(f"processing {year}",flush=True)
    with ExitStack() as stack:
        ds={}
        for h in HOURS:
            p=DATA/f"era5_240_{year}_{h}h.nc"
            if not p.is_file() or p.stat().st_size<4580000000: raise FileNotFoundError(f"Missing/incomplete {p}")
            ds[h]=stack.enter_context(xr.open_dataset(p,engine="netcdf4"))
            if ds[h].sizes.get("time")!=days: raise ValueError(f"Unexpected time count: {p}")
        next00=None
        if year<2017:
            p=DATA/f"era5_240_{year+1}_0h.nc"
            if not p.is_file(): raise FileNotFoundError(p)
            next00=stack.enter_context(xr.open_dataset(p,engine="netcdf4"))
        for name in LEVEL+SURFACE:
            axes=(0,2,3) if name in LEVEL else (0,1,2)
            for start in range(0,days,CHUNK):
                stop=min(start+CHUNK,days)
                b={h:ds[h][name].isel(time=slice(start,stop)).to_numpy() for h in HOURS}
                for h in HOURS: update(state[name],b[h],axes)
                for h1,h2 in zip(HOURS[:-1],HOURS[1:]): update(delta[name],b[h2]-b[h1],axes)
                overnight_stop=min(stop,days-1)
                if start<overnight_stop:
                    count=overnight_stop-start
                    nxt=ds[0][name].isel(time=slice(start+1,start+count+1)).to_numpy()
                    update(delta[name],nxt-b[18][:count],axes)
            # Include 18Z->00Z across year boundaries only through 2016->2017.
            if next00 is not None:
                prev=ds[18][name].isel(time=slice(days-1,days)).to_numpy()
                nxt=next00[name].isel(time=slice(0,1)).to_numpy()
                update(delta[name],nxt-prev,axes)
    print(f" finished {year}",flush=True)
lm=[]; ls=[]; ld=[]
for n in LEVEL:
    m,s=moments(state[n]); _,d=moments(delta[n]); lm.append(m); ls.append(s); ld.append(d)
sm=[]; ss=[]; sd=[]
for n in SURFACE:
    m,s=moments(state[n]); _,d=moments(delta[n]); sm.append(m); ss.append(s); sd.append(d)
stats={
"surface_mean":torch.tensor(np.asarray(sm),dtype=torch.float32).reshape(4,1,1,1),
"surface_std":torch.tensor(np.asarray(ss),dtype=torch.float32).reshape(4,1,1,1),
"level_mean":torch.tensor(np.asarray(lm),dtype=torch.float32).reshape(6,13,1,1),
"level_std":torch.tensor(np.asarray(ls),dtype=torch.float32).reshape(6,13,1,1),
"surface_delta_std":torch.tensor(np.asarray(sd),dtype=torch.float32).reshape(4,1,1,1),
"level_delta_std":torch.tensor(np.asarray(ld),dtype=torch.float32).reshape(6,13,1,1),
"metadata":{"years":[1998,2017],"sampling":"6-hourly","lead_time_hours":6,"grid":[121,240],"reduction":"unweighted time-lat-lon population moments","delta_pairs":"adjacent 6-hour states; endpoints within training split"}
}
for k in ("surface_std","level_std","surface_delta_std","level_delta_std"):
    if not torch.isfinite(stats[k]).all() or (stats[k]<=0).any(): raise ValueError(f"Invalid {k}")
OUTPUT.parent.mkdir(parents=True,exist_ok=True); torch.save(stats,OUTPUT)
print("saved",OUTPUT)
for k in ("surface_mean","surface_std","surface_delta_std","level_std","level_delta_std"): print(k,tuple(stats[k].shape),stats[k].flatten()[:6].tolist())
