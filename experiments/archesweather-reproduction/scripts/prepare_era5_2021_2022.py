#!/usr/bin/env python3
"""Prepare validated ERA5 shards for 2021–2022 forecast inputs."""
import concurrent.futures,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np,xarray as xr
ROOT=Path('/home/jlzhang/Documents/offgrid-da/experiments/archesweather-reproduction')
OUT=ROOT/'data/era5_240/full'; STAGE=ROOT/'data/era5_240/staging_2021_2022'; REPORT=ROOT/'data/era5_240/prepare_2021_2022_report.json'
SRC={2021:'gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr',2022:'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'}
VARS=['10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature','mean_sea_level_pressure','geopotential','u_component_of_wind','v_component_of_wind','temperature','specific_humidity','vertical_velocity']
LEVELS=[50,100,150,200,250,300,400,500,600,700,850,925,1000]
def validate(path,year,hour):
 with xr.open_dataset(path) as ds:
  missing=set(VARS)-set(ds.data_vars)
  if missing: raise ValueError(f'{path}: missing {missing}')
  if dict(ds.sizes)!={'time':365,'longitude':240,'latitude':121,'level':13}: raise ValueError(f'{path}: dimensions {dict(ds.sizes)}')
  expected=np.arange(np.datetime64(f'{year}-01-01T{hour:02d}','h'),np.datetime64(f'{year+1}-01-01T{hour:02d}','h'),np.timedelta64(24,'h')).astype('datetime64[ns]')
  np.testing.assert_array_equal(ds.time.values,expected); np.testing.assert_allclose(ds.longitude.values,np.arange(240)*1.5); np.testing.assert_allclose(np.sort(ds.latitude.values),np.linspace(-90,90,121)); np.testing.assert_array_equal(ds.level.values,LEVELS)
  for name in VARS:
   a=ds[name].isel(time=[0,-1]).values
   if not np.isfinite(a).all(): raise ValueError(f'{path}: nonfinite {name} at boundary')
 return path.stat().st_size
def job(year,hour):
 target=OUT/f'era5_240_{year}_{hour}h.nc'
 if target.exists(): return {'year':year,'hour':hour,'status':'existing_verified','bytes':validate(target,year,hour),'source':SRC[year]}
 temp=STAGE/f'era5_240_{year}_{hour}h.{os.getpid()}.part.nc'
 print('starting',year,hour,flush=True)
 ds=xr.open_zarr(SRC[year],storage_options={'token':'anon'},consolidated=True)
 sub=ds[VARS].sel(time=slice(f'{year}-01-01',f'{year}-12-31T23:59:59'))
 sub=sub.sel(time=sub.time.dt.hour==hour); sub.to_netcdf(temp); ds.close(); size=validate(temp,year,hour)
 if target.exists(): raise FileExistsError(target)
 os.rename(temp,target); print('completed',year,hour,size,flush=True)
 return {'year':year,'hour':hour,'status':'downloaded_verified','bytes':size,'source':SRC[year]}
def main():
 STAGE.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
 for hour in (12,18):
  with xr.open_dataset(OUT/f'era5_240_2020_{hour}h.nc') as ds:
   t=np.datetime64(f'2020-12-31T{hour:02d}:00:00','ns')
   if t not in ds.time.values: raise RuntimeError(f'missing boundary {t}')
 print('2020 boundary 12Z and 18Z present',flush=True); results=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(job,y,h) for y in (2021,2022) for h in (0,6,12,18)]
  for f in concurrent.futures.as_completed(futures): results.append(f.result())
 report={'purpose':'forecast inputs for 2021 and 2022','variables':VARS,'grid':'WB2 conservative 1.5 degrees','boundary_2020_12_18_verified':True,'shards':sorted(results,key=lambda x:(x['year'],x['hour'])),'notes':'2021 uses training archive; 2022 uses extended archive. Only ten forecast variables saved. Checks include all timestamps, grids, levels and finite first/last fields.'}
 temp=REPORT.with_suffix('.tmp'); temp.write_text(json.dumps(report,indent=2)+'\n'); os.replace(temp,REPORT); print('ALL VERIFIED',REPORT,flush=True)
if __name__=='__main__': main()
