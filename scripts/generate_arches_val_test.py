"""Generate and validate 2021/2022 single-step backgrounds on GPU 3."""
import json,os,subprocess
from pathlib import Path
import numpy as np
root=Path('/home/jlzhang/Documents/offgrid-da'); geo=root/'experiments/archesweather-reproduction/geoarches'; env=os.environ.copy(); env['PYTHONPATH']=str(geo)
for year,label in [(2021,'val'),(2022,'test')]:
 out=root/f'outputs/arches_backgrounds_{label}_{year}'; out.mkdir(parents=True,exist_ok=True)
 command=[str(geo/'.venv/bin/python'),'-u',str(root/'scripts/generate_arches_backgrounds.py'),'--start',f'{year}-01-01T00','--end',f'{year+1}-01-01T00','--output',str(out),'--device','cuda:3']
 print('START',year,flush=True)
 with (out/'generation.log').open('ab',buffering=0) as log:
  subprocess.run(command,cwd=geo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 m=json.loads((out/'manifest.json').read_text()); actual=np.array([np.datetime64(s['valid_utc'].removesuffix('Z'),'s') for s in m['samples']]); expected=np.arange(np.datetime64(f'{year}-01-01T00','s'),np.datetime64(f'{year+1}-01-01T00','s'),np.timedelta64(6,'h')); np.testing.assert_array_equal(np.sort(actual),expected)
 assert len(actual)==1460
 (out/'coverage.json').write_text(json.dumps({'year':year,'split':label,'expected':1460,'actual':len(actual),'coverage_complete':True},indent=2)+'\n'); print('COMPLETE',year,len(actual),flush=True)
print('ALL COMPLETE',flush=True)
