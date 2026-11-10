import os,sys,json,subprocess,hashlib,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from offgrid_da.data.arches_training import atomic_json
out=ROOT/'outputs/cycling_t2m_04_24win_eval';out.mkdir(exist_ok=True)
lock=(out/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
smoke=json.loads((out/'smoke/seed42/result.json').read_text());assert smoke['complete']
source=ROOT/'scripts/run_cycling_evaluation.py'
atomic_json(out/'evaluation_protocol.json',{'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'seeds':[42,43],'GPUs':[0,3],'windows':'same fixed 24 windows as cycling_24windows_v1','purpose':'Replay unchanged trajectories to collect region/heldout sufficient statistics','regions':'global, tropics, northern/southern extratropics, Europe/NorthAmerica/EastAsia rectangles','phases':['background','analysis'],'all_channels':82,'aggregation':'window, initialization month, lead time, overall; station SSE pooled by report count','station_split':'existing context/heldout hash split; heldout excluded from inputs','test_2022_used':False})
(out/'source_snapshot.py').write_bytes(source.read_bytes())
children=[]
for seed,gpu in [(42,0),(43,3)]:
 log=(out/f'seed{seed}.log').open('ab',buffering=0)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='src:scripts:experiments/archesweather-reproduction/geoarches')
 p=subprocess.Popen([str(ROOT/'experiments/archesweather-reproduction/geoarches/.venv/bin/python'),'-u',str(source),'--seed',str(seed)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 children.append((seed,p,log))
atomic_json(out/'status.json',{'stage':'running','children':{str(s):p.pid for s,p,l in children}})
failed=[]
for seed,p,log in children:
 code=p.wait();log.close()
 if code:failed.append([seed,code])
if failed:
 atomic_json(out/'status.json',{'stage':'failed','errors':failed});raise RuntimeError(failed)
atomic_json(out/'status.json',{'stage':'aggregating'})
try:subprocess.run([sys.executable,str(ROOT/'scripts/summarize_cycling_evaluation.py')],cwd=ROOT,check=True)
except Exception as error:
 atomic_json(out/'status.json',{'stage':'aggregation_failed','error':repr(error)});raise
