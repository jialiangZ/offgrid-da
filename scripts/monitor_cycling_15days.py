import json,os,time,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from offgrid_da.data.arches_training import atomic_json
out=ROOT/'outputs/cycling_t2m_05_15day';launch=json.loads((out/'launch.json').read_text())
while True:
 states={}
 for seed,e in launch.items():
  if (out/f'formal/seed{seed}/result.json').exists():states[seed]='complete'
  else:
   try:os.kill(e['pid'],0);states[seed]='running'
   except ProcessLookupError:states[seed]='failed'
 atomic_json(out/'status.json',{'stage':'running','seeds':states})
 if 'failed' in states.values():
  atomic_json(out/'status.json',{'stage':'failed','seeds':states});raise RuntimeError(states)
 if all(v=='complete' for v in states.values()):break
 time.sleep(30)
atomic_json(out/'status.json',{'stage':'aggregating'})
try:subprocess.run([sys.executable,str(ROOT/'scripts/summarize_cycling_15days.py')],cwd=ROOT,check=True)
except Exception as error:
 atomic_json(out/'status.json',{'stage':'aggregation_failed','error':repr(error)});raise
