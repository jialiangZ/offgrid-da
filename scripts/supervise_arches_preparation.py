import json,os,pathlib,subprocess,sys,time
root=pathlib.Path('/home/jlzhang/Documents/offgrid-da'); out=root/'outputs/bg_t2m_train_2018_2020'; pid=int((out/'generation.pid').read_text()); print('waiting for generator',pid,flush=True)
while True:
 try:
  text=(pathlib.Path('/proc')/str(pid)/'stat').read_text(); status=text.split(') ',1)[1][0]
  if status=='Z': break
 except FileNotFoundError: break
 time.sleep(30)
manifest=out/'manifest.json'
if not manifest.exists() or len(json.loads(manifest.read_text())['samples'])!=4384:
 print('generation incomplete; statistics and validation not run',flush=True); sys.exit(1)
python=str(root/'.venv/bin/python'); norm=root/'outputs/normalization/arches_train_2018_2020.json'
for cmd in [[python,str(root/'scripts/prepare_arches_norms.py'),'--manifest',str(manifest),'--output',str(norm)],[python,str(root/'scripts/validate_arches_m2_formal_norm.py')]]:
 print('running',cmd,flush=True); subprocess.run(cmd,cwd=root,check=True)
print('ALL COMPLETE',flush=True)
