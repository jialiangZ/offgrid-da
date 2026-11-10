"""Prepare/audit the opt-in five-type METAR token cache on train and val only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
import multiprocessing
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from offgrid_da.data.arches_training import atomic_json
from offgrid_da.data.multivar_dataset import MultivarMetarDataset
from offgrid_da.data.metar_multivar import SPECS,POLICY

DATA={}
def initialize():
    global DATA
    DATA={s:MultivarMetarDataset(ROOT,s) for s in ['train','val']}

def process(item):
    split,index=item
    sample=DATA[split].observation_sample(index)
    summary={'split':split,'index':index,'valid_time':str(sample['valid_time']),
             'qc':sample['qc_report'],'counts':{}}
    for prefix in ['obs','heldout_obs']:
        summary['counts'][prefix]={str(v):int((sample[prefix+'_variables']==v).sum()) for v in SPECS}
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--output',type=Path,default=ROOT/'outputs/single_da_06_obs_metar5_prep')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(a.output/'prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    initialize()
    provenance={s:{'signature':d.signature,'token_signature':d.token_signature,'cache_dir':str(d.cache)} for s,d in DATA.items()}
    atomic_json(a.output/'protocol.json',{'variables':SPECS,'policy':POLICY,'provenance':provenance,
        'test_2022_used':False,'target_scope':'This stage replaces observations; 82-output analysis not yet implemented.'})
    jobs=[(s,i) for s,n in [('train',4384),('val',1460)] for i in range(min(n,a.limit) if a.limit else n)]
    counts={s:{p:{str(v):0 for v in SPECS} for p in ['obs','heldout_obs']} for s in ['train','val']}
    progress={'stage':'running','done':0,'expected':len(jobs),'complete':False,'started_unix':time.time()}
    atomic_json(a.output/'status.json',progress)
    try:
        with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,mp_context=multiprocessing.get_context('spawn')) as pool,(a.output/'per_cycle.jsonl').open('w') as stream:
            for result in pool.map(process,jobs,chunksize=4):
                stream.write(json.dumps(result)+'\n')
                for prefix,values in result['counts'].items():
                    for variable,count in values.items():counts[result['split']][prefix][variable]+=count
                progress['done']+=1
                if progress['done']%20==0:
                    atomic_json(a.output/'status.json',progress);print(progress['done'],'/',len(jobs),flush=True)
        progress.update(stage='complete' if not a.limit else 'diagnostic_complete',complete=not bool(a.limit),finished_unix=time.time())
        atomic_json(a.output/'summary.json',{'complete':progress['complete'],'cycles':progress['done'],'counts':counts,'provenance':provenance,'test_2022_used':False})
    except Exception as error:
        progress.update(stage='failed',error=repr(error));raise
    finally:atomic_json(a.output/'status.json',progress)

if __name__=='__main__':main()
