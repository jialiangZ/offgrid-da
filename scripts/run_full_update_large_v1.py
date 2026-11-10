from run_full_update_v1 import *
import run_full_update_v1 as core
def pipeline():
 out=ROOT/'outputs/single_da_08_full82_large';out.mkdir(exist_ok=True)
 lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 datasets={s:Data(ROOT,s) for s in ['train','val']}
 prep=json.loads((ROOT/'outputs/single_da_07_full82_small_stopped/preparation.json').read_text())
 assert prep['complete'] and prep['signatures']=={s:d.signature for s,d in datasets.items()}
 for d in datasets.values():d[0]
 atomic_json(out/'preparation.json',prep)
 atomic_json(out/'protocol.json',{'model':{'dim':128,'depth':4,'heads':8},'seeds':[42,43],'output_channels':82,'reference_audit_reused':'outputs/single_da_07_full82_small_stopped/preparation.json','test_2022_used':False})
 cfg=json.loads((ROOT/'outputs/single_da_05_input82_ablation/config.json').read_text());cfg['model'].pop('input_channels');cfg['model']['max_neighbors']=80
 cfg['model'].update(dim=128,depth=4,heads=8,query_chunk=256);cfg['loader_workers']=0
 cfg.update(seeds=[42,43],metric='dimensionless_group_balanced_normalized_RMSE',output_normalization='outputs/single_da_05_input82_ablation/normalization.json',arm_gpus={'da':0,'neural_correction':1})
 cfg.pop('input_modes',None)
 for seed in [42,43]:
  for trial in [True,False]:
   children=[]
   for gpu,kind in [(0,'da'),(1,'neural_correction')]:
    while True:
     info=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().split(',')
     if int(info[0])<1024 and int(info[1])<10:break
     time.sleep(30)
    target=out/f'seed{seed}'/kind/('trial' if trial else 'formal');target.mkdir(parents=True,exist_ok=True)
    if (target/'result.json').exists():continue
    c=dict(cfg,seed=seed,name=f'full_update_large_{seed}_{kind}',selection_metric='normalized group-balanced RMSE',output_channels=82)
    atomic_json(target/'config.json',c)
    command=[sys.executable,'-u',str(ROOT/'scripts/train_full_update_prefetch.py'),'--train','--config',str(target/'config.json'),'--preparation',str(out/'preparation.json'),'--output',str(target),'--kind',kind]+(['--trial'] if trial else [])
    if (target/'last.pt').exists():command+=['--resume',str(target/'last.pt')]
    log=(target/'run.log').open('ab');p=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
    children.append((p,log))
   atomic_json(out/'status.json',{'stage':'trial' if trial else 'formal','seed':seed,'pids':[p.pid for p,_ in children]})
   for p,log in children:
    code=p.wait();log.close()
    if code:raise RuntimeError('Child failed; see run.log')
 atomic_json(out/'status.json',{'stage':'complete'})

if __name__=='__main__':
 if '--train' in sys.argv:sys.argv.remove('--train');core.run_train()
 else:pipeline()
