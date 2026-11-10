"""Frozen old-model ablation, per-variable shuffling and attention bypass."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_full_update_v1 import Data
from offgrid_da.data.multivar_dataset import multivar_collate
from offgrid_da.data.full_state_data import CHANNELS,digest
from offgrid_da.models import CrossAttentionAnalysisModel
from train_single_cycle import pack
from offgrid_da.data.arches_training import atomic_json
from error_balanced import latitude_weights,metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=ROOT/'outputs/single_da_09_full82_loss_balanced';out=root/'frozen_diagnostics'/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
    data=Data(ROOT,'val');stats=json.loads((root/'loss_statistics.json').read_text())
    previous=ROOT/f'outputs/single_da_08_full82_large/seed{a.seed}/da/formal'
    cfg=json.loads((previous/'config.json').read_text());ck=torch.load(previous/'best.pt',map_location='cpu',weights_only=False)
    assert ck['signature']['data']['val']==data.signature
    model=CrossAttentionAnalysisModel(channels=82,**cfg['model']).cuda().eval();model.load_state_dict(ck['model'])
    ix=np.linspace(0,len(data)-1,a.limit,dtype=int).tolist() if a.limit else list(range(len(data)))
    loader=DataLoader(Subset(data,ix),batch_size=1,num_workers=2,multiprocessing_context='spawn',collate_fn=multivar_collate)
    names=['background','real','masked','no_attention','shuffle0','shuffle1','shuffle2']
    sums={k:np.zeros(82) for k in names};den=0.;generators=[torch.Generator().manual_seed(20261102+i) for i in range(3)]
    with torch.inference_mode(),(out/'per_cycle.jsonl').open('w') as stream:
        for i,batch in enumerate(loader):
            x,obs,y=pack(batch,'cuda:0');pred={'background':x,'real':model(x,obs)['analysis']}
            empty={k:v.clone() for k,v in obs.items()};empty['observation_mask'].zero_()
            pred['masked']=model(x,empty)['analysis'];assert torch.equal(pred['masked'],x)
            original=model._local
            try:
                model._local=lambda q,*args:torch.zeros_like(q)
                pred['no_attention']=model(x,obs)['analysis']
            finally:model._local=original
            for j,g in enumerate(generators):
                changed={k:v.clone() for k,v in obs.items()}
                for variable in range(5):
                    ids=(obs['observation_mask'][0]&(obs['obs_variables'][0]==variable)).nonzero().flatten()
                    order=torch.randperm(len(ids),generator=g).to(ids.device)
                    changed['obs_values_normalized'][0,ids]=obs['obs_values_normalized'][0,ids[order]]
                pred[f'shuffle{j}']=model(x,changed)['analysis']
            lat=latitude_weights(x).double();weight=float(lat.expand(len(x),1,*x.shape[-2:]).sum());den+=weight
            row={'index':ix[i],'denominator':weight,'sse_normalized':{}}
            for name,value in pred.items():
                assert torch.isfinite(value).all()
                sse=((value.double()-y.double()).square()*lat).sum((0,2,3)).cpu().numpy()
                sums[name]+=sse;row['sse_normalized'][name]=sse.tolist()
            stream.write(json.dumps(row)+'\n')
            if (i+1)%100==0:print('cycles',i+1,flush=True)
    atomic_json(out/'result.json',{'complete':not bool(a.limit),'cycles':len(ix),'seed':a.seed,'checkpoint_sha256':digest(previous/'best.pt'),
        'checkpoint_step':ck['step'],'test_2022_used':False,'masked_identity_enforced_by_architecture':True,
        'metrics':{k:metrics(v,den,stats['loss_weights'],np.array(stats['physical_std']),CHANNELS) for k,v in sums.items()}})
if __name__=='__main__':main()
