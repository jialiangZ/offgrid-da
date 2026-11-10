"""Fit loss scaling on ALL 4384 training backgrounds, never validation or test."""
import json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_full_update_v1 import Data
from offgrid_da.data.multivar_dataset import multivar_collate
from offgrid_da.data.full_state_data import CHANNELS,digest
from offgrid_da.data.arches_training import atomic_json
from error_balanced import fit_weights,latitude_weights,metrics

def main():
    torch.set_num_threads(4)
    out=ROOT/'outputs/single_da_09_full82_loss_balanced';out.mkdir(exist_ok=True)
    data=Data(ROOT,'train');sse=torch.zeros(82,dtype=torch.float64);den=0.;count=0
    loader=DataLoader(data,batch_size=1,num_workers=4,multiprocessing_context='spawn',collate_fn=multivar_collate)
    for batch in loader:
        x,y=batch['background_normalized'].double(),batch['reference_normalized'].double()
        lat=latitude_weights(x).double()
        sse+=((x-y).square()*lat).sum((0,2,3));den+=float(lat.expand(len(x),1,*x.shape[-2:]).sum());count+=len(x)
        if count%100==0:
            atomic_json(out/'statistics_progress.json',{'cycles':count,'expected':4384});print(count,flush=True)
    assert count==4384
    mse=(sse/den).numpy();weights,scale=fit_weights(mse)
    std=data.base.std[:,0,0].numpy()
    report={'complete':True,'cycles':count,'split':'train','interval':['2018-01-01','2021-01-01'],
        'channel_names':CHANNELS,'training_normalized_background_mse':mse.tolist(),
        'loss_weights':weights.tolist(),'effective_normalized_error_scale':scale.tolist(),
        'physical_error_rms':(np.sqrt(mse)*std).tolist(),'physical_std':std.tolist(),
        'floor_policy':{'within_group_relative_rms':.1,'absolute_normalized_rms':1e-6},
        'group_policy':'equal 0.1 training-background loss contribution after floored per-channel scaling',
        'data_signature':data.signature,'test_2022_used':False,
        'training_baseline':metrics(sse.numpy(),den,weights,std,CHANNELS)}
    atomic_json(out/'loss_statistics.json',report)
    assert all(abs(v-.1)<1e-8 for v in report['training_baseline']['group_weighted_mse'].values())
    atomic_json(out/'preparation.json',{'complete':True,'signatures':{s:Data(ROOT,s).signature for s in ['train','val']},
        'loss_statistics_sha256':digest(out/'loss_statistics.json')})
if __name__=='__main__':main()
