"""Full 2021 paired scoring; never opens 2022 data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from offgrid_da.data.full_state_data import FullStateInputDataset, full_state_collate
from offgrid_da.data.arches_training import atomic_json
from offgrid_da.models.full_state_t2m import FullStateT2MAnalysis
from single_cycle_controls import FullStateCorrection, local_oi, shuffled_observations
from train_single_cycle import pack, weights_for


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mode', choices=['full82','t2m_masked'], default='full82')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--oi-only', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    experiment = ROOT/'outputs/single_da_05_input82_ablation'
    dataset = FullStateInputDataset(ROOT,'val',experiment,a.mode)
    cfg = json.loads((experiment/f'seed{a.seed}'/a.mode/'config.json').read_text())
    std = dataset.norm_stats.std('background','2m_temperature')
    selected = np.linspace(0,len(dataset)-1,a.limit,dtype=int).tolist() if a.limit else list(range(len(dataset)))
    loader = DataLoader(Subset(dataset,selected),batch_size=1,num_workers=2,
                        multiprocessing_context='spawn',collate_fn=full_state_collate)
    model = None
    provenance = {}
    if not a.oi_only:
        path = experiment/f'seed{a.seed}'/a.mode/'best.pt'
        ckpt = torch.load(path,map_location='cpu',weights_only=False)
        if ckpt['signature']['data']['val'] != dataset.signature:
            raise ValueError('DA checkpoint/data mismatch')
        model = FullStateT2MAnalysis(**cfg['model']).cuda().eval()
        model.load_state_dict(ckpt['model'])
        provenance = {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'step':ckpt['step']}
    names = ['background','local_oi'] if a.oi_only else ['background','real','masked','shuffle0','shuffle1','shuffle2']
    generators = [torch.Generator().manual_seed(20261101+i) for i in range(3)]
    sums = {name:0. for name in names}
    denominator = 0.
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'per_cycle.jsonl').open('w') as stream, torch.inference_mode():
        for i,batch in enumerate(loader):
            x,obs,y = pack(batch,'cuda:0')
            obs['grid_state'] = batch['grid_state'].cuda()
            predictions = {'background':x}
            if a.oi_only:
                predictions['local_oi'] = local_oi(x,obs,std)
            else:
                predictions['real'] = model(x,obs)['analysis']
                empty = {k:v.clone() for k,v in obs.items()}
                empty['observation_mask'].zero_()
                predictions['masked'] = model(x,empty)['analysis']
                if not torch.equal(predictions['masked'],x):
                    raise ValueError('Empty-observation identity failed')
                for j,g in enumerate(generators):
                    predictions[f'shuffle{j}'] = model(x,shuffled_observations(obs,g))['analysis']
            weight = weights_for(x).double()
            den = float(weight.expand_as(x).sum())
            row = {'index':selected[i],'denominator':den,'sse_K2':{}}
            for name,pred in predictions.items():
                if not torch.isfinite(pred).all():
                    raise ValueError('Nonfinite output')
                value = float(((pred.double()-y.double()).square()*weight).sum())*std**2
                sums[name] += value
                row['sse_K2'][name] = value
            denominator += den
            stream.write(json.dumps(row)+'\n')
            if (i+1)%100 == 0:
                print('cycles',i+1,flush=True)
    atomic_json(a.output/'result.json',{'complete':True,'cycles':len(selected),'seed':a.seed,
        'mode':a.mode,'rmse_K':{k:math.sqrt(v/denominator) for k,v in sums.items()},
        'data_signature':dataset.signature,'checkpoint':provenance,'test_2022_used':False,
        'oi_parameters':{'sigma_b_K':1.,'sigma_o_K':2.5,'length_km':500.,'radius_km':1000.,'neighbors':16},
        'note':'2021 validation only. OI parameters fixed a priori; 2.5 K is an approximate representativeness scale.'})

if __name__ == '__main__':
    main()
