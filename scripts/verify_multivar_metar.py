"""Real-batch engineering gate; does not train a full-state analysis model."""
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from offgrid_da.data.multivar_dataset import MultivarMetarDataset,multivar_collate
from offgrid_da.data.metar_multivar import SPECS
from offgrid_da.models.full_state_t2m import FullStateT2MAnalysis
from offgrid_da.data.arches_training import atomic_json
from train_single_cycle import pack,training_loss

def main():
    torch.manual_seed(42);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    samples=[];summaries=[]
    for split,index in [('train',0),('train',1000),('val',0),('val',720)]:
        dataset=MultivarMetarDataset(ROOT,split)
        sample=dataset[index];samples.append(sample)
        assert not set(sample['context_station_ids'])&set(sample['heldout_station_ids'])
        for prefix,stations,role in [('obs','context_station_ids',dataset.context_stations),('heldout_obs','heldout_station_ids',dataset.heldout_stations)]:
            assert len(sample[stations])==len(sample[prefix+'_values'])
            assert all(station in role for station in sample[stations])
            offsets=sample[prefix+'_time_offsets'];assert ((offsets>-60)&(offsets<=0)).all()
            for v,(mean,std) in dataset.scales.items():
                ids=sample[prefix+'_variables']==v
                torch.testing.assert_close(sample[prefix+'_values_normalized'][ids],(sample[prefix+'_values'][ids]-mean)/std)
        summaries.append({'split':split,'index':index,'valid_time':str(sample['valid_time']),
            'context_by_id':{str(v):int((sample['obs_variables']==v).sum()) for v in SPECS},
            'heldout_by_id':{str(v):int((sample['heldout_obs_variables']==v).sum()) for v in SPECS}})
    batch=multivar_collate(samples[:1]+samples[2:3])
    x,obs,target=pack(batch,'cuda:0');obs['grid_state']=batch['grid_state'].cuda()
    model=FullStateT2MAnalysis(dim=32,heads=4,depth=2,max_neighbors=80,query_chunk=256).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    losses=[]
    for _ in range(3):
        optimizer.zero_grad();prediction=model(x,obs)['analysis'];loss=training_loss(prediction,target)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step();losses.append(float(loss.detach()))
    model.eval()
    with torch.inference_mode():
        real=model(x,obs)['analysis'];empty={k:v.clone() for k,v in obs.items()};empty['observation_mask'].zero_()
        assert torch.equal(model(x,empty)['analysis'],x)
        sensitivity={}
        for variable in SPECS:
            changed={k:v.clone() for k,v in obs.items()}
            mask=changed['observation_mask']&(changed['obs_variables']==variable)
            assert mask.any()
            changed['obs_values_normalized'][mask]+=1.
            value=float((model(x,changed)['analysis']-real).abs().max())
            assert value>1e-9, f'No sensitivity to variable {variable}'
            sensitivity[str(variable)]=value
    result={'passed':True,'samples':summaries,'batch_grid_shape':list(batch['grid_state'].shape),
        'batch_observation_shape':list(batch['observation_mask'].shape),'losses':losses,
        'finite_forward_backward':True,'empty_exact_identity':True,'sensitivity_by_type':sensitivity,
        'peak_memory_MiB':torch.cuda.max_memory_allocated()/2**20,
        'scope':'Observation pipeline acceptance with existing T2M-output model; not a trained 82-output model.'}
    atomic_json(ROOT/'outputs/single_da_06_obs_metar5_prep/acceptance.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
