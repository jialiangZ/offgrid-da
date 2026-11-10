"""Real fixed-batch overfit and checkpoint acceptance for the new loss."""
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_full_update_v1 import Data
from offgrid_da.data.multivar_dataset import multivar_collate
from offgrid_da.models import CrossAttentionAnalysisModel
from train_single_cycle import pack,BackgroundCorrection
from error_balanced import balanced_loss
from offgrid_da.data.arches_training import atomic_json

def main():
    out=ROOT/'outputs/single_da_09_full82_loss_balanced';stats=json.loads((out/'loss_statistics.json').read_text())
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    batch=multivar_collate([Data(ROOT,'train')[0]]);x,obs,y=pack(batch,'cuda:0')
    results={}
    for kind in ['da','neural_correction']:
        torch.manual_seed(42);template=CrossAttentionAnalysisModel(channels=82,dim=128,depth=4,heads=8,max_neighbors=80,query_chunk=256)
        model=(template if kind=='da' else BackgroundCorrection(template)).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        with torch.no_grad():assert torch.equal(model(x,obs)['analysis'],x)
        history=[]
        for step in range(64):
            opt.zero_grad();loss=balanced_loss(model(x,obs)['analysis'],y,stats['loss_weights'])
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            history.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            prediction=model(x,obs)['analysis'];last=float(balanced_loss(prediction,y,stats['loss_weights']))
            assert last<history[0],f'Fixed batch did not improve: {kind}'
            if kind=='da':
                empty={k:v.clone() for k,v in obs.items()};empty['observation_mask'].zero_();assert torch.equal(model(x,empty)['analysis'],x)
            else:
                changed={k:v.clone() for k,v in obs.items()};changed['obs_values_normalized']+=100
                assert torch.equal(model(x,changed)['analysis'],prediction)
            checkpoint=out/f'acceptance_{kind}.pt';torch.save(model.state_dict(),checkpoint)
            model.load_state_dict(torch.load(checkpoint,weights_only=True))
            assert torch.equal(model(x,obs)['analysis'],prediction)
        results[kind]={'passed':True,'initial_loss':history[0],'final_loss':last,'loss_history':history,'checkpoint_reload_exact':True}
        del model,opt,template,prediction,loss;torch.cuda.empty_cache()
    atomic_json(out/'acceptance.json',{'passed':True,'models':results})
if __name__=='__main__':main()
