import sys,json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_full_update_v1 import Data,loss
from offgrid_da.data.multivar_dataset import multivar_collate
from offgrid_da.models import CrossAttentionAnalysisModel
from train_single_cycle import BackgroundCorrection,pack
from offgrid_da.data.arches_training import atomic_json

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 dataset=Data(ROOT,'train');batch=multivar_collate([dataset[0]]);x,obs,target=pack(batch,'cuda:0')
 results={}
 for kind in ['da','neural_correction']:
  folder=ROOT/f'outputs/single_da_08_full82_large/seed42/{kind}/formal';cfg=json.loads((folder/'config.json').read_text())
  template=CrossAttentionAnalysisModel(channels=82,**cfg['model'])
  model=(template if kind=='da' else BackgroundCorrection(template)).cuda().eval()
  checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model'])
  with torch.inference_mode():
   prediction=model(x,obs)['analysis'];assert prediction.shape==x.shape and torch.isfinite(prediction).all()
   empty={k:v.clone() for k,v in obs.items()};empty['observation_mask'].zero_()
   if kind=='da':assert torch.equal(model(x,empty)['analysis'],x)
   else:assert torch.equal(model(x,empty)['analysis'],prediction)
  # Fit one training batch with an in-memory clone; frozen checkpoint is never overwritten.
  optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.)
  history=[]
  for step in range(16):
   model.train();optimizer.zero_grad();value=loss(model(x,obs)['analysis'],target)
   assert torch.isfinite(value);history.append(float(value));value.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
  assert min(history[1:])<history[0], 'Fixed training batch failed to improve'
  results[kind]={'initial_loss':history[0],'best_loss':min(history),'steps':16,'finite_gradients':True,'full82_forward':True,'empty_identity_or_NC_independence':True}
 atomic_json(ROOT/'outputs/cycling_full82_02_large/real_batch_gate.json',{'passed':True,'results':results,'no_weights_saved':True})
if __name__=='__main__':main()
