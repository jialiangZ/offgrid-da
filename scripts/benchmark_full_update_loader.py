import sys,time,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_full_update_v1 import Data,loss
from offgrid_da.data.multivar_dataset import multivar_collate
from offgrid_da.models import CrossAttentionAnalysisModel
from train_single_cycle import pack
from offgrid_da.data.arches_training import atomic_json

def main():
 torch.set_num_threads(4);results=[]
 data=Data(ROOT,'train');subset=Subset(data,[i*101 for i in range(16)])
 for workers in [0,2,4]:
  torch.manual_seed(42)
  model=CrossAttentionAnalysisModel(channels=82,dim=128,heads=8,depth=4,max_neighbors=80,query_chunk=256).cuda()
  opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
  kw=dict(batch_size=1,num_workers=workers,pin_memory=True,collate_fn=multivar_collate)
  if workers:kw.update(multiprocessing_context='spawn',prefetch_factor=2)
  loader=DataLoader(subset,**kw);it=iter(loader);waits=[];computes=[]
  for i in range(16):
   t=time.perf_counter();b=next(it);wait=time.perf_counter()-t
   t=time.perf_counter();x,o,y=pack(b,'cuda:0');opt.zero_grad();v=loss(model(x,o)['analysis'],y);v.backward();opt.step();torch.cuda.synchronize()
   if i>=2:waits.append(wait);computes.append(time.perf_counter()-t)
  result=dict(workers=workers,mean_data_wait_s=sum(waits)/len(waits),mean_compute_s=sum(computes)/len(computes),steps_per_second=len(waits)/(sum(waits)+sum(computes)))
  results.append(result);print(json.dumps(result),flush=True)
  del it,loader,model,opt,v;torch.cuda.empty_cache()
 atomic_json(ROOT/'outputs/single_da_08_full82_large/loader_benchmark.json',results)
if __name__=='__main__':main()
