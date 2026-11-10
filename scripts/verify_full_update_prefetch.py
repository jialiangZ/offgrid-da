import sys,json,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_full_update_v1 import Data
from offgrid_da.data.multivar_dataset import multivar_collate
from torch.utils.data import DataLoader,Subset
from train_full_update_prefetch import loader
from offgrid_da.data.arches_training import atomic_json

def main():
 torch.set_num_threads(4)
 data=Subset(Data(ROOT,'train'),[0,101,202,303])
 a=DataLoader(data,batch_size=1,num_workers=0,collate_fn=multivar_collate)
 b=loader(data,batch_size=1,num_workers=0,collate_fn=multivar_collate)
 count=0
 for x,y in zip(a,b):
  assert x['sample_ids']==y['sample_ids']
  for key in x:
   if isinstance(x[key],torch.Tensor):assert torch.equal(x[key],y[key]),key
  count+=1
 assert count==4
 atomic_json(ROOT/'outputs/single_da_08_full82_large/prefetch_data_acceptance.json',{'passed':True,'samples':count,'all_tensor_fields_bitwise_equal':True})
 print('PASS: all tensor fields and sample ordering exactly equal')
if __name__=='__main__':main()
