import torch
from offgrid_da.models import CrossAttentionAnalysisModel
from train_single_cycle import BackgroundCorrection

def test_large_full82_contract():
 torch.set_num_threads(4)
 model=CrossAttentionAnalysisModel(channels=82,dim=128,depth=4,heads=8,max_neighbors=80,query_chunk=256).eval()
 x=torch.randn(1,82,5,8)
 obs={'obs_values_normalized':torch.randn(1,5),'obs_locations':torch.tensor([[[0.,0.,0.]]*5]),'obs_variables':torch.arange(5)[None],'observation_mask':torch.ones(1,5,dtype=torch.bool)}
 assert sum(p.numel() for p in model.parameters())>1_000_000
 with torch.inference_mode():
  assert torch.equal(model(x,obs)['analysis'],x)
  model.output.bias.fill_(.1)
  assert model(x,obs)['analysis'].shape==x.shape
  obs['observation_mask'].zero_()
  assert torch.equal(model(x,obs)['analysis'],x)
 nc=BackgroundCorrection(model)
 assert sum(p.numel() for p in nc.parameters())>1_000_000
 with torch.inference_mode():assert torch.isfinite(nc(x,obs)['analysis']).all()
