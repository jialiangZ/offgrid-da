import torch
from offgrid_da.models import CrossAttentionAnalysisModel
from train_single_cycle import BackgroundCorrection

def sample():
 torch.manual_seed(42)
 x=torch.randn(1,82,5,8)
 o={'obs_values_normalized':torch.randn(1,5),'obs_locations':torch.tensor([[[0.,0.,0.]]*5]),'obs_variables':torch.arange(5)[None],'observation_mask':torch.ones(1,5,dtype=torch.bool)}
 return x,o

def test_full82_zero_and_masked_identity_after_updates():
 x,o=sample();m=CrossAttentionAnalysisModel(channels=82,dim=8,heads=2,depth=1)
 assert torch.equal(m(x,o)['analysis'],x)
 optimizer=torch.optim.Adam(m.parameters(),lr=.003)
 initial=None
 for _ in range(8):
  optimizer.zero_grad();loss=(m(x,o)['analysis']-x-.2).square().mean()
  initial=float(loss) if initial is None else initial
  loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());optimizer.step()
 assert float(loss)<initial
 empty={**o,'observation_mask':torch.zeros_like(o['observation_mask'])}
 assert torch.equal(m(x,empty)['analysis'],x)
 assert m(x,o)['analysis'].shape==x.shape
 clone=CrossAttentionAnalysisModel(channels=82,dim=8,heads=2,depth=1);clone.load_state_dict(m.state_dict())
 torch.testing.assert_close(m(x,o)['analysis'],clone(x,o)['analysis'],rtol=0,atol=0)

def test_nc_full82_observation_independence():
 x,o=sample();m=BackgroundCorrection(CrossAttentionAnalysisModel(channels=82,dim=8,heads=2,depth=1))
 with torch.no_grad():m.output.bias.fill_(.1)
 changed={**o,'obs_values_normalized':o['obs_values_normalized']+100}
 assert torch.equal(m(x,o)['analysis'],m(x,changed)['analysis'])
 assert m(x,o)['analysis'].shape==x.shape

def test_all82_longitude_and_level_layout_roundtrip():
 state=torch.arange(82*5*8).reshape(1,82,5,8).float()
 surface=state[:,:4].roll(4,-1).unsqueeze(2)
 upper=state[:,4:].reshape(1,6,13,5,8).roll(4,-1)
 canonical=torch.cat([surface[:,:,0],upper.flatten(1,2)],1).roll(-4,-1)
 assert torch.equal(canonical,state)
