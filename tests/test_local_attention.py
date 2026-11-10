import torch
from offgrid_da.models import CrossAttentionAnalysisModel

def test_zero_initialization_and_empty_identity_after_update():
    model=CrossAttentionAnalysisModel(dim=8,heads=2)
    bg=torch.randn(1,1,3,4)
    obs={'obs_values':torch.ones(1,1),'obs_locations':torch.zeros(1,1,3),'observation_mask':torch.ones(1,1,dtype=torch.bool)}
    assert torch.equal(model(bg,obs)['analysis'],bg)
    with torch.no_grad(): model.output.weight.fill_(0.2); model.output.bias.fill_(1)
    obs['observation_mask'][:]=False
    assert torch.equal(model(bg,obs)['analysis'],bg)

def test_dateline_poles_and_no_neighbor():
    torch.manual_seed(9)
    model=CrossAttentionAnalysisModel(dim=8,heads=2,radius_km=300,max_neighbors=2).eval()
    q=torch.randn(1,12,8); tokens=torch.randn(1,1,8); mask=torch.ones(1,1,dtype=torch.bool)
    loc=torch.tensor([[[0.,359.,0.]]])
    update=model._local(q,tokens,mask,loc,3,4,0)
    assert update[0,4].abs().max()>0  # equator, longitude 0, across dateline
    assert torch.equal(update[0,6],torch.zeros(8))  # opposite side
    loc=torch.tensor([[[89.,180.,0.]]])
    update=model._local(q,tokens,mask,loc,3,4,0)
    assert (update[0,:4].abs().sum(-1)>0).all()
