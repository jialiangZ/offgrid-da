import numpy as np
import torch
from error_balanced import fit_weights,balanced_loss,GROUPS,metrics

def test_training_baseline_group_contributions_are_equal_with_floor():
    mse=np.geomspace(1e-12,10,82);weights,scale=fit_weights(mse)
    assert np.isfinite(weights).all() and (scale>0).all()
    for _,ix in GROUPS:np.testing.assert_allclose((weights[ix]*mse[ix]).sum(),.1)
    np.testing.assert_allclose((weights*mse).sum(),1)

def test_loss_uses_area_average_not_sum_of_channel_weights():
    mse=np.geomspace(.00001,1,82);weights,_=fit_weights(mse)
    x=torch.zeros(2,82,5,8,requires_grad=True)
    target=torch.tensor(np.sqrt(mse),dtype=torch.float32)[None,:,None,None].expand_as(x)
    value=balanced_loss(x,target,weights)
    torch.testing.assert_close(value,torch.tensor(1.))
    value.backward();assert torch.isfinite(x.grad).all()

def test_reporting_score_and_contributions_match():
    mse=np.linspace(.001,1,82);weights,_=fit_weights(mse)
    names=[f'surface{i}' for i in range(4)]+[f'group{i}@{p}' for i in range(6) for p in range(13)]
    result=metrics(mse*100,100,weights,np.ones(82),names)
    assert abs(result['score']-1)<1e-10
    assert len(result['group_fraction'])==10
    assert abs(sum(result['group_fraction'].values())-1)<1e-10
