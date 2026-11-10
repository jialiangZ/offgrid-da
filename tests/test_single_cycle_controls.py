import torch
from single_cycle_controls import FullStateCorrection,local_oi,shuffled_observations


def test_correction_cannot_read_observations_and_can_correct_empty_samples():
    model = FullStateCorrection(dim=8,heads=2,depth=1)
    bg = torch.zeros(1,1,5,8)
    state = torch.randn(1,82,5,8)
    with torch.no_grad():
        model.output.bias.fill_(.2)
    a = {'grid_state':state,'obs_values_normalized':torch.zeros(1,2),'observation_mask':torch.zeros(1,2,dtype=torch.bool)}
    b = {**a,'obs_values_normalized':torch.ones(1,2)*100,'observation_mask':torch.ones(1,2,dtype=torch.bool)}
    assert torch.equal(model(bg,a)['analysis'],model(bg,b)['analysis'])
    assert (model(bg,a)['analysis'] != bg).any()


def test_oi_single_observation_analytic_and_mask_and_longitude_seam():
    bg = torch.zeros(1,1,5,8)
    obs = {'obs_locations':torch.tensor([[[0.,0.,0.]]]),'obs_values_normalized':torch.tensor([[2.]]),'observation_mask':torch.ones(1,1,dtype=torch.bool)}
    result = local_oi(bg,obs,std=1.,sigma_b_K=1.,sigma_o_K=2.5)
    torch.testing.assert_close(result[0,0,2,0],torch.tensor(2/(1+2.5**2)),atol=1e-6,rtol=0)
    equivalent = {**obs,'obs_locations':torch.tensor([[[0.,360.,0.]]])}
    torch.testing.assert_close(result,local_oi(bg,equivalent,std=1.),atol=1e-6,rtol=0)
    obs['observation_mask'].zero_()
    assert torch.equal(local_oi(bg,obs,std=1.),bg)


def test_shuffle_retains_masked_values_and_metadata():
    obs = {'obs_values_normalized':torch.tensor([[1.,2.,3.,999.]]),'observation_mask':torch.tensor([[True,True,True,False]]),'obs_locations':torch.randn(1,4,3)}
    changed = shuffled_observations(obs,torch.Generator().manual_seed(42))
    assert torch.equal(changed['obs_locations'],obs['obs_locations'])
    assert changed['obs_values_normalized'][0,3] == 999
    assert sorted(changed['obs_values_normalized'][0,:3].tolist()) == [1.,2.,3.]
