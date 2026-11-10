import pytest
import torch
from offgrid_da.models.full_state_t2m import FullStateT2MAnalysis


def inputs():
    background = torch.randn(1, 1, 5, 7)
    state = torch.randn(1, 82, 5, 7)
    state[:, 2:3] = background
    obs = {'obs_values_normalized': torch.randn(1, 3), 'obs_locations': torch.zeros(1, 3, 3),
           'observation_mask': torch.ones(1, 3, dtype=torch.bool), 'grid_state': state}
    return background, obs


def test_full_input_only_t2m_output_and_empty_identity_after_learning():
    torch.manual_seed(42)
    model = FullStateT2MAnalysis(dim=8, heads=2, depth=1)
    bg, obs = inputs()
    assert torch.equal(model(bg, obs)['analysis'], bg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        prediction = model(bg, obs)['analysis']
        assert prediction.shape == bg.shape
        (prediction - bg - 1).square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    changed = {k: v.clone() for k, v in obs.items()}
    changed['grid_state'][:, 10] += 2
    with torch.no_grad():
        assert (model(bg, obs)['analysis'] - model(bg, changed)['analysis']).abs().max() > 1e-7
        changed['observation_mask'].zero_()
        assert torch.equal(model(bg, changed)['analysis'], bg)


def test_equal_seed_gives_identical_architecture_and_initialization():
    torch.manual_seed(43)
    a = FullStateT2MAnalysis(dim=8, heads=2, depth=1)
    torch.manual_seed(43)
    b = FullStateT2MAnalysis(dim=8, heads=2, depth=1)
    for name, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[name])
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_rejects_missing_or_incorrect_context():
    model = FullStateT2MAnalysis(dim=8, heads=2, depth=1)
    bg, obs = inputs()
    del obs['grid_state']
    with pytest.raises(ValueError):
        model(bg, obs)
    obs['grid_state'] = torch.zeros(1, 81, 5, 7)
    with pytest.raises(ValueError):
        model(bg, obs)

