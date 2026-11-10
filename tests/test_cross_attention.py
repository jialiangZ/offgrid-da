import pytest
import torch

from offgrid_da.models import CrossAttentionAnalysisModel


def observations(n=5):
    return {
        "obs_values_normalized": torch.randn(2, n),
        "obs_locations": torch.randn(2, n, 3),
        "obs_time_offsets": torch.randn(2, n),
        "obs_variables": torch.zeros(2, n, dtype=torch.long),
        "obs_sources": torch.zeros(2, n, dtype=torch.long),
        "obs_quality": torch.zeros(2, n, dtype=torch.long),
        "observation_mask": torch.ones(2, n, dtype=torch.bool),
    }


@pytest.mark.parametrize("case", ["normal", "mixed", "empty", "nan", "zero_length"])
@pytest.mark.parametrize("injection", ["entry", "layered"])
def test_finite_forward_and_backward(case, injection):
    torch.manual_seed(42)
    obs = observations(0 if case == "zero_length" else 5)
    if case == "mixed":
        obs["observation_mask"][1] = False
    if case == "empty":
        obs["observation_mask"][:] = False
    if case == "nan":
        obs["obs_locations"][0, 0] = float("nan")
        obs["obs_values_normalized"][1, 1] = float("nan")
        obs["obs_time_offsets"][1, 2] = float("inf")
    model = CrossAttentionAnalysisModel(dim=8, heads=2)
    model.injection = injection
    out = model(torch.randn(2, 1, 5, 7), obs)["analysis"]
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_order_padding_and_observation_sensitivity():
    torch.manual_seed(42)
    model = CrossAttentionAnalysisModel(dim=8, heads=2).eval()
    with torch.no_grad():
        model.output.weight.fill_(0.1)
    bg = torch.randn(2, 1, 5, 7)
    obs = observations()
    obs["observation_mask"][:, -1] = False
    with torch.no_grad():
        expected = model(bg, obs)["analysis"]
        perm = torch.tensor([4, 2, 0, 3, 1])
        shuffled = {k: v[:, perm] for k, v in obs.items()}
        torch.testing.assert_close(model(bg, shuffled)["analysis"], expected)
        padded = {k: v.clone() for k, v in obs.items()}
        for key in padded:
            if key != "observation_mask":
                padded[key][:, -1] = 12345
        torch.testing.assert_close(model(bg, padded)["analysis"], expected)
        changed = {k: v.clone() for k, v in obs.items()}
        changed["obs_values_normalized"][:, :4] += 2
        assert (model(bg, changed)["analysis"] - expected).abs().max() > 1e-6
