import torch
import pytest

from offgrid_da.models import ObservationEncoder


def sample():
    return {
        "obs_values": torch.tensor([[2., float("nan"), 0.]]),
        "obs_locations": torch.tensor([[[30., 120., 20.], [0., 0., 0.], [45., 10., 0.]]]),
        "obs_time_offsets": torch.tensor([[-5., 0., 0.]]),
        "obs_variables": torch.tensor([[1, 0, 2]]),
        "obs_sources": torch.tensor([[0, 0, 1]]),
        "obs_quality": torch.tensor([[3, 0, 2]]),
        "observation_mask": torch.tensor([[True, True, False]]),
        "obs_innovation": torch.tensor([[1.5, 0., 0.]]),
    }


def test_encoder_preserves_irregular_tokens_and_masks_missing_values():
    encoder = ObservationEncoder(dim=24, value_mean=1., value_std=2.)
    out = encoder(sample())
    assert out["tokens"].shape == (1, 3, 24)
    assert out["observation_mask"].tolist() == [[True, False, False]]
    assert out["innovation_mask"].tolist() == [[True, False, False]]
    assert torch.equal(out["tokens"][0, 1:], torch.zeros(2, 24))


def test_encoder_backpropagates_for_valid_records():
    encoder = ObservationEncoder(dim=16)
    tokens = encoder(sample())["tokens"]
    tokens.sum().backward()
    assert encoder.value.weight.grad is not None


def test_encoder_rejects_invalid_location_shape():
    batch = sample()
    batch["obs_locations"] = torch.zeros(1, 3, 2)
    with pytest.raises(ValueError, match="locations"):
        ObservationEncoder(dim=8)(batch)
