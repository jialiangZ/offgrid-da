import pandas as pd
import pytest
import torch

from offgrid_da.data.forecast_contract import archesweather_t2m_background, replace_archesweather_t2m


def example():
    surface = torch.zeros(1, 4, 1, 121, 240)
    surface[:, 2] = torch.arange(240).float()[None, None, None, :]
    prediction = {"surface": surface, "level": torch.zeros(1, 6, 13, 121, 240)}
    stats = {"surface_mean": torch.ones(4,1,1,1)*270.25, "surface_std": torch.ones(4,1,1,1)*2,
             "level_mean": torch.zeros(6,13,1,1), "level_std": torch.ones(6,13,1,1)}
    return prediction, stats


def test_physical_grid_time_and_full_state_roundtrip():
    prediction, stats = example()
    original = {k: v.clone() for k,v in prediction.items()}
    result = archesweather_t2m_background(prediction, stats, ["2020-01-01T00:00Z"])
    assert result["valid_times"] == [pd.Timestamp("2020-01-01T06:00Z")]
    assert result["background"].shape == (1,1,121,240)
    assert result["background"][0,0,0,0] == 510.25
    assert result["grid_latitude"][0] == 90
    assert result["grid_longitude"][-1] == 358.5
    restored = replace_archesweather_t2m(prediction, result["background"], stats)
    for k in prediction:
        torch.testing.assert_close(restored[k], original[k])
        torch.testing.assert_close(prediction[k], original[k])
    changed = replace_archesweather_t2m(prediction, result["background"]+1, stats)
    torch.testing.assert_close(changed["surface"][:,2], prediction["surface"][:,2]+0.5)
    torch.testing.assert_close(changed["level"], original["level"])
    torch.testing.assert_close(changed["surface"][:,[0,1,3]], original["surface"][:,[0,1,3]])


@pytest.mark.parametrize("lead", [0, -6, 1, 24.0, True])
def test_reject_bad_lead(lead):
    prediction, stats = example()
    with pytest.raises(ValueError, match="lead_hours"):
        archesweather_t2m_background(prediction, stats, ["2020-01-01T00:00Z"], lead_hours=lead)


def test_timezone_and_nonfinite_checks():
    prediction, stats = example()
    with pytest.raises(ValueError, match="timezone"):
        archesweather_t2m_background(prediction, stats, ["2020-01-01"])
    prediction["level"][0,0,0,0,0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        archesweather_t2m_background(prediction, stats, ["2020-01-01T00:00Z"])


def test_bfloat16_does_not_round_normalization_mean():
    prediction, stats = example()
    prediction = {k: v.to(torch.bfloat16) for k,v in prediction.items()}
    prediction["surface"][:] = 0
    result = archesweather_t2m_background(prediction, stats, ["2020-01-01T00:00Z"])
    assert result["background"].dtype == torch.float32
    assert result["background"][0,0,0,0] == 270.25
