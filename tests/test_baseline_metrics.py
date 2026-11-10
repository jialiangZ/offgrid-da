"""Baseline 0 and fixed grid metrics."""
from __future__ import annotations

import numpy as np
import pytest

from offgrid_da.evaluation import area_weighted_mae, area_weighted_rmse, baseline0


def test_baseline0_returns_background_unchanged():
    background = np.arange(24, dtype=np.float32).reshape(1, 3, 8)
    analysis = baseline0(background)
    assert np.array_equal(analysis, background)
    assert analysis.dtype == background.dtype


def test_weighted_rmse_and_mae_have_expected_values():
    latitude = np.array([60.0, 0.0])
    prediction = np.array([[2.0], [0.0]])
    reference = np.zeros_like(prediction)
    expected_rmse = np.sqrt((0.5 * 4.0 + 1.0 * 0.0) / 1.5)
    expected_mae = (0.5 * 2.0 + 1.0 * 0.0) / 1.5
    assert area_weighted_rmse(prediction, reference, latitude) == pytest.approx(expected_rmse)
    assert area_weighted_mae(prediction, reference, latitude) == pytest.approx(expected_mae)


def test_weighted_metrics_ignore_nonfinite_cells():
    latitude = np.array([0.0, 0.0])
    prediction = np.array([[3.0], [np.nan]])
    reference = np.zeros_like(prediction)
    assert area_weighted_rmse(prediction, reference, latitude) == pytest.approx(3.0)
    assert area_weighted_mae(prediction, reference, latitude) == pytest.approx(3.0)


def test_weighted_metric_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="identical shapes"):
        area_weighted_rmse(np.zeros((2, 3)), np.zeros((3, 2)), np.array([0.0, 1.0]))
