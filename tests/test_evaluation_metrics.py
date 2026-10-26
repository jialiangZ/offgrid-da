"""Area-weighted grid metrics (plan section 5.2) and station metrics.

This guards against the weighting bug found in Phase 1 development
(dividing by row weights instead of broadcast cell weights).
"""
from __future__ import annotations

import numpy as np


def area_weighted_rmse(pred: np.ndarray, target: np.ndarray, lat: np.ndarray) -> float:
    err2 = (pred - target) ** 2
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], err2.shape)
    return float(np.sqrt(np.nansum(w * err2) / np.nansum(w)))


def test_uniform_grid_weighted_equals_unweighted():
    lat = np.array([60.0, 0.0, -60.0])
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 5))
    b = rng.normal(size=(3, 5))
    # for constant weights (equator-only band) the two must agree
    lat_eq = np.zeros(3)
    assert abs(area_weighted_rmse(a, b, lat_eq) - np.sqrt(((a - b) ** 2).mean())) < 1e-9


def test_weighting_follows_cosine_of_latitude():
    lat = np.array([90.0, 0.0])  # pole row weight ~ cos(90) = 6e-17, not exactly 0
    a = np.array([[1000.0], [0.0]])
    b = np.zeros((2, 1))
    # the pole row's contribution is suppressed by ~17 orders of magnitude
    assert abs(area_weighted_rmse(a, b, lat)) < 1e-4


def test_single_cell_exact():
    lat = np.array([0.0])
    a = np.array([[3.0, 4.0]])
    b = np.zeros((1, 2))
    assert abs(area_weighted_rmse(a, b, lat) - np.sqrt((9 + 16) / 2)) < 1e-12
