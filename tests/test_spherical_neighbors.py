"""Spherical geometry tests: distances and the 359.9/0.1 seam (plan 4.7 Test D)."""
from __future__ import annotations

import numpy as np

from offgrid_da.data.alignment import spherical_haversine_km


def test_seam_points_are_close():
    d = spherical_haversine_km(np.array([10.0]), np.array([359.9]), np.array([10.0]), np.array([0.1]))
    assert d[0] < 40.0  # 0.2 deg at 10N ~ 22 km

def test_opposite_points_are_far():
    d = spherical_haversine_km(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([180.0]))
    assert 19000 < d[0] < 21000  # half circumference ~20015 km

def test_symmetry():
    a = spherical_haversine_km(np.array([52.0, -33.0]), np.array([13.0, 150.0]),
                               np.array([35.0, 45.0]), np.array([260.0, 20.0]))
    b = spherical_haversine_km(np.array([35.0, 45.0]), np.array([260.0, 20.0]),
                               np.array([52.0, -33.0]), np.array([13.0, 150.0]))
    assert np.allclose(a, b)

def test_pole_to_pole():
    d = spherical_haversine_km(np.array([90.0]), np.array([0.0]), np.array([-90.0]), np.array([123.0]))
    assert 19000 < d[0] < 21000
