"""Analytic tests for the bilinear grid interpolation (plan section 3.7).

Uses artificial analytic fields on the real 240x121 grid geometry:
 1. linear-in-latitude field  -> exact recovery anywhere
 2. harmonic-in-longitude field -> exact recovery with periodic wrap
 3. points at 359.9 and 0.1 must see adjacent values across the seam
 4. pole behaviour (clamp, no extrapolation)
 5. NaN neighbour propagation
 6. numpy == torch implementation
"""
from __future__ import annotations

import numpy as np
import torch

from offgrid_da.data.alignment import (
    check_uniform_grid,
    normalize_longitude_360,
    sample_background_at_obs,
    sample_background_at_obs_torch,
)

# real grid geometry (240x121 equiangular with poles, 1.5 deg)
LAT = 90.0 - 1.5 * np.arange(121)          # descending 90 -> -90
LON = np.arange(240) * 1.5                 # ascending 0 -> 358.5


def make_field(lat_fn=None, lon_fn=None):
    lat = LAT[:, None] * np.ones((1, 240))
    lon = np.ones((121, 1)) * LON[None, :]
    f = np.zeros_like(lat)
    if lat_fn is not None:
        f += lat_fn(lat)
    if lon_fn is not None:
        f += lon_fn(lon)
    return f


def test_grid_geometry_uniform():
    dlat, dlon = check_uniform_grid(LAT, LON)
    assert abs(dlat - 1.5) < 1e-9 and abs(dlon - 1.5) < 1e-9
    assert LAT[0] == 90.0 and LAT[-1] == -90.0


def test_linear_in_latitude_exact():
    field = make_field(lat_fn=lambda x: 2.0 * x + 5.0)
    rng = np.random.default_rng(0)
    lat = rng.uniform(-89.9, 89.9, 200)
    lon = rng.uniform(0, 358.4, 200)
    out = sample_background_at_obs(field, LAT, LON, lat, lon)
    assert np.allclose(out, 2.0 * lat + 5.0, atol=1e-9)


def test_periodic_longitude_gridpoint_exact():
    # interpolation at grid longitudes must be exact (zero weight on neighbours)
    field = make_field(lon_fn=lambda x: np.sin(2 * np.pi * x / 360.0))
    lat = np.zeros(240)
    out = sample_background_at_obs(field, LAT, LON, lat, LON)
    assert np.allclose(out, np.sin(2 * np.pi * LON / 360.0), atol=1e-12)


def test_seam_stencil_structural():
    """The stencil at lon in (358.5, 360) must combine columns 239 and 0."""
    field = np.full((121, 240), np.nan)
    field[:, 239] = 10.0
    field[:, 0] = 20.0
    # midpoint between 358.5 and 360(=0) -> average of the two columns
    out = sample_background_at_obs(field, LAT, LON, np.array([0.0]), np.array([359.25]))
    assert abs(out[0] - 15.0) < 1e-12
    # quarter point: 358.875 -> 75% col239 (358.5) + 25% col0 (360)
    out2 = sample_background_at_obs(field, LAT, LON, np.array([45.0]), np.array([358.875]))
    assert abs(out2[0] - 12.5) < 1e-12


def test_longitude_seam_neighbors():
    """359.9 and 0.1 sample the smooth periodic field correctly (Test D)."""
    field = make_field(lon_fn=lambda x: np.sin(2 * np.pi * x / 360.0))
    out = sample_background_at_obs(
        field, LAT, LON,
        np.array([10.0, 10.0]), np.array([359.9, 0.1]),
    )
    expected = np.sin(2 * np.pi * np.array([359.9, 0.1]) / 360.0)
    # bilinear truncation of a sinusoid on a 1.5-deg grid is ~1e-6 near the
    # seam (verified analytically); a broken seam would give O(0.03) error
    assert np.allclose(out, expected, atol=1e-5)


def test_seam_error_comparable_to_interior():
    """Interpolation error at the seam must match interior-point error scale."""
    field = make_field(lon_fn=lambda x: np.sin(2 * np.pi * x / 360.0))
    rng = np.random.default_rng(3)
    lon_seam = rng.uniform(350.0, 360.0, 500)
    lon_int = rng.uniform(90.0, 100.0, 500)
    lat = rng.uniform(-60, 60, 500)
    err_seam = np.abs(
        sample_background_at_obs(field, LAT, LON, lat, lon_seam)
        - np.sin(2 * np.pi * lon_seam / 360.0)
    )
    err_int = np.abs(
        sample_background_at_obs(field, LAT, LON, lat, lon_int)
        - np.sin(2 * np.pi * lon_int / 360.0)
    )
    assert err_seam.max() < 3 * err_int.max() + 1e-12


def test_longitude_normalization():
    assert normalize_longitude_360(np.array([-180.0, -0.25, 0.0, 359.9, 180.0])).tolist() == [
        180.0, 359.75, 0.0, 359.9, 180.0,
    ]
    t = torch.tensor([-180.0, -0.25, 0.0, 359.9, 180.0])
    assert torch.allclose(
        normalize_longitude_360(t), torch.tensor([180.0, 359.75, 0.0, 359.9, 180.0], dtype=torch.float64)
    )


def test_pole_clamp_no_extrapolation():
    # constant field: pole queries must return the constant exactly
    field = np.full((121, 240), 3.0)
    out = sample_background_at_obs(field, LAT, LON, np.array([90.0, -90.0]), np.array([10.0, 200.0]))
    assert np.allclose(out, 3.0)
    # linear field: pole values equal grid row values (clamped, not extrapolated)
    field2 = make_field(lat_fn=lambda x: x)
    out2 = sample_background_at_obs(field2, LAT, LON, np.array([90.0, -90.0]), np.array([10.0, 200.0]))
    assert np.allclose(out2, [90.0, -90.0])


def test_nan_neighbour_propagation():
    field = make_field(lat_fn=lambda x: x)
    field[60, 100] = np.nan
    out = sample_background_at_obs(field, LAT, LON, np.array([LAT[60]]), np.array([LON[100]]))
    assert np.isnan(out)
    out2 = sample_background_at_obs(field, LAT, LON, np.array([LAT[60] + 1.4]), np.array([LON[100]]))
    assert np.isnan(out2)  # still touches the NaN cell within bilinear stencil


def test_numpy_torch_consistency_and_differentiable():
    rng = np.random.default_rng(2)
    field = rng.normal(size=(121, 240)).astype(np.float64)
    lat = rng.uniform(-88, 88, 64)
    lon = rng.uniform(0, 360, 64)
    np_out = sample_background_at_obs(field, LAT, LON, lat, lon)

    ft = torch.tensor(field, requires_grad=False)
    lat_t = torch.tensor(lat, requires_grad=True)
    lon_t = torch.tensor(lon, requires_grad=True)
    t_out = sample_background_at_obs_torch(ft, torch.tensor(LAT), torch.tensor(LON), lat_t, lon_t)
    assert torch.allclose(torch.tensor(np_out), t_out, atol=1e-10)

    t_out.sum().backward()
    assert lat_t.grad is not None and torch.isfinite(lat_t.grad).all()
    assert lon_t.grad is not None and torch.isfinite(lon_t.grad).all()
