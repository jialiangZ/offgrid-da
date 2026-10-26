"""Time alignment and grid interpolation (plan sections 3.3, 3.7).

Time rule (verified against real data):
    valid_time = init_time + lead_time   (Pangu: prediction_timedelta 6h..240h)
    init hours 00/12 UTC + lead 6h  =>  analysis times 06/18 UTC, which all
    exist in the 6-hourly ERA5 (00/06/12/18 UTC).

Grid interpolation:
- canonical grid: latitude DESCENDING (90 -> -90), longitude ASCENDING [0, 360)
- bilinear with periodic longitude (0 <-> 358.5 are adjacent on the 1.5 deg grid)
- latitude clamped at the poles (no periodicity)
- a query is NaN if any participating grid neighbour is NaN
- both numpy and differentiable torch implementations; they are tested to
  agree on analytic fields.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .analysis import ERA5Reader
from .forecast import PanguReader

# --------------------------------------------------------------------------- #
# sample index
# --------------------------------------------------------------------------- #


def build_sample_index(
    pangu: PanguReader,
    era5: ERA5Reader,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    lead_hours: int = 6,
    init_hours: tuple[int, ...] = (0, 12),
) -> pd.DataFrame:
    """Build the single-cycle sample index.

    One row per (init_time, lead_time) pair such that
      * init_time exists in Pangu
      * lead exists in Pangu
      * valid_time = init_time + lead exists in ERA5
      * valid_time within [start, end] (inclusive)
    """
    if not pangu.has_lead(lead_hours):
        raise KeyError(
            f"Pangu lead_time {lead_hours}h does not exist in this dataset; "
            f"available leads: {sorted(set(int(x) for x in pangu.lead_hours))}"
        )
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for init_time in pangu.init_times:
        if init_time.hour not in init_hours:
            continue
        valid_time = init_time + pd.Timedelta(hours=lead_hours)
        if not (start <= valid_time <= end):
            continue
        if not era5.has_time(valid_time):
            continue
        rows.append(
            {
                "sample_id": valid_time.strftime("%Y%m%dT%H"),
                "init_time": init_time,
                "lead_time_hours": lead_hours,
                "valid_time": valid_time,
                "forecast_index": int(pangu.init_times.get_loc(init_time)),
                "analysis_index": int(era5.times.get_loc(valid_time)),
                "observation_window_start": valid_time - pd.Timedelta(minutes=60),
                "observation_window_end": valid_time,
            }
        )
    df = pd.DataFrame(rows).sort_values("valid_time").reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"No aligned samples in [{start}, {end}] with lead={lead_hours}h; "
            "check time overlap between Pangu and ERA5."
        )
    return df


# --------------------------------------------------------------------------- #
# grid helpers
# --------------------------------------------------------------------------- #


def check_uniform_grid(lat: np.ndarray, lon: np.ndarray, tol: float = 1e-4) -> tuple[float, float]:
    """Assert the canonical grid and return (dlat, dlon) in degrees."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    if not np.all(dlat < 0):
        raise ValueError("latitude must be descending in canonical layout")
    if not np.all(dlon > 0):
        raise ValueError("longitude must be ascending in canonical layout")
    if np.ptp(dlat) > tol or np.ptp(dlon) > tol:
        raise ValueError("grid must be uniform (equiangular)")
    return float(-dlat[0]), float(dlon[0])


def normalize_longitude_360(lon):
    """Normalize longitude to [0, 360)."""
    if isinstance(lon, np.ndarray):
        return np.mod(lon.astype(np.float64), 360.0)
    return torch.remainder(lon.double(), 360.0)


def _bilinear_weights_uniform(lat, lon, grid_lat, grid_lon, dlat, dlon):
    """Fractional grid coordinates for a uniform canonical grid.

    lat/lon: query coordinates (lon in [0, 360)); grid_*: canonical axes.
    Returns (lat_idx0, lat_idx1, lat_w, lon_idx0, lon_idx1, lon_w) where
    idx1 indices may wrap in longitude and weights are in [0, 1].
    """
    nlat, nlon = len(grid_lat), len(grid_lon)
    lat0, lon0 = float(grid_lat[0]), float(grid_lon[0])

    lat_f = (lat0 - lat) / dlat          # descending: row 0 at north pole
    lat_f = np.clip(lat_f, 0.0, nlat - 1)
    lat_i0 = np.minimum(np.floor(lat_f).astype(np.int64), nlat - 2)
    lat_i1 = lat_i0 + 1
    lat_w = lat_f - lat_i0               # weight of row i1

    lon_f = (lon - lon0) / dlon
    lon_floor = np.floor(lon_f)
    lon_w = lon_f - lon_floor            # weight of column j1
    lon_i0 = np.mod(lon_floor.astype(np.int64), nlon)  # periodic wrap
    lon_i1 = np.mod(lon_i0 + 1, nlon)
    return lat_i0, lat_i1, lat_w, lon_i0, lon_i1, lon_w


def sample_background_at_obs(
    field: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    obs_lat: np.ndarray,
    obs_lon: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample ``field`` [H, W] at observation locations.

    ``obs_lon`` may be in any convention; it is normalized to [0, 360).
    Returns float64 [N]. NaN at any of the 4 neighbours -> NaN result.
    """
    grid_lat = np.asarray(grid_lat, dtype=np.float64)
    grid_lon = np.asarray(grid_lon, dtype=np.float64)
    obs_lat = np.asarray(obs_lat, dtype=np.float64)
    obs_lon = normalize_longitude_360(np.asarray(obs_lon, dtype=np.float64))
    if obs_lat.shape != obs_lon.shape:
        raise ValueError("obs_lat/obs_lon shape mismatch")
    dlat, dlon = check_uniform_grid(grid_lat, grid_lon)
    if obs_lat.size == 0:
        return np.zeros(0, dtype=np.float64)

    li0, li1, lw, ji0, ji1, jw = _bilinear_weights_uniform(
        obs_lat, obs_lon, grid_lat, grid_lon, dlat, dlon
    )
    f = field.astype(np.float64)
    v00 = f[li0, ji0]
    v01 = f[li0, ji1]
    v10 = f[li1, ji0]
    v11 = f[li1, ji1]
    with np.errstate(invalid="ignore"):
        out = (
            v00 * (1 - lw) * (1 - jw)
            + v01 * (1 - lw) * jw
            + v10 * lw * (1 - jw)
            + v11 * lw * jw
        )
    nan_mask = np.isnan(v00) | np.isnan(v01) | np.isnan(v10) | np.isnan(v11)
    out = np.where(nan_mask, np.nan, out)
    return out


def sample_background_at_obs_torch(
    field: torch.Tensor,
    grid_lat: torch.Tensor,
    grid_lon: torch.Tensor,
    obs_lat: torch.Tensor,
    obs_lon: torch.Tensor,
) -> torch.Tensor:
    """Differentiable bilinear sampling (same math as the numpy version).

    field: [H, W]; grid axes 1-D canonical; obs_*: [N]. NaN neighbours give
    NaN, matching the numpy implementation.
    """
    grid_lat = grid_lat.double()
    grid_lon = grid_lon.double()
    obs_lat = obs_lat.double()
    obs_lon = normalize_longitude_360(obs_lon.double())
    nlat, nlon = grid_lat.numel(), grid_lon.numel()
    dlat = float((grid_lat[0] - grid_lat[1]).item())
    dlon = float((grid_lon[1] - grid_lon[0]).item())

    lat_f = (grid_lat[0] - obs_lat) / dlat
    lat_f = lat_f.clamp(0.0, float(nlat - 1))
    lat_i0 = lat_f.floor().clamp(0, nlat - 2).long()
    lat_i1 = lat_i0 + 1
    lw = lat_f - lat_i0.double()

    lon_f = (obs_lon - grid_lon[0]) / dlon
    lon_floor = lon_f.floor()
    jw = lon_f - lon_floor
    ji0 = lon_floor.long() % nlon
    ji1 = (ji0 + 1) % nlon

    f = field.double()
    v00 = f[lat_i0, ji0]
    v01 = f[lat_i0, ji1]
    v10 = f[lat_i1, ji0]
    v11 = f[lat_i1, ji1]
    out = (
        v00 * (1 - lw) * (1 - jw)
        + v01 * (1 - lw) * jw
        + v10 * lw * (1 - jw)
        + v11 * lw * jw
    )
    nan_mask = torch.isnan(v00) | torch.isnan(v01) | torch.isnan(v10) | torch.isnan(v11)
    return torch.where(nan_mask, torch.full_like(out, float("nan")), out)


def spherical_haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorized; lon in any convention)."""
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1 = np.radians(np.mod(np.asarray(lon1, dtype=np.float64), 360.0))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2 = np.radians(np.mod(np.asarray(lon2, dtype=np.float64), 360.0))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
