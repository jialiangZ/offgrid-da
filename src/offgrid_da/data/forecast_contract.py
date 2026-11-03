"""Boundary conversion for the inspected six-hour GeoArches forecast profile.

The caller supplies a decoded normalized next-state prediction, not backbone
features or loss-scaled deltas. Forecast weights and native statistics stay fixed.
"""
from collections.abc import Mapping

import pandas as pd
import torch

SURFACE_VARIABLES = (
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "2m_temperature", "mean_sea_level_pressure",
)
LEVEL_VARIABLES = (
    "geopotential", "u_component_of_wind", "v_component_of_wind",
    "temperature", "specific_humidity", "vertical_velocity",
)
PRESSURE_LEVELS_HPA = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)


def _validate(prediction: Mapping, stats: Mapping):
    surface, level = prediction["surface"], prediction["level"]
    if surface.ndim != 5 or tuple(surface.shape[1:]) != (4, 1, 121, 240):
        raise ValueError("surface must be [B,4,1,121,240]")
    if tuple(level.shape) != (surface.shape[0], 6, 13, 121, 240):
        raise ValueError("level must be [B,6,13,121,240]")
    if not surface.isfinite().all() or not level.isfinite().all():
        raise ValueError("forecast contains nonfinite state values")
    for key in ("surface_mean", "surface_std", "level_mean", "level_std"):
        expected = (4, 1, 1, 1) if key.startswith("surface") else (6, 13, 1, 1)
        value = stats[key]
        if tuple(value.shape) != expected or not value.isfinite().all():
            raise ValueError(f"invalid forecast statistics: {key}")
        if key.endswith("std") and not (value > 0).all():
            raise ValueError(f"forecast standard deviation must be positive: {key}")
    return surface


def _t2m_stats(stats, reference):
    return (stats["surface_mean"][2, 0, 0, 0].to(device=reference.device, dtype=torch.float32),
            stats["surface_std"][2, 0, 0, 0].to(device=reference.device, dtype=torch.float32))


def archesweather_t2m_background(prediction, stats, init_times, lead_hours=6):
    """Export physical T2M on the DA grid; init_times are timezone-aware UTC.

    Direct one-step forward has lead=6. For an explicitly rolled-out state,
    lead may be a positive multiple of six measured from the original init.
    """
    surface = _validate(prediction, stats)
    if isinstance(lead_hours, bool) or not isinstance(lead_hours, int) or lead_hours <= 0 or lead_hours % 6:
        raise ValueError("lead_hours must be a positive integer multiple of six")
    times = [pd.Timestamp(value) for value in init_times]
    if len(times) != surface.shape[0] or any(t is pd.NaT or t.tzinfo is None for t in times):
        raise ValueError("one explicit timezone-aware init time is required per sample")
    times = [t.tz_convert("UTC") for t in times]
    mean, std = _t2m_stats(stats, surface)
    # Undo GeoArches' half-world longitude roll; latitude is already descending.
    physical = (surface[:, 2].float() * std.float() + mean.float()).roll(-120, -1)
    return {
        "background": physical,
        "grid_latitude": torch.linspace(90, -90, 121, device=surface.device),
        "grid_longitude": torch.arange(240, device=surface.device, dtype=torch.float32) * 1.5,
        "init_times": times,
        "valid_times": [t + pd.Timedelta(hours=lead_hours) for t in times],
        "lead_hours": lead_hours, "variables": ("2m_temperature",), "units": ("K",),
    }


def replace_archesweather_t2m(prediction, analysis_kelvin, stats):
    """Return a cloned full forecast state with only T2M replaced by DA output.

    Input analysis is physical Kelvin [B,1,121,240] in canonical DA orientation.
    Native history state management remains the cycling caller's responsibility.
    """
    surface = _validate(prediction, stats)
    if tuple(analysis_kelvin.shape) != (surface.shape[0], 1, 121, 240):
        raise ValueError("analysis must be [B,1,121,240]")
    if not analysis_kelvin.isfinite().all():
        raise ValueError("analysis must be finite physical Kelvin")
    mean, std = _t2m_stats(stats, surface)
    updated = {key: value.clone() for key, value in prediction.items()}
    normalized = (analysis_kelvin.to(device=surface.device, dtype=torch.float32).roll(120, -1)
                  - mean.float()) / std.float()
    updated["surface"] = surface.clone()
    updated["surface"][:, 2] = normalized.to(surface.dtype)
    return updated
