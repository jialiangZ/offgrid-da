"""Encode padded, irregular scalar observations as attention-ready tokens."""
from __future__ import annotations

import torch
from torch import nn


class ObservationEncoder(nn.Module):
    """Encode collated observations without projecting them onto a grid.

    Inputs follow single_cycle_collate: locations are (latitude, longitude,
    elevation metres), offsets are minutes, and values should be normalized
    with training-set statistics. Missing values and padding are masked.
    """

    def __init__(self, dim: int = 128, num_variables: int = 32,
                 num_sources: int = 16, num_quality_codes: int = 16,
                 value_mean: float = 0., value_std: float = 1.,
                 time_scale_minutes: float = 60., use_innovation: bool = True):
        super().__init__()
        if dim <= 0 or value_std <= 0 or time_scale_minutes <= 0:
            raise ValueError("dim, value_std, and time_scale_minutes must be positive")
        self.register_buffer("value_mean", torch.tensor(float(value_mean)))
        self.register_buffer("value_std", torch.tensor(float(value_std)))
        self.time_scale_minutes = float(time_scale_minutes)
        self.variable = nn.Embedding(num_variables, dim)
        self.source = nn.Embedding(num_sources, dim)
        self.quality = nn.Embedding(num_quality_codes, dim)
        self.value = nn.Linear(1, dim)
        self.location = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.time = nn.Linear(1, dim)
        self.innovation = nn.Linear(1, dim) if use_innovation else None
        self.fusion = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                                    nn.GELU(), nn.Linear(dim, dim))

    def forward(self, batch: dict[str, torch.Tensor], prefix: str = "obs") -> dict[str, torch.Tensor]:
        get = lambda key: batch.get(f"{prefix}_{key}")
        # Dataset normalizations are fitted on the training split; consume
        # them directly when present to avoid per-batch/statistic leakage.
        values = get("values_normalized")
        if values is None:
            values = get("values")
        loc = get("locations")
        if values is None or values.ndim != 2:
            raise ValueError(f"{prefix}_values must have shape [batch, observations]")
        if loc is None or loc.shape != (*values.shape, 3):
            raise ValueError(f"{prefix}_locations must have shape [batch, observations, 3]")
        mask = batch.get("observation_mask" if prefix == "obs" else f"{prefix}_mask")
        if mask is None:
            mask = torch.ones_like(values, dtype=torch.bool)
        if mask.shape != values.shape:
            raise ValueError("observation mask must match values")
        mask = mask.bool() & values.isfinite() & loc.isfinite().all(-1)
        safe_values = torch.where(mask, values, self.value_mean.to(values))
        # Sanitize before any differentiable operation: masking outputs alone
        # cannot prevent NaN parameter gradients from invalid input records.
        safe_loc = torch.where(mask.unsqueeze(-1), loc, torch.zeros_like(loc))
        lat, lon, elevation = safe_loc.unbind(-1)
        lat, lon = torch.deg2rad(lat), torch.deg2rad(lon)
        xyz = torch.stack((lat.cos()*lon.cos(), lat.cos()*lon.sin(), lat.sin()), -1)
        elevation = torch.nan_to_num(elevation) / 1000
        offsets = get("time_offsets")
        offsets = torch.zeros_like(values) if offsets is None else offsets
        mask = mask & offsets.isfinite()
        offsets = torch.where(mask, offsets, torch.zeros_like(offsets))
        if get("values_normalized") is None:
            safe_values = (safe_values - self.value_mean.to(values)) / self.value_std.to(values)
        token = self.value(safe_values.unsqueeze(-1))
        token += self.location(torch.cat((xyz, elevation.unsqueeze(-1)), -1))
        token += self.time((offsets/self.time_scale_minutes).unsqueeze(-1))
        for key, embedding in (("variables", self.variable), ("sources", self.source), ("quality", self.quality)):
            ids = get(key)
            if ids is not None:
                token += embedding(ids.long().clamp(0, embedding.num_embeddings-1))
        innovation_mask = torch.zeros_like(mask)
        innovation = get("innovation_normalized")
        if innovation is None:
            innovation = get("innovation")
        if self.innovation is not None and innovation is not None:
            innovation_mask = innovation.isfinite() & mask
            token += self.innovation(torch.nan_to_num(innovation).unsqueeze(-1))
        tokens = self.fusion(token).masked_fill(~mask.unsqueeze(-1), 0)
        return {"tokens": tokens, "observation_mask": mask, "innovation_mask": innovation_mask}
