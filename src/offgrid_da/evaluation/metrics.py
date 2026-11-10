"""Fixed evaluation metrics for grid fields."""
from __future__ import annotations

import numpy as np


def _area_weighted_error_sums(
    prediction: np.ndarray, reference: np.ndarray, latitude: np.ndarray
) -> tuple[float, float, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    latitude = np.asarray(latitude, dtype=np.float64).reshape(-1)
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference must have identical shapes")
    if prediction.ndim < 2 or prediction.shape[-2] != latitude.size:
        raise ValueError("latitude length must match the second-to-last field dimension")

    base_weights = np.maximum(np.cos(np.deg2rad(latitude)), 0.0)
    weight_shape = (1,) * (prediction.ndim - 2) + (latitude.size, 1)
    weights = np.broadcast_to(base_weights.reshape(weight_shape), prediction.shape)
    valid = np.isfinite(prediction) & np.isfinite(reference)
    effective_weights = np.where(valid, weights, 0.0)
    weight_sum = float(effective_weights.sum(dtype=np.float64))
    if weight_sum <= 0.0:
        raise ValueError("no finite grid cells with positive area weight")

    error = np.where(valid, prediction - reference, 0.0)
    squared_error_sum = float((effective_weights * error * error).sum(dtype=np.float64))
    absolute_error_sum = float((effective_weights * np.abs(error)).sum(dtype=np.float64))
    return squared_error_sum, absolute_error_sum, weight_sum


def area_weighted_rmse(
    prediction: np.ndarray, reference: np.ndarray, latitude: np.ndarray
) -> float:
    """Cos(latitude)-weighted RMSE over all leading dimensions and grid cells."""
    squared_error_sum, _, weight_sum = _area_weighted_error_sums(
        prediction, reference, latitude
    )
    return float(np.sqrt(squared_error_sum / weight_sum))


def area_weighted_mae(
    prediction: np.ndarray, reference: np.ndarray, latitude: np.ndarray
) -> float:
    """Cos(latitude)-weighted MAE over all leading dimensions and grid cells."""
    _, absolute_error_sum, weight_sum = _area_weighted_error_sums(
        prediction, reference, latitude
    )
    return float(absolute_error_sum / weight_sum)
