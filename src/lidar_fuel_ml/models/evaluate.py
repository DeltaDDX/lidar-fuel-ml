"""Evaluation scaffold."""

from __future__ import annotations

import math

import numpy as np

def evaluate_predictions(predictions: list[float], targets: list[float]) -> dict[str, float]:
    """Compute evaluation metrics for model output."""

    if len(predictions) != len(targets):
        raise ValueError("Prediction and target lengths must match.")

    predictions_array = np.asarray(predictions, dtype=float)
    targets_array = np.asarray(targets, dtype=float)
    if targets_array.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "bias": 0.0, "r2": 0.0}

    errors = predictions_array - targets_array
    mae = float(np.mean(np.abs(errors)))
    rmse = math.sqrt(float(np.mean(errors * errors)))
    bias = float(np.mean(errors))
    target_mean = float(np.mean(targets_array))
    ss_res = float(np.sum(errors * errors))
    ss_tot = float(np.sum((targets_array - target_mean) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else 0.0
    return {"mae": mae, "rmse": rmse, "bias": bias, "r2": r2}
