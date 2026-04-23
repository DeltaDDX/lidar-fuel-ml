"""Inference-set assembly scaffold."""

from __future__ import annotations

import numpy as np

from ..schemas import FeatureRecord


def make_inference_set(
    feature_rows: list[FeatureRecord],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return numeric model inputs, feature names, and record ids."""

    if not feature_rows:
        return np.empty((0, 0), dtype=float), [], []

    feature_names = sorted(
        {
            feature_name
            for row in feature_rows
            for feature_name in row.features
        }
    )
    matrix = np.asarray(
        [
            [float(row.features.get(feature_name, 0.0)) for feature_name in feature_names]
            for row in feature_rows
        ],
        dtype=float,
    )
    record_ids = [row.record_id or row.scene_id for row in feature_rows]
    return matrix, feature_names, record_ids
