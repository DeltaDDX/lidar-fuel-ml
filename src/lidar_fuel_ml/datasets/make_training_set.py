"""Training-set assembly scaffold."""

from __future__ import annotations

import numpy as np

from ..schemas import FeatureRecord


def make_training_set(
    feature_rows: list[FeatureRecord],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return numeric model inputs, targets, feature names, and record ids."""

    supervised_rows = [row for row in feature_rows if row.target is not None]
    if not supervised_rows:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float), [], []

    feature_names = sorted(
        {
            feature_name
            for row in supervised_rows
            for feature_name in row.features
        }
    )
    matrix = np.asarray(
        [
            [float(row.features.get(feature_name, 0.0)) for feature_name in feature_names]
            for row in supervised_rows
        ],
        dtype=float,
    )
    targets = np.asarray(
        [float(row.target) for row in supervised_rows if row.target is not None],
        dtype=float,
    )
    record_ids = [row.record_id or row.scene_id for row in supervised_rows]
    return matrix, targets, feature_names, record_ids
