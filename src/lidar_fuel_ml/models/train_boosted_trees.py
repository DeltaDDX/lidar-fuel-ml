"""Boosted-tree regression using scikit-learn."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from ..schemas import ModelArtifact


def train_boosted_trees(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str],
    *,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 4,
    min_samples_leaf: int = 5,
    l2_regularization: float = 0.0,
    random_state: int = 42,
) -> ModelArtifact:
    """Train a histogram-based gradient boosting regressor."""

    if features.size == 0 or targets.size == 0:
        raise ValueError("Training requires at least one feature row and target.")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("Feature and target lengths must match.")

    model = HistGradientBoostingRegressor(
        max_iter=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )
    with threadpool_limits(limits=1):
        model.fit(features, targets)

    return ModelArtifact(
        model_name="hist_gradient_boosting",
        artifact_path=Path("models") / "trained" / "hist_gradient_boosting.joblib",
        metadata={
            "model": model,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "feature_names": feature_names,
        },
    )
