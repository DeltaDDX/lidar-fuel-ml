"""Random forest training scaffold."""

from __future__ import annotations

from ..schemas import ModelArtifact


def train_random_forest(features: list[dict[str, float]], targets: list[float]) -> ModelArtifact:
    """Train a random forest model and persist its artifact."""

    raise NotImplementedError("Random forest training is not implemented yet.")
