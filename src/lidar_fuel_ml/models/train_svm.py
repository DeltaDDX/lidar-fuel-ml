"""SVM training scaffold."""

from __future__ import annotations

from ..schemas import ModelArtifact


def train_svm(features: list[dict[str, float]], targets: list[float]) -> ModelArtifact:
    """Train an SVM model and persist its artifact."""

    raise NotImplementedError("SVM training is not implemented yet.")
