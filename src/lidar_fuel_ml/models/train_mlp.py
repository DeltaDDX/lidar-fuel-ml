"""MLP training scaffold."""

from __future__ import annotations

from ..schemas import ModelArtifact


def train_mlp(features: list[dict[str, float]], targets: list[float]) -> ModelArtifact:
    """Train a multilayer perceptron model and persist its artifact."""

    raise NotImplementedError("MLP training is not implemented yet.")
