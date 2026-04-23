"""Prediction scaffold."""

from __future__ import annotations

from typing import Any


def predict(model: Any, features: list[dict[str, float]]) -> list[float]:
    """Run batch predictions with a trained model."""

    raise NotImplementedError("Prediction is not implemented yet.")
