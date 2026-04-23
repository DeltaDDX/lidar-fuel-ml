"""Prediction scaffold."""

from __future__ import annotations

from typing import Any

import numpy as np


def predict(model: Any, features: np.ndarray) -> list[float]:
    """Run batch predictions with a trained model."""

    if hasattr(model, "predict"):
        return list(np.asarray(model.predict(features), dtype=float))
    raise TypeError("Model object does not expose a usable predict method.")
