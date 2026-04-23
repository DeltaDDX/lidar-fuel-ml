"""Training-set assembly scaffold."""

from __future__ import annotations

from ..schemas import FeatureRecord


def make_training_set(feature_rows: list[FeatureRecord]) -> tuple[list[dict[str, float]], list[float]]:
    """Return model inputs and targets for supervised training."""

    raise NotImplementedError("Training-set assembly is not implemented yet.")
