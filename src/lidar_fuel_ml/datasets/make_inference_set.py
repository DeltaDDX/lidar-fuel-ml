"""Inference-set assembly scaffold."""

from __future__ import annotations

from ..schemas import FeatureRecord


def make_inference_set(feature_rows: list[FeatureRecord]) -> list[dict[str, float]]:
    """Return model inputs for batch inference."""

    raise NotImplementedError("Inference-set assembly is not implemented yet.")
