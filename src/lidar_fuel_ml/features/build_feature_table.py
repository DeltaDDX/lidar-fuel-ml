"""Feature-table assembly scaffold."""

from __future__ import annotations

from ..schemas import DatasetRecord, FeatureRecord


def build_feature_table(dataset_index: list[DatasetRecord]) -> list[FeatureRecord]:
    """Convert dataset records into model-ready feature rows."""

    raise NotImplementedError("Feature table assembly is not implemented yet.")
