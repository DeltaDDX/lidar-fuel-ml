"""Spatially-aware dataset splitting helpers."""

from __future__ import annotations

import math

from ..schemas import FeatureRecord


def split_data(
    records: list[FeatureRecord],
    *,
    n_splits: int = 5,
    block_size: float = 30.0,
) -> list[dict[str, list[FeatureRecord]]]:
    """Create spatial block cross-validation folds from plot-level records."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    block_to_records: dict[tuple[int, int], list[FeatureRecord]] = {}
    for record in records:
        center_x = float(record.metadata.get("plot_center_x", 0.0))
        center_y = float(record.metadata.get("plot_center_y", 0.0))
        block_key = (
            math.floor(center_x / block_size),
            math.floor(center_y / block_size),
        )
        block_to_records.setdefault(block_key, []).append(record)

    blocks = sorted(block_to_records)
    if not blocks:
        return []

    folds = []
    for fold_index in range(min(n_splits, len(blocks))):
        test_blocks = {
            block
            for index, block in enumerate(blocks)
            if index % min(n_splits, len(blocks)) == fold_index
        }
        train = []
        test = []
        for block, block_records in block_to_records.items():
            if block in test_blocks:
                test.extend(block_records)
            else:
                train.extend(block_records)
        if train and test:
            folds.append({"train": train, "test": test})
    return folds
