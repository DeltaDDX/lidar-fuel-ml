"""Dataset indexing helpers."""

from __future__ import annotations

from pathlib import Path

from ..schemas import DatasetRecord


def build_dataset_index(data_root: Path) -> list[DatasetRecord]:
    """Build a manifest of available source files.

    The final implementation should walk the raw-data directories and emit one
    dataset record per scene or tile.
    """

    raise NotImplementedError("Dataset indexing is not implemented yet.")
