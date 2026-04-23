"""Dataset split scaffold."""

from __future__ import annotations

from ..schemas import DatasetRecord


def split_data(records: list[DatasetRecord]) -> dict[str, list[DatasetRecord]]:
    """Partition records into train/validation/test splits."""

    raise NotImplementedError("Dataset splitting is not implemented yet.")
