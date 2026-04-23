"""Shared filesystem path helpers for the project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved paths used across ingestion, training, and reporting."""

    project_root: Path
    configs_dir: Path
    data_dir: Path
    raw_data_dir: Path
    interim_data_dir: Path
    processed_data_dir: Path
    sample_data_dir: Path
    models_dir: Path
    reports_dir: Path


def get_project_paths(start: Path | None = None) -> ProjectPaths:
    """Resolve project directories relative to the package location."""

    anchor = start or Path(__file__).resolve()
    project_root = anchor.parents[2]
    data_dir = project_root / "data"

    return ProjectPaths(
        project_root=project_root,
        configs_dir=project_root / "configs",
        data_dir=data_dir,
        raw_data_dir=data_dir / "raw",
        interim_data_dir=data_dir / "interim",
        processed_data_dir=data_dir / "processed",
        sample_data_dir=data_dir / "sample",
        models_dir=project_root / "models",
        reports_dir=project_root / "reports",
    )
