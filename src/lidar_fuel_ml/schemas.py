"""Typed schema objects shared across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetRecord:
    """Metadata for a single LiDAR scene or tile."""

    scene_id: str
    source_path: Path
    label_path: Path | None = None
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureRecord:
    """Feature table row keyed to a dataset record."""

    scene_id: str
    features: dict[str, float]
    target: float | None = None


@dataclass(frozen=True)
class ModelArtifact:
    """Reference to a trained model or related evaluation artifact."""

    model_name: str
    artifact_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
