"""Typed schema objects shared across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PointRecord:
    """A single point from a point cloud."""

    x: float
    y: float
    z: float
    classification: int | None = None
    intensity: float | None = None
    return_number: int | None = None
    num_returns: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PointCloud:
    """A collection of point records tied to a source scene."""

    scene_id: str
    points: tuple[PointRecord, ...]
    crs: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.points)


@dataclass(frozen=True)
class PlotWindow:
    """A circular field plot definition used for point extraction."""

    plot_id: str
    center_x: float
    center_y: float
    radius: float
    target_cbh: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def area(self) -> float:
        return math.pi * self.radius * self.radius


@dataclass(frozen=True)
class PlotPointCloud:
    """Point subset extracted for a single plot."""

    plot: PlotWindow
    scene_id: str
    points: tuple[PointRecord, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.points)


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
    record_id: str | None = None
    target: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelArtifact:
    """Reference to a trained model or related evaluation artifact."""

    model_name: str
    artifact_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
