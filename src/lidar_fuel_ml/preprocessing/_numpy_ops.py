"""Helpers for working with point-cloud records as numpy arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..schemas import PointCloud, PointRecord


def point_cloud_to_arrays(point_cloud: PointCloud) -> dict[str, np.ndarray]:
    """Convert a point cloud into column arrays."""

    if not point_cloud.points:
        return {
            "x": np.empty((0,), dtype=float),
            "y": np.empty((0,), dtype=float),
            "z": np.empty((0,), dtype=float),
            "classification": np.empty((0,), dtype=float),
            "intensity": np.empty((0,), dtype=float),
            "return_number": np.empty((0,), dtype=float),
            "num_returns": np.empty((0,), dtype=float),
        }

    return {
        "x": np.asarray([point.x for point in point_cloud.points], dtype=float),
        "y": np.asarray([point.y for point in point_cloud.points], dtype=float),
        "z": np.asarray([point.z for point in point_cloud.points], dtype=float),
        "classification": np.asarray(
            [np.nan if point.classification is None else point.classification for point in point_cloud.points],
            dtype=float,
        ),
        "intensity": np.asarray(
            [np.nan if point.intensity is None else point.intensity for point in point_cloud.points],
            dtype=float,
        ),
        "return_number": np.asarray(
            [np.nan if point.return_number is None else point.return_number for point in point_cloud.points],
            dtype=float,
        ),
        "num_returns": np.asarray(
            [np.nan if point.num_returns is None else point.num_returns for point in point_cloud.points],
            dtype=float,
        ),
    }


def rebuild_point_cloud(
    point_cloud: PointCloud,
    *,
    z: np.ndarray | None = None,
    classification: np.ndarray | None = None,
    metadata_overrides: list[dict[str, Any]] | None = None,
    cloud_metadata: dict[str, Any] | None = None,
) -> PointCloud:
    """Rebuild a point cloud using updated numeric arrays."""

    z_values = z if z is not None else np.asarray([point.z for point in point_cloud.points], dtype=float)
    classification_values = (
        classification
        if classification is not None
        else np.asarray(
            [np.nan if point.classification is None else point.classification for point in point_cloud.points],
            dtype=float,
        )
    )
    metadata_values = metadata_overrides or [{} for _ in point_cloud.points]

    rebuilt_points = []
    for index, point in enumerate(point_cloud.points):
        classification_value = classification_values[index]
        rebuilt_points.append(
            PointRecord(
                x=point.x,
                y=point.y,
                z=float(z_values[index]),
                classification=None if np.isnan(classification_value) else int(classification_value),
                intensity=point.intensity,
                return_number=point.return_number,
                num_returns=point.num_returns,
                metadata={**point.metadata, **metadata_values[index]},
            )
        )

    return PointCloud(
        scene_id=point_cloud.scene_id,
        points=tuple(rebuilt_points),
        crs=point_cloud.crs,
        metadata={**point_cloud.metadata, **(cloud_metadata or {})},
    )
