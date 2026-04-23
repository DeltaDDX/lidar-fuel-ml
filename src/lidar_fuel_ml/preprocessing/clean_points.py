"""Point-cleaning scaffold."""

from __future__ import annotations

import math

from ..schemas import PointCloud, PointRecord


def clean_points(point_cloud: PointCloud, *, min_z: float | None = None) -> PointCloud:
    """Remove invalid or noisy points before feature generation."""

    cleaned = []
    for point in point_cloud.points:
        if not (math.isfinite(point.x) and math.isfinite(point.y) and math.isfinite(point.z)):
            continue
        if min_z is not None and point.z < min_z:
            continue
        cleaned.append(point)

    return PointCloud(
        scene_id=point_cloud.scene_id,
        points=tuple(cleaned),
        crs=point_cloud.crs,
        metadata={**point_cloud.metadata, "cleaned": True},
    )
