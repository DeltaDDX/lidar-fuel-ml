"""Height normalization scaffold."""

from __future__ import annotations

import numpy as np

from ..schemas import PointCloud
from ._numpy_ops import point_cloud_to_arrays, rebuild_point_cloud
from .interpolate_dtm import build_ground_model, estimate_ground_elevations


def normalize_heights(point_cloud: PointCloud) -> PointCloud:
    """Normalize point elevations against an interpolated ground surface."""

    ground_points = build_ground_model(point_cloud)
    arrays = point_cloud_to_arrays(point_cloud)
    query_xy = np.column_stack((arrays["x"], arrays["y"]))
    ground_xy = np.asarray([(point.x, point.y) for point in ground_points], dtype=float)
    ground_z = np.asarray([point.z for point in ground_points], dtype=float)
    estimated_ground = estimate_ground_elevations(query_xy, ground_xy, ground_z)
    normalized_z = arrays["z"] - estimated_ground
    metadata_overrides = [{"ground_z": float(value)} for value in estimated_ground]

    return rebuild_point_cloud(
        point_cloud,
        z=normalized_z,
        metadata_overrides=metadata_overrides,
        cloud_metadata={"height_normalized": True},
    )
