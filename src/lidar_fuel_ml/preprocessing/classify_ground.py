"""Progressive densification style ground classification."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..schemas import PointCloud
from ._numpy_ops import point_cloud_to_arrays, rebuild_point_cloud


def classify_ground_ptd(
    point_cloud: PointCloud,
    *,
    seed_cell_size: float = 10.0,
    min_cell_size: float = 2.5,
    base_height_threshold: float = 0.25,
    max_height_threshold: float = 1.25,
    slope_threshold: float = 0.15,
    neighbor_count: int = 8,
) -> PointCloud:
    """Classify likely ground points with a multi-pass PTD-style filter.

    The workflow is:
    1. Seed one lowest point per coarse cell.
    2. Iteratively halve the cell size.
    3. At each stage, compare candidate points to nearby accepted ground points.
       Candidates are accepted if they remain close to the local ground surface
       under both a height and slope constraint.
    """

    if not point_cloud.points:
        return point_cloud

    arrays = point_cloud_to_arrays(point_cloud)
    x = arrays["x"]
    y = arrays["y"]
    z = arrays["z"]
    classifications = arrays["classification"].copy()

    sort_index = np.argsort(z, kind="stable")
    accepted_mask = np.zeros(x.shape[0], dtype=bool)

    coarse_seeds = _lowest_indices_by_cell(x, y, z, seed_cell_size)
    accepted_mask[coarse_seeds] = True

    cell_size = seed_cell_size
    while cell_size >= min_cell_size:
        candidate_indices = _lowest_indices_by_cell(x, y, z, cell_size)
        accepted_mask = _accept_candidates(
            x,
            y,
            z,
            accepted_mask,
            candidate_indices[~accepted_mask[candidate_indices]],
            base_height_threshold=base_height_threshold,
            max_height_threshold=max_height_threshold,
            slope_threshold=slope_threshold,
            neighbor_count=neighbor_count,
        )
        cell_size /= 2.0

    accepted_mask = _accept_candidates(
        x,
        y,
        z,
        accepted_mask,
        sort_index[~accepted_mask[sort_index]],
        base_height_threshold=base_height_threshold,
        max_height_threshold=max_height_threshold,
        slope_threshold=slope_threshold,
        neighbor_count=neighbor_count,
    )

    classifications[accepted_mask] = 2.0
    return rebuild_point_cloud(
        point_cloud,
        classification=classifications,
        cloud_metadata={"ground_classifier": "progressive_densification_numpy"},
    )


def _lowest_indices_by_cell(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell_size: float,
) -> np.ndarray:
    cell_x = np.floor_divide(x, cell_size).astype(int)
    cell_y = np.floor_divide(y, cell_size).astype(int)
    cell_keys = np.column_stack((cell_x, cell_y))
    order = np.argsort(z, kind="stable")
    sorted_keys = cell_keys[order]
    _, unique_positions = np.unique(sorted_keys, axis=0, return_index=True)
    return order[np.sort(unique_positions)]


def _accept_candidates(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    accepted_mask: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    base_height_threshold: float,
    max_height_threshold: float,
    slope_threshold: float,
    neighbor_count: int,
) -> np.ndarray:
    if candidate_indices.size == 0 or not np.any(accepted_mask):
        return accepted_mask

    ground_xy = np.column_stack((x[accepted_mask], y[accepted_mask]))
    ground_z = z[accepted_mask]
    tree = cKDTree(ground_xy)

    query_xy = np.column_stack((x[candidate_indices], y[candidate_indices]))
    k = max(1, min(neighbor_count, ground_xy.shape[0]))
    distances, neighbor_positions = tree.query(query_xy, k=k)
    if k == 1:
        distances = distances[:, None]
        neighbor_positions = neighbor_positions[:, None]

    local_ground = np.median(ground_z[neighbor_positions], axis=1)
    nearest_distance = distances[:, 0]
    allowed_height = np.clip(
        base_height_threshold + slope_threshold * nearest_distance,
        base_height_threshold,
        max_height_threshold,
    )
    candidate_heights = z[candidate_indices] - local_ground
    newly_ground = candidate_indices[candidate_heights <= allowed_height]
    updated_mask = accepted_mask.copy()
    updated_mask[newly_ground] = True
    return updated_mask
