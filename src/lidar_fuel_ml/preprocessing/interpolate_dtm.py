"""Ground model interpolation utilities."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from ..schemas import PointCloud, PointRecord


def build_ground_model(point_cloud: PointCloud) -> tuple[PointRecord, ...]:
    """Return ground-classified points used to interpolate a DTM."""

    return tuple(point for point in point_cloud.points if point.classification == 2)


def estimate_ground_elevation(
    x: float,
    y: float,
    ground_points: tuple[PointRecord, ...],
    *,
    search_radius: float = 15.0,
    power: float = 2.0,
) -> float:
    """Estimate ground elevation with inverse-distance weighting."""

    if not ground_points:
        raise ValueError("At least one ground point is required to estimate a DTM.")

    ground_xy = np.asarray([(point.x, point.y) for point in ground_points], dtype=float)
    ground_z = np.asarray([point.z for point in ground_points], dtype=float)
    return float(
        estimate_ground_elevations(
            np.asarray([[x, y]], dtype=float),
            ground_xy,
            ground_z,
            search_radius=search_radius,
            power=power,
        )[0]
    )


def estimate_ground_elevations(
    query_xy: np.ndarray,
    ground_xy: np.ndarray,
    ground_z: np.ndarray,
    *,
    search_radius: float = 20.0,
    power: float = 2.0,
    k_neighbors: int = 8,
) -> np.ndarray:
    """Estimate ground elevation for many XY coordinates with k-NN IDW."""

    if ground_xy.size == 0 or ground_z.size == 0:
        raise ValueError("Ground arrays must be non-empty.")

    tree = cKDTree(ground_xy)
    k = max(1, min(k_neighbors, ground_xy.shape[0]))
    distances, neighbor_positions = tree.query(
        query_xy,
        k=k,
        distance_upper_bound=search_radius,
    )
    if k == 1:
        distances = distances[:, None]
        neighbor_positions = neighbor_positions[:, None]

    valid = np.isfinite(distances) & (neighbor_positions < ground_xy.shape[0])
    nearest_positions = np.clip(neighbor_positions[:, 0], 0, ground_xy.shape[0] - 1)
    exact_match = valid & (distances == 0)
    with np.errstate(divide="ignore"):
        weights = np.where(valid, 1.0 / np.power(distances, power), 0.0)
    weights[exact_match] = 0.0

    neighbor_ground = ground_z[np.clip(neighbor_positions, 0, ground_z.shape[0] - 1)]
    weighted_sum = np.sum(weights * neighbor_ground, axis=1)
    weight_total = weights.sum(axis=1)
    estimates = np.divide(
        weighted_sum,
        weight_total,
        out=ground_z[nearest_positions].astype(float).copy(),
        where=weight_total > 0,
    )

    exact_rows, exact_cols = np.where(exact_match)
    if exact_rows.size:
        estimates[exact_rows] = neighbor_ground[exact_rows, exact_cols]
    return estimates


def interpolate_dtm_grid(
    ground_points: tuple[PointRecord, ...],
    *,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    cell_size: float,
    smoothing_sigma: float = 1.0,
) -> list[dict[str, float]]:
    """Interpolate a coarse DTM grid for QA or export."""

    x_values = np.arange(min_x, max_x + cell_size, cell_size, dtype=float)
    y_values = np.arange(min_y, max_y + cell_size, cell_size, dtype=float)
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="xy")
    query_xy = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    ground_xy = np.asarray([(point.x, point.y) for point in ground_points], dtype=float)
    ground_z = np.asarray([point.z for point in ground_points], dtype=float)
    elevations = estimate_ground_elevations(query_xy, ground_xy, ground_z)
    grid = elevations.reshape(grid_y.shape)
    if smoothing_sigma > 0:
        grid = gaussian_filter(grid, sigma=smoothing_sigma, mode="nearest")

    cells = []
    for (x_value, y_value), elevation in zip(query_xy, grid.ravel()):
        cells.append({"x": float(x_value), "y": float(y_value), "ground_z": float(elevation)})
    return cells
