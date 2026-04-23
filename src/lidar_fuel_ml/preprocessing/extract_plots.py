"""Plot extraction utilities."""

from __future__ import annotations

import numpy as np

from ..schemas import PlotPointCloud, PlotWindow, PointCloud
from ._numpy_ops import point_cloud_to_arrays


def extract_plot_clouds(point_cloud: PointCloud, plots: list[PlotWindow]) -> list[PlotPointCloud]:
    """Extract circular plot subsets from a normalized point cloud."""

    arrays = point_cloud_to_arrays(point_cloud)
    xy = np.column_stack((arrays["x"], arrays["y"])) if len(point_cloud.points) else np.empty((0, 2), dtype=float)
    extracted = []
    for plot in plots:
        radius_sq = plot.radius * plot.radius
        deltas = xy - np.asarray([[plot.center_x, plot.center_y]], dtype=float)
        mask = np.sum(deltas * deltas, axis=1) <= radius_sq if xy.size else np.empty((0,), dtype=bool)
        selected_indices = np.flatnonzero(mask)
        points = tuple(point_cloud.points[index] for index in selected_indices)
        point_density = len(points) / plot.area if plot.area else 0.0
        extracted.append(
            PlotPointCloud(
                plot=plot,
                scene_id=point_cloud.scene_id,
                points=points,
                metadata={
                    "point_density": point_density,
                    "plot_radius": plot.radius,
                    "plot_center_x": plot.center_x,
                    "plot_center_y": plot.center_y,
                    "plot_area": plot.area,
                },
            )
        )
    return extracted
