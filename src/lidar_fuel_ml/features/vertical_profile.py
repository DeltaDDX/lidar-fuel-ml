"""Vertical profile and occupancy features."""

from __future__ import annotations

import numpy as np

from ..schemas import PlotPointCloud


def compute_vertical_profile(
    plot_cloud: PlotPointCloud,
    bins: int = 12,
    max_profile_height: float = 24.0,
) -> dict[str, float]:
    """Compute vertical and occupancy profiles from normalized heights."""

    if not plot_cloud.points:
        return {f"vertical_bin_{index}": 0.0 for index in range(bins)} | {
            f"occupancy_bin_{index}": 0.0 for index in range(bins)
        }

    heights = np.asarray([max(point.z, 0.0) for point in plot_cloud.points], dtype=float)
    clipped = np.clip(heights, 0.0, max_profile_height)
    edges = np.linspace(0.0, max_profile_height, bins + 1, dtype=float)
    counts, _ = np.histogram(clipped, bins=edges)
    occupancy = (counts > 0).astype(float)
    total = max(1, clipped.size)
    features = {f"vertical_bin_{index}": float(counts[index] / total) for index in range(bins)}
    features.update({f"occupancy_bin_{index}": float(occupancy[index]) for index in range(bins)})
    return features
