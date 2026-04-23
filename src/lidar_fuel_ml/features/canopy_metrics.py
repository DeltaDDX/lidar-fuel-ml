"""Canopy and CBH-proxy metrics."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..schemas import PlotPointCloud


def compute_plot_summary_metrics(plot_cloud: PlotPointCloud) -> dict[str, float]:
    """Compute basic plot-level metrics from normalized points."""

    if not plot_cloud.points:
        return {
            "num_points": 0.0,
            "mean_height": 0.0,
            "max_height": 0.0,
            "p95_height": 0.0,
            "canopy_cover_2m": 0.0,
        }

    heights = np.sort(np.asarray([max(point.z, 0.0) for point in plot_cloud.points], dtype=float))
    return {
        "num_points": float(heights.size),
        "mean_height": float(np.mean(heights)),
        "max_height": float(heights[-1]),
        "p95_height": _percentile(heights, 0.95),
        "canopy_cover_2m": float(np.mean(heights >= 2.0)),
    }


def compute_cbh_proxies(plot_cloud: PlotPointCloud) -> dict[str, float]:
    """Compute weak CBH proxy features from a normalized plot cloud."""

    if not plot_cloud.points:
        return {
            "cbh_proxy_gap": 0.0,
            "cbh_proxy_sparse_first_bin": 0.0,
            "cbh_proxy_energy_threshold": 0.0,
            "cbh_proxy_transition": 0.0,
        }

    heights = np.asarray([max(point.z, 0.0) for point in plot_cloud.points], dtype=float)
    canopy_points = heights[heights > 0]
    if canopy_points.size == 0:
        return {
            "cbh_proxy_gap": 0.0,
            "cbh_proxy_sparse_first_bin": 0.0,
            "cbh_proxy_energy_threshold": 0.0,
            "cbh_proxy_transition": 0.0,
        }

    profile = _profile_density(canopy_points, bin_size=0.5, max_height=max(24.0, float(np.max(canopy_points)) + 0.5))
    smoothed = gaussian_filter1d(profile["density"], sigma=1.0, mode="nearest")
    heights_axis = profile["heights"]

    cbh_gap = _first_sustained_density_height(
        heights_axis,
        smoothed,
        min_density=0.03,
        sustain_bins=2,
    )
    cbh_sparse = _first_canopy_after_empty_understory(
        heights_axis,
        smoothed,
        low_density=0.015,
        canopy_density=0.04,
        minimum_empty_bins=2,
    )
    cbh_energy = _height_at_cumulative_density(heights_axis, smoothed, threshold=0.15)
    cbh_transition = _largest_profile_gradient_height(heights_axis, smoothed)
    return {
        "cbh_proxy_gap": cbh_gap,
        "cbh_proxy_sparse_first_bin": cbh_sparse,
        "cbh_proxy_energy_threshold": cbh_energy,
        "cbh_proxy_transition": cbh_transition,
    }


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def _profile_density(values: np.ndarray, *, bin_size: float, max_height: float) -> dict[str, np.ndarray]:
    edges = np.arange(0.0, max_height + bin_size, bin_size, dtype=float)
    counts, _ = np.histogram(values, bins=edges)
    density = counts / max(1, values.size)
    heights = edges[:-1]
    return {"heights": heights, "density": density.astype(float)}


def _first_sustained_density_height(
    heights: np.ndarray,
    density: np.ndarray,
    *,
    min_density: float,
    sustain_bins: int,
) -> float:
    run = 0
    for index, value in enumerate(density):
        run = run + 1 if value >= min_density else 0
        if run >= sustain_bins:
            return float(heights[index - sustain_bins + 1])
    return float(heights[-1]) if heights.size else 0.0


def _first_canopy_after_empty_understory(
    heights: np.ndarray,
    density: np.ndarray,
    *,
    low_density: float,
    canopy_density: float,
    minimum_empty_bins: int,
) -> float:
    empty_run = 0
    for index, value in enumerate(density):
        if value <= low_density:
            empty_run += 1
            continue
        if empty_run >= minimum_empty_bins and value >= canopy_density:
            return float(heights[index])
        empty_run = 0
    return float(heights[-1]) if heights.size else 0.0


def _height_at_cumulative_density(
    heights: np.ndarray,
    density: np.ndarray,
    *,
    threshold: float,
) -> float:
    total = float(np.sum(density))
    if total <= 0:
        return 0.0
    cumulative = np.cumsum(density) / total
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    index = min(index, heights.size - 1)
    return float(heights[index])


def _largest_profile_gradient_height(heights: np.ndarray, density: np.ndarray) -> float:
    if heights.size == 0:
        return 0.0
    gradient = np.gradient(density)
    index = int(np.argmax(gradient))
    return float(heights[index])
