"""Feature pipeline scaffold."""

from __future__ import annotations

from pathlib import Path

from ..features import build_feature_table
from ..ingestion import load_las, load_plot_windows_csv
from ..preprocessing import (
    classify_ground_ptd,
    clean_points,
    extract_plot_clouds,
    normalize_heights,
)
from ..schemas import FeatureRecord


def run_feature_pipeline(point_cloud_path: Path, plot_path: Path) -> list[FeatureRecord]:
    """Run the baseline ingestion-to-features workflow for one scene."""

    point_cloud = load_las(point_cloud_path)
    plots = load_plot_windows_csv(plot_path)
    cleaned = clean_points(point_cloud)
    classified = classify_ground_ptd(cleaned)
    normalized = normalize_heights(classified)
    plot_clouds = extract_plot_clouds(normalized, plots)
    return build_feature_table(plot_clouds)
