"""Wall-to-wall CBH raster inference pipeline."""

from __future__ import annotations

from pathlib import Path

from ..datasets import make_inference_set, make_training_set
from ..features import build_feature_table
from ..ingestion import load_las, load_plot_windows_csv
from ..models import predict, train_boosted_trees
from ..preprocessing import classify_ground_ptd, clean_points, extract_plot_clouds, normalize_heights
from ..schemas import FeatureRecord, PlotWindow
from ..visualization.export_maps import export_maps


def run_raster_pipeline(
    point_cloud_path: Path,
    output_path: Path,
    *,
    plot_path: Path | None = None,
    cell_size: float = 20.0,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 4,
    min_samples_leaf: int = 5,
) -> dict[str, object]:
    """Generate wall-to-wall CBH predictions and export an ASCII raster."""

    point_cloud = load_las(point_cloud_path)
    cleaned = clean_points(point_cloud)
    classified = classify_ground_ptd(cleaned)
    normalized = normalize_heights(classified)

    training_rows: list[FeatureRecord] = []
    if plot_path is not None and plot_path.exists():
        plots = load_plot_windows_csv(plot_path)
        training_rows = build_feature_table(extract_plot_clouds(normalized, plots))

    grid_windows = _build_grid_windows(normalized, cell_size=cell_size)
    grid_rows = build_feature_table(extract_plot_clouds(normalized, grid_windows))
    predictions = _predict_cbh(grid_rows, training_rows, n_estimators, learning_rate, max_depth, min_samples_leaf)
    raster_summary = export_maps(
        output_path=output_path,
        feature_rows=grid_rows,
        predictions=predictions,
        cell_size=cell_size,
    )
    return {
        "grid_rows": grid_rows,
        "training_rows": training_rows,
        "raster": raster_summary,
    }


def _predict_cbh(
    grid_rows: list[FeatureRecord],
    training_rows: list[FeatureRecord],
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_samples_leaf: int,
) -> list[float]:
    supervised_rows = [row for row in training_rows if row.target is not None]
    if supervised_rows:
        train_x, train_y, feature_names, _ = make_training_set(supervised_rows)
        inference_x, _, _ = make_inference_set_with_feature_names(grid_rows, feature_names)
        artifact = train_boosted_trees(
            train_x,
            train_y,
            feature_names,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        return predict(artifact.metadata["model"], inference_x)

    return [_proxy_cbh_value(row) for row in grid_rows]


def make_inference_set_with_feature_names(
    feature_rows: list[FeatureRecord],
    feature_names: list[str],
) -> tuple["np.ndarray", list[str], list[str]]:
    import numpy as np

    if not feature_rows:
        return np.empty((0, 0), dtype=float), feature_names, []
    matrix = np.asarray(
        [[float(row.features.get(feature_name, 0.0)) for feature_name in feature_names] for row in feature_rows],
        dtype=float,
    )
    record_ids = [row.record_id or row.scene_id for row in feature_rows]
    return matrix, feature_names, record_ids


def _proxy_cbh_value(row: FeatureRecord) -> float:
    proxies = [
        row.features.get("cbh_proxy_gap", 0.0),
        row.features.get("cbh_proxy_sparse_first_bin", 0.0),
        row.features.get("cbh_proxy_energy_threshold", 0.0),
        row.features.get("cbh_proxy_transition", 0.0),
    ]
    valid = [value for value in proxies if value > 0]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


def _build_grid_windows(point_cloud, *, cell_size: float) -> list[PlotWindow]:
    xs = [point.x for point in point_cloud.points]
    ys = [point.y for point in point_cloud.points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    half = cell_size / 2.0
    windows: list[PlotWindow] = []

    row = 0
    y = min_y + half
    while y <= max_y - half:
        col = 0
        x = min_x + half
        while x <= max_x - half:
            windows.append(
                PlotWindow(
                    plot_id=f"cell_r{row}_c{col}",
                    center_x=x,
                    center_y=y,
                    radius=half,
                    metadata={"grid_cell": True, "cell_size": cell_size},
                )
            )
            x += cell_size
            col += 1
        y += cell_size
        row += 1
    return windows
