"""Training pipeline scaffold."""

from __future__ import annotations

from pathlib import Path

from ..datasets import make_training_set, split_data
from ..models import evaluate_predictions, predict, train_boosted_trees
from ..pipelines.run_feature_pipeline import run_feature_pipeline


def run_training_pipeline(
    point_cloud_path: Path,
    plot_path: Path,
    *,
    n_splits: int = 3,
    block_size: float = 30.0,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 4,
    min_samples_leaf: int = 5,
) -> dict[str, object]:
    """Run feature generation, boosted-tree training, and spatial CV."""

    feature_rows = run_feature_pipeline(point_cloud_path, plot_path)
    folds = split_data(feature_rows, n_splits=n_splits, block_size=block_size)
    fold_metrics = []

    for fold in folds:
        train_x, train_y, feature_names, train_ids = make_training_set(fold["train"])
        test_x, test_y, _, test_ids = make_training_set(fold["test"])
        artifact = train_boosted_trees(
            train_x,
            train_y,
            feature_names,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        model = artifact.metadata["model"]
        predictions = predict(model, test_x)
        metrics = evaluate_predictions(predictions, test_y)
        fold_metrics.append(
            {
                "metrics": metrics,
                "train_ids": train_ids,
                "test_ids": test_ids,
                "feature_names": feature_names,
            }
        )

    overall = _average_metrics([fold["metrics"] for fold in fold_metrics])
    return {
        "feature_rows": feature_rows,
        "folds": fold_metrics,
        "overall_metrics": overall,
    }


def _average_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_list:
        return {"mae": 0.0, "rmse": 0.0, "bias": 0.0, "r2": 0.0}
    keys = metrics_list[0].keys()
    return {
        key: sum(metrics[key] for metrics in metrics_list) / len(metrics_list)
        for key in keys
    }
