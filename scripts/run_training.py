from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lidar_fuel_ml.pipelines.run_training_pipeline import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model training with spatial CV for a LAS/LAZ scene.")
    parser.add_argument(
        "--point-cloud",
        type=Path,
        default=Path("data/raw/points.laz"),
        help="Path to the source LAS/LAZ file.",
    )
    parser.add_argument(
        "--plots",
        type=Path,
        default=Path("data/raw/plots.csv"),
        help="Path to the plot definition CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tables/training_metrics.json"),
        help="Where to write the training summary JSON.",
    )
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--block-size", type=float, default=30.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    args = parser.parse_args()

    result = run_training_pipeline(
        args.point_cloud,
        args.plots,
        n_splits=args.n_splits,
        block_size=args.block_size,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
    serializable = {
        "overall_metrics": result["overall_metrics"],
        "folds": result["folds"],
        "num_feature_rows": len(result["feature_rows"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"Wrote training summary to {args.output}")


if __name__ == "__main__":
    main()
