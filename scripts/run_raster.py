from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lidar_fuel_ml.pipelines.run_raster_pipeline import run_raster_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a wall-to-wall CBH raster from a LAS/LAZ scene.")
    parser.add_argument("--point-cloud", type=Path, default=Path("data/raw/points.laz"))
    parser.add_argument("--plots", type=Path, default=Path("data/raw/plots.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/tables/cbh_raster.asc"))
    parser.add_argument("--cell-size", type=float, default=20.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    args = parser.parse_args()

    plot_path = args.plots if args.plots.exists() else None
    result = run_raster_pipeline(
        args.point_cloud,
        args.output,
        plot_path=plot_path,
        cell_size=args.cell_size,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
    print(f"Wrote raster to {result['raster']['output_path']}")
    print(f"Grid rows: {len(result['grid_rows'])}")
    print(f"Used training plots: {len(result['training_rows'])}")


if __name__ == "__main__":
    main()
