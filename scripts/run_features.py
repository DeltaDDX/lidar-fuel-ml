from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lidar_fuel_ml.pipelines.run_feature_pipeline import run_feature_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature generation for a LAS/LAZ scene.")
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
        default=Path("reports/tables/feature_rows.json"),
        help="Where to write the feature rows as JSON.",
    )
    args = parser.parse_args()

    feature_rows = run_feature_pipeline(args.point_cloud, args.plots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "scene_id": row.scene_id,
            "record_id": row.record_id,
            "target": row.target,
            "features": row.features,
            "metadata": row.metadata,
        }
        for row in feature_rows
    ]
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(feature_rows)} feature rows to {args.output}")


if __name__ == "__main__":
    main()
