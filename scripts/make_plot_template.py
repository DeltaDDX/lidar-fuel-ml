from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lidar_fuel_ml.ingestion.load_las import load_las


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a plot CSV template from LAS/LAZ bounds.")
    parser.add_argument(
        "--point-cloud",
        type=Path,
        default=Path("data/raw/points.laz"),
        help="Path to the source LAS/LAZ file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/plots.csv"),
        help="Where to write the template plot CSV.",
    )
    parser.add_argument("--radius", type=float, default=11.28, help="Default plot radius in map units.")
    parser.add_argument("--rows", type=int, default=1, help="Number of template plots to create.")
    args = parser.parse_args()

    cloud = load_las(args.point_cloud)
    xs = [point.x for point in cloud.points]
    ys = [point.y for point in cloud.points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    centers = _make_centers(min_x, max_x, min_y, max_y, args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["plot_id", "center_x", "center_y", "radius", "target_cbh"])
        for index, (center_x, center_y) in enumerate(centers, start=1):
            writer.writerow([f"plot_{index}", f"{center_x:.3f}", f"{center_y:.3f}", args.radius, ""])

    print(f"Wrote {len(centers)} template plots to {args.output}")


def _make_centers(min_x: float, max_x: float, min_y: float, max_y: float, rows: int) -> list[tuple[float, float]]:
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    if rows <= 1:
        return [(center_x, center_y)]

    span_x = max_x - min_x
    step = span_x / max(rows, 1)
    start_x = min_x + step / 2.0
    return [(start_x + step * index, center_y) for index in range(rows)]


if __name__ == "__main__":
    main()
