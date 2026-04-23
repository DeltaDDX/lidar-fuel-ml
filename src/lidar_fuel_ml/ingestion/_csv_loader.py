"""CSV-based fallback loader for point clouds and plot definitions."""

from __future__ import annotations

import csv
from pathlib import Path

from ..schemas import PlotWindow, PointCloud, PointRecord


def load_point_cloud_csv(path: Path) -> PointCloud:
    """Load a point cloud from a delimited text file.

    Expected columns are ``x``, ``y``, and ``z``. Optional columns include
    ``classification``, ``intensity``, ``return_number``, and ``num_returns``.
    """

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        points = []
        for row in reader:
            points.append(
                PointRecord(
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row["z"]),
                    classification=_maybe_int(row.get("classification")),
                    intensity=_maybe_float(row.get("intensity")),
                    return_number=_maybe_int(row.get("return_number")),
                    num_returns=_maybe_int(row.get("num_returns")),
                )
            )

    return PointCloud(scene_id=path.stem, points=tuple(points), metadata={"source_path": str(path)})


def load_plot_windows_csv(path: Path) -> list[PlotWindow]:
    """Load circular plot definitions from CSV."""

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            PlotWindow(
                plot_id=row["plot_id"],
                center_x=float(row["center_x"]),
                center_y=float(row["center_y"]),
                radius=float(row["radius"]),
                target_cbh=_maybe_float(row.get("target_cbh")),
            )
            for row in reader
        ]


def _maybe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _maybe_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
