"""Raster export helpers."""

from __future__ import annotations

from pathlib import Path

from ..schemas import FeatureRecord


def export_maps(
    *,
    output_path: Path,
    feature_rows: list[FeatureRecord],
    predictions: list[float],
    cell_size: float,
    nodata_value: float = -9999.0,
) -> dict[str, object]:
    """Export CBH predictions to an ESRI ASCII raster."""

    if len(feature_rows) != len(predictions):
        raise ValueError("Feature row count must match prediction count.")
    if not feature_rows:
        raise ValueError("At least one feature row is required to export a raster.")

    x_values = sorted({float(row.metadata["plot_center_x"]) for row in feature_rows})
    y_values = sorted({float(row.metadata["plot_center_y"]) for row in feature_rows}, reverse=True)
    grid = {(float(row.metadata["plot_center_x"]), float(row.metadata["plot_center_y"])): value for row, value in zip(feature_rows, predictions)}

    ncols = len(x_values)
    nrows = len(y_values)
    xllcorner = min(x_values) - cell_size / 2.0
    yllcorner = min(y_values) - cell_size / 2.0

    lines = [
        f"ncols {ncols}",
        f"nrows {nrows}",
        f"xllcorner {xllcorner}",
        f"yllcorner {yllcorner}",
        f"cellsize {cell_size}",
        f"NODATA_value {nodata_value}",
    ]
    for y in y_values:
        row_values = []
        for x in x_values:
            value = grid.get((x, y), nodata_value)
            row_values.append(f"{value:.4f}" if value != nodata_value else str(nodata_value))
        lines.append(" ".join(row_values))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "output_path": str(output_path),
        "ncols": ncols,
        "nrows": nrows,
        "cell_size": cell_size,
        "nodata": nodata_value,
    }
