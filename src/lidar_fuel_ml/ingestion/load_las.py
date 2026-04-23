"""LAS/LAZ loading entrypoint."""

from __future__ import annotations

from pathlib import Path

from ..schemas import PointCloud, PointRecord
from ._csv_loader import load_point_cloud_csv


def load_las(path: Path) -> PointCloud:
    """Load a point cloud from LAS/LAZ or a CSV fallback."""

    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return load_point_cloud_csv(path)

    try:
        import laspy  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "LAS/LAZ loading requires `laspy`, or provide a CSV fallback with x/y/z columns."
        ) from exc

    las = laspy.read(path)
    points = []
    classifications = getattr(las, "classification", None)
    intensities = getattr(las, "intensity", None)
    return_numbers = getattr(las, "return_number", None)
    num_returns = getattr(las, "number_of_returns", None)

    for index, (x, y, z) in enumerate(zip(las.x, las.y, las.z)):
        points.append(
            PointRecord(
                x=float(x),
                y=float(y),
                z=float(z),
                classification=int(classifications[index]) if classifications is not None else None,
                intensity=float(intensities[index]) if intensities is not None else None,
                return_number=int(return_numbers[index]) if return_numbers is not None else None,
                num_returns=int(num_returns[index]) if num_returns is not None else None,
            )
        )

    return PointCloud(
        scene_id=path.stem,
        points=tuple(points),
        crs=_safe_parse_crs(las),
        metadata={"source_path": str(path)},
    )


def _safe_parse_crs(las: object) -> str | None:
    parse_crs = getattr(getattr(las, "header", None), "parse_crs", None)
    if parse_crs is None:
        return None
    try:
        crs = parse_crs()
    except Exception:
        return None
    return str(crs) if crs else None
