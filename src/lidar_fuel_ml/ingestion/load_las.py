"""LAS/LAZ loading entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_las(path: Path) -> Any:
    """Load a point cloud from disk.

    The implementation is intentionally deferred until the LAS stack is chosen.
    """

    raise NotImplementedError("LAS loading is not implemented yet.")
