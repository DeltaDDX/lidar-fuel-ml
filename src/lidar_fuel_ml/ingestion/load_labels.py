"""Label loading entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_labels(path: Path) -> Any:
    """Load supervision labels for a scene or tile."""

    raise NotImplementedError("Label loading is not implemented yet.")
