"""Top-level package for the lidar fuel modeling project."""

from .logging_utils import configure_logging, get_logger
from .paths import ProjectPaths, get_project_paths

__all__ = [
    "ProjectPaths",
    "configure_logging",
    "get_logger",
    "get_project_paths",
]
