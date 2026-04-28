"""tirapari – a napari plugin for image segmentation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tirapari")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__"]
