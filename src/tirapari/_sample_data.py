"""Provides a built-in sample image for the tirapari plugin."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def make_sample_data() -> List[Tuple[np.ndarray, dict, str]]:
    """Return a list of layer data tuples containing a sample grayscale image.

    The sample consists of a synthetic 256×256 image with several circular
    blobs that are suitable for demonstrating the segmentation widget.

    Returns
    -------
    list of (data, meta, layer_type)
        Compatible with the napari ``add_*`` layer protocol.
    """
    try:
        from skimage.data import cells3d

        # Use the nuclei channel (channel 1) of slice 30
        data = cells3d()[30, 1].astype(np.float32)
        data = (data - data.min()) / (data.max() - data.min())
    except BaseException:
        # Fallback: use deterministic synthetic blobs when data is unavailable
        data = _synthetic_blobs()

    meta = {"name": "sample_cells", "colormap": "gray"}
    return [(data, meta, "image")]


# ---------------------------------------------------------------------------
# Fallback synthetic data
# ---------------------------------------------------------------------------

def _synthetic_blobs(shape: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Create a synthetic image with Gaussian blobs on a dark background."""
    rng = np.random.default_rng(42)
    image = np.zeros(shape, dtype=np.float32)

    n_blobs = 15
    ys = rng.integers(30, shape[0] - 30, n_blobs)
    xs = rng.integers(30, shape[1] - 30, n_blobs)
    radii = rng.integers(10, 30, n_blobs)
    intensities = rng.uniform(0.5, 1.0, n_blobs)

    yy, xx = np.ogrid[: shape[0], : shape[1]]
    for y, x, r, intensity in zip(ys, xs, radii, intensities):
        dist = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)
        blob = intensity * np.exp(-(dist**2) / (2 * (r / 2) ** 2))
        image += blob.astype(np.float32)

    # Normalise to [0, 1]
    image = np.clip(image, 0, 1)
    # Add mild Gaussian noise
    image += rng.normal(0, 0.02, shape).astype(np.float32)
    image = np.clip(image, 0, 1)
    return image
