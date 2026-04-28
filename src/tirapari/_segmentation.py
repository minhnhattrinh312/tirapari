"""Core segmentation algorithms used by the tirapari napari plugin."""

from __future__ import annotations

from typing import Literal

import numpy as np
from skimage import filters, morphology, segmentation
from skimage.measure import label


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ThresholdMethod = Literal["otsu", "li", "triangle", "yen", "isodata"]


def threshold_segment(
    image: np.ndarray,
    method: ThresholdMethod = "otsu",
    min_size: int = 64,
    fill_holes: bool = True,
) -> np.ndarray:
    """Segment *image* using a global threshold.

    Parameters
    ----------
    image:
        2-D or 3-D grayscale image.
    method:
        Thresholding algorithm.  One of ``"otsu"``, ``"li"``,
        ``"triangle"``, ``"yen"``, or ``"isodata"``.
    min_size:
        Minimum object size (in pixels) to keep after thresholding.
    fill_holes:
        If ``True``, fill holes in binary mask before labelling.

    Returns
    -------
    np.ndarray
        Integer label array (same shape as *image*).
    """
    _threshold_fns = {
        "otsu": filters.threshold_otsu,
        "li": filters.threshold_li,
        "triangle": filters.threshold_triangle,
        "yen": filters.threshold_yen,
        "isodata": filters.threshold_isodata,
    }
    if method not in _threshold_fns:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {list(_threshold_fns)}"
        )
    thresh_fn = _threshold_fns[method]
    threshold = thresh_fn(image)
    binary = image > threshold
    if fill_holes:
        if binary.ndim == 2:
            binary = morphology.remove_small_holes(binary)
        else:
            for z in range(binary.shape[0]):
                binary[z] = morphology.remove_small_holes(binary[z])
    binary = morphology.remove_small_objects(binary, max_size=min_size)
    return label(binary).astype(np.int32)


def multi_otsu_segment(
    image: np.ndarray,
    n_classes: int = 3,
    min_size: int = 64,
) -> np.ndarray:
    """Segment *image* into *n_classes* regions using multi-Otsu thresholding.

    Parameters
    ----------
    image:
        2-D or 3-D grayscale image.
    n_classes:
        Number of classes (regions) to produce.  Must be ≥ 2.
    min_size:
        Minimum object size to keep in the highest-intensity class.

    Returns
    -------
    np.ndarray
        Integer label array where each value encodes the class
        (0 = background, 1 … n_classes-1 = foreground classes).
    """
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2.")
    thresholds = filters.threshold_multiotsu(image, classes=n_classes)
    regions = np.digitize(image, bins=thresholds).astype(np.int32)
    # Remove small objects from the foreground classes
    foreground = regions >= 1
    foreground = morphology.remove_small_objects(foreground, max_size=min_size)
    regions[~foreground] = 0
    return regions


def watershed_segment(
    image: np.ndarray,
    min_distance: int = 10,
    min_size: int = 64,
    fill_holes: bool = True,
    threshold_method: ThresholdMethod = "otsu",
) -> np.ndarray:
    """Segment *image* using marker-based watershed.

    The markers are derived from local maxima of the distance transform of
    the binary mask obtained by *threshold_method*.

    Parameters
    ----------
    image:
        2-D or 3-D grayscale image.
    min_distance:
        Minimum distance (in pixels) between watershed markers.
    min_size:
        Minimum object size to keep after watershed.
    fill_holes:
        Fill holes in the binary mask before computing the distance
        transform.
    threshold_method:
        Thresholding algorithm used to create the initial binary mask.

    Returns
    -------
    np.ndarray
        Integer label array (same shape as *image*).
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.feature import peak_local_max

    # 1. Binary mask
    binary = _binarise(image, threshold_method, min_size=0, fill_holes=fill_holes)

    # 2. Distance transform
    distance = distance_transform_edt(binary)

    # 3. Markers from local maxima
    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=binary,
    )
    markers = np.zeros_like(binary, dtype=np.int32)
    markers[tuple(coords.T)] = 1
    markers = label(markers)

    # 4. Watershed
    labels = segmentation.watershed(-distance, markers, mask=binary)
    labels = morphology.remove_small_objects(
        labels.astype(bool), max_size=min_size
    )
    # Re-label after removing small objects
    return label(labels).astype(np.int32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binarise(
    image: np.ndarray,
    method: ThresholdMethod,
    min_size: int,
    fill_holes: bool,
) -> np.ndarray:
    """Return a binary mask for *image* using *method*."""
    _threshold_fns = {
        "otsu": filters.threshold_otsu,
        "li": filters.threshold_li,
        "triangle": filters.threshold_triangle,
        "yen": filters.threshold_yen,
        "isodata": filters.threshold_isodata,
    }
    thresh_fn = _threshold_fns[method]
    threshold = thresh_fn(image)
    binary = image > threshold
    if fill_holes:
        if binary.ndim == 2:
            binary = morphology.remove_small_holes(binary)
        else:
            for z in range(binary.shape[0]):
                binary[z] = morphology.remove_small_holes(binary[z])
    if min_size > 0:
        binary = morphology.remove_small_objects(binary, max_size=min_size)
    return binary
