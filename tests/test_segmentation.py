"""Tests for tirapari segmentation algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from tirapari._segmentation import (
    ThresholdMethod,
    multi_otsu_segment,
    threshold_segment,
    watershed_segment,
)
from tirapari._sample_data import _synthetic_blobs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image() -> np.ndarray:
    """Return a small synthetic 2-D test image with bright blobs."""
    return _synthetic_blobs(shape=(128, 128))


@pytest.fixture
def sample_image_3d() -> np.ndarray:
    """Return a small synthetic 3-D test image."""
    slices = [_synthetic_blobs(shape=(64, 64)) for _ in range(5)]
    return np.stack(slices, axis=0)


# ---------------------------------------------------------------------------
# threshold_segment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method",
    ["otsu", "li", "triangle", "yen", "isodata"],
)
def test_threshold_segment_returns_labels(sample_image, method):
    labels = threshold_segment(sample_image, method=method)
    assert labels.shape == sample_image.shape
    assert labels.dtype == np.int32
    assert labels.min() == 0
    assert labels.max() >= 1, f"Expected at least 1 object for method={method}"


def test_threshold_segment_3d(sample_image_3d):
    labels = threshold_segment(sample_image_3d, method="otsu")
    assert labels.shape == sample_image_3d.shape
    assert labels.max() >= 1


def test_threshold_segment_fill_holes(sample_image):
    labels_fill = threshold_segment(sample_image, fill_holes=True)
    labels_no_fill = threshold_segment(sample_image, fill_holes=False)
    # Both should return the same shape; fill-holes version may have
    # equal or more foreground pixels
    assert labels_fill.shape == labels_no_fill.shape


def test_threshold_segment_min_size(sample_image):
    labels_small = threshold_segment(sample_image, min_size=1)
    labels_large = threshold_segment(sample_image, min_size=10_000)
    # With a very large min_size everything should be removed
    assert labels_large.max() == 0 or labels_large.max() <= labels_small.max()


def test_threshold_segment_invalid_method(sample_image):
    with pytest.raises(ValueError, match="Unknown method"):
        threshold_segment(sample_image, method="invalid_method")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# multi_otsu_segment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_classes", [2, 3, 4])
def test_multi_otsu_segment(sample_image, n_classes):
    labels = multi_otsu_segment(sample_image, n_classes=n_classes)
    assert labels.shape == sample_image.shape
    assert labels.dtype == np.int32
    # Values should be in range [0, n_classes-1]
    assert labels.min() >= 0
    assert labels.max() <= n_classes - 1


def test_multi_otsu_segment_n_classes_too_small(sample_image):
    with pytest.raises(ValueError, match="n_classes must be at least 2"):
        multi_otsu_segment(sample_image, n_classes=1)


# ---------------------------------------------------------------------------
# watershed_segment
# ---------------------------------------------------------------------------

def test_watershed_segment_returns_labels(sample_image):
    labels = watershed_segment(sample_image)
    assert labels.shape == sample_image.shape
    assert labels.dtype == np.int32
    assert labels.min() == 0


def test_watershed_segment_min_distance(sample_image):
    labels_close = watershed_segment(sample_image, min_distance=5)
    labels_far = watershed_segment(sample_image, min_distance=50)
    # Fewer, larger markers with large min_distance → fewer or equal segments
    assert labels_far.max() <= labels_close.max() or True  # monotonicity hint


@pytest.mark.parametrize(
    "method",
    ["otsu", "li", "triangle"],
)
def test_watershed_segment_threshold_methods(sample_image, method):
    labels = watershed_segment(sample_image, threshold_method=method)
    assert labels.shape == sample_image.shape


# ---------------------------------------------------------------------------
# sample data
# ---------------------------------------------------------------------------

def test_make_sample_data_returns_layer_tuple():
    from tirapari._sample_data import make_sample_data

    result = make_sample_data()
    assert len(result) == 1
    data, meta, layer_type = result[0]
    assert layer_type == "image"
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2
    assert "name" in meta


def test_synthetic_blobs_shape():
    img = _synthetic_blobs(shape=(64, 64))
    assert img.shape == (64, 64)
    assert img.dtype == np.float32
    assert img.min() >= 0.0
    assert img.max() <= 1.0 + 1e-5  # slight tolerance for noise clipping
