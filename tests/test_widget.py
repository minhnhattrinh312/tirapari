"""Tests for the tirapari Qt widget (requires napari + Qt)."""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _napari_available() -> bool:
    try:
        import napari  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def make_napari_viewer(qtbot):
    """Fixture that creates and cleans up a napari viewer."""
    import napari

    viewers = []

    def factory(**kwargs):
        viewer = napari.Viewer(**kwargs)
        viewers.append(viewer)
        return viewer

    yield factory

    for v in viewers:
        v.close()


# ---------------------------------------------------------------------------
# Widget smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _napari_available(),
    reason="napari not installed",
)
def test_widget_creates_without_error(make_napari_viewer, qtbot):
    from tirapari._widget import SegmentationWidget

    viewer = make_napari_viewer(show=False)
    widget = SegmentationWidget(viewer)
    qtbot.addWidget(widget)
    assert widget is not None


@pytest.mark.skipif(
    not _napari_available(),
    reason="napari not installed",
)
def test_widget_segments_image_layer(make_napari_viewer, qtbot):
    from tirapari._sample_data import _synthetic_blobs
    from tirapari._widget import SegmentationWidget

    viewer = make_napari_viewer(show=False)
    image = _synthetic_blobs(shape=(64, 64))
    viewer.add_image(image, name="test_image")

    widget = SegmentationWidget(viewer)
    qtbot.addWidget(widget)

    # Select the threshold method and run
    widget._method_combo.setCurrentText("Threshold")
    widget._run_btn.click()

    # A labels layer should have been added
    label_layers = [
        l for l in viewer.layers if l.__class__.__name__ == "Labels"
    ]
    assert len(label_layers) >= 1
