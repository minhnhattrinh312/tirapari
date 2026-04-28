"""Qt widget for the tirapari segmentation plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari


class SegmentationWidget(QWidget):
    """A dock widget for interactive image segmentation.

    The widget exposes three segmentation strategies:

    * **Threshold** – global threshold (Otsu, Li, Triangle, Yen, ISODATA).
    * **Multi-Otsu** – partition image into *N* intensity classes.
    * **Watershed** – marker-based watershed using distance transform.

    Each method produces a *Labels* layer that is added to (or updated in)
    the napari viewer.
    """

    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self._viewer = napari_viewer
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        # ── Input layer ──────────────────────────────────────────────
        input_group = QGroupBox("Input")
        input_form = QFormLayout(input_group)
        self._layer_combo = QComboBox()
        self._layer_combo.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        input_form.addRow("Image layer:", self._layer_combo)

        # ── Method selector ──────────────────────────────────────────
        method_group = QGroupBox("Method")
        method_layout = QVBoxLayout(method_group)
        self._method_combo = QComboBox()
        self._method_combo.addItems(["Threshold", "Multi-Otsu", "Watershed"])
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        method_layout.addWidget(self._method_combo)

        # ── Parameter panels (stacked via show/hide) ─────────────────
        self._threshold_panel = self._build_threshold_panel()
        self._multi_otsu_panel = self._build_multi_otsu_panel()
        self._watershed_panel = self._build_watershed_panel()

        # ── Common options ───────────────────────────────────────────
        common_group = QGroupBox("Common Options")
        common_form = QFormLayout(common_group)

        self._min_size_spin = QSpinBox()
        self._min_size_spin.setRange(1, 100_000)
        self._min_size_spin.setValue(64)
        self._min_size_spin.setSuffix(" px")
        common_form.addRow("Min object size:", self._min_size_spin)

        self._fill_holes_cb = QCheckBox("Fill holes")
        self._fill_holes_cb.setChecked(True)
        common_form.addRow("", self._fill_holes_cb)

        # ── Run button ───────────────────────────────────────────────
        self._run_btn = QPushButton("Run Segmentation")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self._run_segmentation)

        # ── Status label ─────────────────────────────────────────────
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setAlignment(Qt.AlignCenter)

        # ── Assemble ─────────────────────────────────────────────────
        main_layout.addWidget(input_group)
        main_layout.addWidget(method_group)
        main_layout.addWidget(self._threshold_panel)
        main_layout.addWidget(self._multi_otsu_panel)
        main_layout.addWidget(self._watershed_panel)
        main_layout.addWidget(common_group)
        main_layout.addWidget(self._run_btn)
        main_layout.addWidget(self._status_label)
        main_layout.addStretch()

        # Initial state
        self._on_method_changed("Threshold")
        self._refresh_layer_list()

        # Keep layer list in sync with viewer
        self._viewer.layers.events.inserted.connect(self._refresh_layer_list)
        self._viewer.layers.events.removed.connect(self._refresh_layer_list)

    def _build_threshold_panel(self) -> QGroupBox:
        group = QGroupBox("Threshold Options")
        form = QFormLayout(group)
        self._thresh_method_combo = QComboBox()
        self._thresh_method_combo.addItems(
            ["otsu", "li", "triangle", "yen", "isodata"]
        )
        form.addRow("Algorithm:", self._thresh_method_combo)
        return group

    def _build_multi_otsu_panel(self) -> QGroupBox:
        group = QGroupBox("Multi-Otsu Options")
        form = QFormLayout(group)
        self._n_classes_spin = QSpinBox()
        self._n_classes_spin.setRange(2, 10)
        self._n_classes_spin.setValue(3)
        form.addRow("Number of classes:", self._n_classes_spin)
        return group

    def _build_watershed_panel(self) -> QGroupBox:
        group = QGroupBox("Watershed Options")
        form = QFormLayout(group)

        self._ws_thresh_combo = QComboBox()
        self._ws_thresh_combo.addItems(
            ["otsu", "li", "triangle", "yen", "isodata"]
        )
        form.addRow("Threshold method:", self._ws_thresh_combo)

        self._min_distance_spin = QSpinBox()
        self._min_distance_spin.setRange(1, 500)
        self._min_distance_spin.setValue(10)
        self._min_distance_spin.setSuffix(" px")
        form.addRow("Min marker distance:", self._min_distance_spin)
        return group

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_layer_list(self, *_args) -> None:
        current = self._layer_combo.currentText()
        self._layer_combo.clear()
        image_layers = [
            layer.name
            for layer in self._viewer.layers
            if hasattr(layer, "data")
            and not layer.__class__.__name__ == "Labels"
        ]
        self._layer_combo.addItems(image_layers)
        # Restore previous selection if still present
        idx = self._layer_combo.findText(current)
        if idx >= 0:
            self._layer_combo.setCurrentIndex(idx)

    def _on_method_changed(self, method: str) -> None:
        self._threshold_panel.setVisible(method == "Threshold")
        self._multi_otsu_panel.setVisible(method == "Multi-Otsu")
        self._watershed_panel.setVisible(method == "Watershed")

    def _set_status(self, msg: str, error: bool = False) -> None:
        color = "#cc0000" if error else "#007700"
        self._status_label.setText(
            f'<span style="color:{color}">{msg}</span>'
        )

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def _run_segmentation(self) -> None:
        from tirapari._segmentation import (
            multi_otsu_segment,
            threshold_segment,
            watershed_segment,
        )

        layer_name = self._layer_combo.currentText()
        if not layer_name:
            self._set_status("No image layer selected.", error=True)
            return

        try:
            layer = self._viewer.layers[layer_name]
        except KeyError:
            self._set_status(f"Layer '{layer_name}' not found.", error=True)
            return

        image = np.asarray(layer.data, dtype=np.float64)
        if image.ndim not in (2, 3):
            self._set_status(
                "Only 2-D or 3-D images are supported.", error=True
            )
            return

        # Normalise to [0, 1] to make thresholds comparable across dtypes
        vmin, vmax = image.min(), image.max()
        if vmax > vmin:
            image = (image - vmin) / (vmax - vmin)

        method = self._method_combo.currentText()
        min_size = self._min_size_spin.value()
        fill_holes = self._fill_holes_cb.isChecked()

        try:
            if method == "Threshold":
                algo = self._thresh_method_combo.currentText()
                labels = threshold_segment(
                    image,
                    method=algo,
                    min_size=min_size,
                    fill_holes=fill_holes,
                )
                result_name = f"{layer_name}_threshold_{algo}"
            elif method == "Multi-Otsu":
                n_classes = self._n_classes_spin.value()
                labels = multi_otsu_segment(
                    image,
                    n_classes=n_classes,
                    min_size=min_size,
                )
                result_name = f"{layer_name}_multi_otsu_{n_classes}"
            elif method == "Watershed":
                thresh_algo = self._ws_thresh_combo.currentText()
                min_dist = self._min_distance_spin.value()
                labels = watershed_segment(
                    image,
                    min_distance=min_dist,
                    min_size=min_size,
                    fill_holes=fill_holes,
                    threshold_method=thresh_algo,
                )
                result_name = f"{layer_name}_watershed"
            else:
                self._set_status(f"Unknown method: {method}", error=True)
                return
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Error: {exc}", error=True)
            return

        n_labels = int(labels.max())
        self._set_status(f"Done – found {n_labels} object(s).")

        # Update or create the labels layer
        if result_name in self._viewer.layers:
            self._viewer.layers[result_name].data = labels
        else:
            self._viewer.add_labels(labels, name=result_name)
