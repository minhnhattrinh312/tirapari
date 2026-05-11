"""
Authors:
    Minh Nhat Trinh  
    Teresa Correia

Created:
    Feb 2, 2023
"""

# %%
import os
from ._widget import Settings, Combo_box
from .processors import SEG_module
import gc
import torch
import datetime
from magicgui import magic_factory
import napari
from qtpy.QtWidgets import QVBoxLayout, QSplitter, QHBoxLayout, QWidget, QPushButton, QTabWidget, QSpinBox, QDoubleSpinBox, QFormLayout, QComboBox, QLabel, QProgressBar, QRadioButton, QButtonGroup


from qtpy.QtCore import Qt, QThread, Signal
from napari.layers import Image

from napari.qt.threading import thread_worker
from time import time
from enum import Enum
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


# this thread is used to update the progress bar
class BarThread(QThread):
    """Thread used to update a progress bar during reconstruction.

    Computes a percent completion based on:
        value, min, max → emits progressChanged(int)

    Signals:
        progressChanged (int): Percentage (0–100).

    Attributes:
        min (int): Lower bound of the progress range.
        max (int): Upper bound of the progress range.
        value (int): Current progress position.
    """
    progressChanged = Signal(int)

    def __init__(self, parent=None):
        super(BarThread, self).__init__(parent)
        self.max = 1
        self.min = 0
        self.value = 1

    def run(self):
        percent = (self.value - self.min) / (self.max - self.min) * 100
        self.progressChanged.emit(int(percent))

class SegModel(Enum):
    """Supported reconstruction modes."""
    TIRAMISU_ACDC = 0
    TIRAMISU_EMIDEC = 1

class EdgeDevice(Enum):
    JETSON_NANO = 0
    RASBERRY_PI = 1

class SegmentationWidget(QTabWidget):

    name = "Segmentator"

    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        super().__init__()
        self.setup_ui_segmentation()

        self.bar_thread_segmentation = BarThread(self)
        self.bar_thread_segmentation.progressChanged.connect(self.progressBar_segmentation.setValue)


    def setup_ui_segmentation(self):
        def add_section(_layout, _title):
            _layout.addWidget(QLabel(_title))
            _layout.addWidget(QSplitter(Qt.Vertical))
        
        # Tab 1 - Basic settings and reconstruction
        
        # i) add a tab widget
        self.params_widget_basic = QWidget()
        self.addTab(self.params_widget_basic, "Segmentation")
        
        # ii) layout
        self.segmentation_layout = QVBoxLayout()
        self.segmentation_widget = QWidget()
        # self.basic_reconstruction_layout.addWidget(QLabel("Basic reconstruction"))
        self.segmentation_layout.addWidget(self.segmentation_widget)
        
        self.choose_layer_widget_segmentation = choose_layer()
        self.choose_layer_widget_segmentation.call_button.visible = False
        self.add_magic_function(self.choose_layer_widget_segmentation, self.segmentation_layout)
        select_button = QPushButton("Select image layer")
        select_button.clicked.connect(self.select_layer_segmentation)
        self.segmentation_layout.addWidget(select_button)

        settings_layout = QVBoxLayout()
        add_section(settings_layout, "Settings")
        self.segmentation_layout.addLayout(settings_layout)
        # remove space between Select image layer and settings
        self.createSettingsSegmentation(settings_layout)
        self.params_widget_basic.setLayout(self.segmentation_layout)
        
    
    def createSettingsSegmentation(self, slayout):
        self.edge_device = Combo_box(
            "Edge device", initial=EdgeDevice.JETSON_NANO.value, choices=EdgeDevice, layout=slayout, write_function=self.set_segmentation_processor
        )
        self.segmentation_model = Combo_box(
            "Segmentation model", initial=SegModel.TIRAMISU_ACDC.value, choices=SegModel, 
            layout=slayout, write_function=self.set_segmentation_processor
        )

        self.myo_only = Settings(
            "Myocardium only", dtype=bool, initial=False, layout=slayout, write_function=self.set_segmentation_processor
        )
        slayout.addSpacing(500)
        # add calculate segmentation button
        calculate_btn = QPushButton("Segment")
        calculate_btn.clicked.connect(self.volume_segmentation)
        slayout.addWidget(calculate_btn)

        self.progressBar_segmentation = QProgressBar()
        slayout.addWidget(self.progressBar_segmentation)

    def show_segmentation(self, image_values, fullname, **kwargs):

        if "scale" in kwargs.keys():
            scale = kwargs["scale"]
        else:
            scale = [1.0] * image_values.ndim

        if "hold" in kwargs.keys() and fullname in self.viewer.layers:

            self.viewer.layers[fullname].data = image_values
            self.viewer.layers[fullname].scale = scale

        else:
            layer = self.viewer.add_labels(
                image_values, name=fullname, affine=self.affine,
            )
            return layer

    def select_layer_segmentation(self, image: Image):
        """Select input sinogram for basic reconstruction.

        Determines whether the input is 2D or 3D and initializes the Segmentation Processor.

        Args:
            image (Image): Napari image layer selected by the user.
        """
        image = self.choose_layer_widget_segmentation.image.value

        if image.data.ndim == 3 and image.data.shape[2] > 1:
            self.input_type = "3D"
            self.imageRaw_name = image.name
            self.affine = image.affine
            print(self.affine)
            sz, sy, sx = image.data.shape
            print(sz, sy, sx)
            if not hasattr(self, "h_segmentation"):
                self.start_segmentation_processor()
            print(f"Selected image layer: {image.name}")
        else:
            self.input_type = "2D"
            self.imageRaw_name = image.name
            sy, sx = image.data.shape
            print(sy, sx)
            if not hasattr(self, "h_segmentation"):
                self.start_segmentation_processor()
            print(f"Selected image layer: {image.name}")

    def volume_segmentation(self):

        self.scale_segmentation = self.viewer.layers[self.imageRaw_name].scale
        def update_segmentation_image(stack):

            imname = "segmentation_" + self.imageRaw_name
            self.show_segmentation(stack, fullname=imname, scale=self.scale_segmentation, affine=self.affine)
            print("Segmentation completed")
            gc.collect()
            torch.cuda.empty_cache()

        @thread_worker(
            connect={"returned": update_segmentation_image},
        )
        def _segmentation():
            print("myocardium only: ", self.h_segmentation.myo_only)
            volume = self.get_image()
            # tranpose the volume from (D, H, W) to (H, W, D)
            volume_transposed = volume.transpose(2, 1, 0)

            seg = self.h_segmentation.segment(volume_transposed)
            seg = seg.transpose(2, 1, 0)
            
            return seg

        _segmentation()



    def get_image(self):
        try:
            return self.viewer.layers[self.imageRaw_name].data
        except:
            raise (KeyError(r"Please select a valid image"))

    def set_segmentation_processor(self, *args):

        if hasattr(self, "h_segmentation"):
            
            self.h_segmentation.model_name = self.segmentation_model.val
            self.h_segmentation.myo_only = self.myo_only.val
            self.h_segmentation.edge_device = self.edge_device.val

    def stop_segmentation_processor(self):
        """Stop the Segmentation Processor instance."""
        if hasattr(self, "h_segmentation"):
            delattr(self, "h_segmentation")

    def start_segmentation_processor(self):
        """Initialize or reset the Segmentation Processor instance."""

        if hasattr(self, "h_segmentation"):
            self.stop_segmentation_processor()
            self.start_segmentation_processor()
        else:
            print("Reset")
            self.h_segmentation = SEG_module()



    def add_magic_function(self, widget, _layout):
        """Attach a magicgui widget to the layout and auto-refresh layer list.

        Args:
            widget: MagicGUI widget instance.
            _layout: Parent Qt layout.
        """
        self.viewer.layers.events.inserted.connect(widget.reset_choices)
        self.viewer.layers.events.removed.connect(widget.reset_choices)
        _layout.addWidget(widget.native)


@magic_factory
def choose_layer(image: Image):
    """Layer-selection helper used by magicgui."""
    pass  # TODO: substitute with a qtwidget without magic functions
