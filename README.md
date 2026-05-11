# myopari

myopari is a napari plugin for cardiac MRI segmentation using ONNX models.
It provides a dock widget that lets you select an image layer, choose a
segmentation model, run inference, and visualize the predicted labels directly
in napari.

## Features

- napari dock widget for interactive segmentation
- Two built-in ONNX segmentation models:
  - TIRAMISU ACDC
  - TIRAMISU EMIDEC
- CPU and CUDA execution provider support through ONNX Runtime
- Optional myocardium-only output mode
- Designed for 2D and 3D image layers

## Requirements

- Python 3.9+
- napari
- numpy
- scikit-image
- scipy
- onnxruntime (CPU) or onnxruntime-gpu (CUDA)

## Installation

### Option 1: Install from source (recommended during development)

```bash
git clone <your-repo-url>
cd myopari
pip install -e .
```

### Option 2: Install with pip

```bash
pip install myopari
```

If you want GPU inference, install ONNX Runtime GPU:

```bash
pip install onnxruntime-gpu
```

## Usage in napari

1. Open napari.
2. Open your image (2D or 3D stack).
3. Open the plugin widget:
	- Plugins -> myopari -> myopari
4. In the widget:
	- Click Select image layer
	- Choose Edge device (Jetson Nano or Rasberry Pi)
	- Choose Segmentation model
	- Optional: enable Myocardium only
	- Click Segment
5. The result is added as a labels layer named `segmentation_<input_layer_name>`.

## Local plugin test

Run the included test launcher:

```bash
python plugin_test.py
```

This opens napari and docks the myopari widget.

## Package information

- Package name: `myopari`
- License: MIT
- Authors: Minh Nhat Trinh, Teresa Correia

## Notes

- The plugin bundles ONNX model files under `src/myopari/Resources/`.
- For best performance with CUDA, ensure your GPU drivers/CUDA stack match your
  installed `onnxruntime-gpu` version.
