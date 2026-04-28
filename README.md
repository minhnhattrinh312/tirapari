# tirapari

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/tirapari.svg)](https://pypi.org/project/tirapari)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/tirapari)](https://napari-hub.org/plugins/tirapari)

A [napari] plugin for interactive image segmentation.

**tirapari** provides a dock widget with three segmentation strategies:

| Method | Description |
|---|---|
| **Threshold** | Global thresholding (Otsu, Li, Triangle, Yen, ISODATA) |
| **Multi-Otsu** | Partition the image into N intensity classes |
| **Watershed** | Marker-based watershed using the distance transform |

---

## Installation

```bash
pip install tirapari
```

Or install directly from source:

```bash
git clone https://github.com/minhnhattrinh312/tirapari.git
cd tirapari
pip install -e .
```

---

## Usage

1. Open [napari].
2. From the **Plugins** menu choose **tirapari ▸ Segmentation**.
3. Select your image layer from the *Image layer* dropdown.
4. Choose a segmentation **Method** and adjust the parameters.
5. Click **Run Segmentation**.

A new *Labels* layer is added (or updated) in the viewer for each run.

### Sample data

A built-in sample image is available via **File ▸ Open Sample ▸ tirapari ▸ Sample Cells Image**.

---

## Segmentation methods

### Threshold

Applies a single global threshold derived by the chosen algorithm and labels
connected components in the resulting binary mask.

| Parameter | Description |
|---|---|
| Algorithm | `otsu` / `li` / `triangle` / `yen` / `isodata` |
| Min object size | Objects smaller than this (px) are discarded |
| Fill holes | Remove holes inside foreground regions |

### Multi-Otsu

Splits the image into *N* intensity classes by finding *N − 1* threshold
values simultaneously (Otsu 1979, Liao 2001).

| Parameter | Description |
|---|---|
| Number of classes | How many intensity regions to create (2–10) |
| Min object size | Objects smaller than this (px) are discarded |

### Watershed

Computes a binary mask (via threshold), applies the distance transform, finds
local maxima as seeds, then runs the watershed algorithm to delineate object
boundaries.

| Parameter | Description |
|---|---|
| Threshold method | Algorithm used to create the initial binary mask |
| Min marker distance | Minimum distance (px) between watershed seeds |
| Min object size | Objects smaller than this (px) are discarded |
| Fill holes | Remove holes before the distance transform |

---

## Development

```bash
pip install -e ".[testing]"
pytest
```

---

## License

Distributed under the MIT License.  See [LICENSE](LICENSE) for details.

[napari]: https://napari.org
