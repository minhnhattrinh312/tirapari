try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
__author__ = "Minh Nhat Trinh and Teresa Correia"
__email__ = "ntminh@ualg.pt"


from ._segmentation_widget import SegmentationWidget
