"""
This package contains VGG-based model classes for mold vision tasks.
Exports:
    - MoldVision
    - MoldVGG_6Chan
    - MoldVGG_Default
"""

from .default_model import MoldVGG_Default
from .moldvision import MoldVision
from .sixchan_model import MoldVGG_6Chan

__all__ = [
    "MoldVision",
    "MoldVGG_6Chan",
    "MoldVGG_Default",
]
