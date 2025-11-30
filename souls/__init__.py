"""
Soul package - Convolution-based image processors.

This package provides various "souls" (convolution processors) that transform
images using different strategies and mythical personas.
"""

from .base import Soul
from .seraphim import Seraphim
from .cherubim import Cherubim
from .pandemonium import Pandemonium

__all__ = ['Soul', 'Seraphim', 'Cherubim', 'Pandemonium']
