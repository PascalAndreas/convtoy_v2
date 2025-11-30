"""
Soul package - Convolution-based image processors.

This package provides various "souls" (convolution processors) that transform
images using different strategies and mythical personas.
"""

from .base import Soul
from .seraphim import Seraphim
from .cherubim import Cherubim
from .pandemonium import Pandemonium
from .lenia import Lenia
from .inception import InceptionSoul
from .cryptolenia import CryptoLenia

__all__ = ['Soul', 'Seraphim', 'Cherubim', 'Pandemonium', 'Lenia', 'InceptionSoul', 'CryptoLenia']
