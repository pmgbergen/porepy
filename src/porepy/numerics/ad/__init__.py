""" Init file for all AD functionality.

They should all be accessible through a calling
   >>> import porepy as pp
   >>> pp.ad.Matrix???
etc.

"""
__all__ = []

from . import operators
from .operators import *
from . import functions
from .functions import *
from . import forward_mode
from .forward_mode import *

__all__.extend(operators.__all__)
__all__.extend(functions.__all__)
__all__.extend(forward_mode.__all__)

from porepy.numerics.ad.utils import concatenate
