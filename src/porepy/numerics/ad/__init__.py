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
from . import grid_operators
from .grid_operators import *
from . import equation_manager

__all__.extend(operators.__all__)
__all__.extend(functions.__all__)
__all__.extend(forward_mode.__all__)
__all__.extend(grid_operators.__all__)
__all__.extend(equation_manager.__all__)

from porepy.numerics.ad.utils import concatenate
