""" Init file for all AD functionality.

They should all be accessible through a calling
   >>> import porepy as pp
   >>> pp.ad.Matrix???
etc.

"""
__all__ = []

from . import (
    discretizations,
    equation_manager,
    forward_mode,
    functions,
    grid_operators,
    operators,
)
from .discretizations import *
from .equation_manager import *
from .forward_mode import *
from .functions import *
from .grid_operators import *
from .operators import *

__all__.extend(operators.__all__)
__all__.extend(discretizations.__all__)
__all__.extend(functions.__all__)
__all__.extend(forward_mode.__all__)
__all__.extend(grid_operators.__all__)
__all__.extend(equation_manager.__all__)

from porepy.numerics.ad.utils import concatenate
