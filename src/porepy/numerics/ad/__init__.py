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
    operator_functions,
    operators,
)
from .discretizations import *
from .equation_manager import *
from .forward_mode import *
from .functions import *
from .grid_operators import *
from .operator_functions import *
from .operators import *
from .time_derivatives import *

__all__.extend(operators.__all__)
__all__.extend(operator_functions.__all__)
__all__.extend(discretizations.__all__)
__all__.extend(functions.__all__)
__all__.extend(forward_mode.__all__)
__all__.extend(grid_operators.__all__)
__all__.extend(equation_manager.__all__)
__all__.extend(time_derivatives.__all__)
