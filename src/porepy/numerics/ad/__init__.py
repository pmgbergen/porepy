"""Init file for all AD functionality.

They should all be accessible through a calling
   >>> import porepy as pp
   >>> pp.ad.SparseArray
etc.

"""

__all__ = []

from . import (
    ad_utils,
    discretizations,
    equation_system,
    forward_mode,
    functions,
    get_set_values,
    grid_entity,
    grid_operators,
    indexers,
    operator_functions,
    operator_space,
    operators,
    surrogate_operator,
    time_derivatives,
)
from .ad_utils import *
from .discretizations import *
from .equation_system import *
from .forward_mode import *
from .functions import *
from .get_set_values import *
from .grid_entity import *
from .grid_operators import *
from .indexers import *
from .operator_functions import *
from .operator_space import *
from .operators import *
from .surrogate_operator import *
from .time_derivatives import *

__all__.extend(grid_entity.__all__)
__all__.extend(ad_utils.__all__)
__all__.extend(get_set_values.__all__)
__all__.extend(operators.__all__)
__all__.extend(operator_functions.__all__)
__all__.extend(operator_space.__all__)
__all__.extend(discretizations.__all__)
__all__.extend(functions.__all__)
__all__.extend(forward_mode.__all__)
__all__.extend(grid_operators.__all__)
__all__.extend(equation_system.__all__)
__all__.extend(time_derivatives.__all__)
__all__.extend(surrogate_operator.__all__)
__all__.extend(indexers.__all__)
