""" Composite submodule in PorePy. Contains classes representing components and phases.
Define compositional flow models using available substances.
"""

__all__ = []

from . import (
    component,
    composition,
    model_fluids,
    model_phases,
    model_solids,
    phase,
    physical_domain,
)
from ._composite_utils import R_IDEAL
from .component import *
from .composition import *
from .model_fluids import *
from .model_phases import *
from .model_solids import *
from .phase import *
from .physical_domain import *

__all__.extend(component.__all__)
__all__.extend(composition.__all__)
__all__.extend(model_fluids.__all__)
__all__.extend(model_phases.__all__)
__all__.extend(model_solids.__all__)
__all__.extend(phase.__all__)
__all__.extend(physical_domain.__all__)
