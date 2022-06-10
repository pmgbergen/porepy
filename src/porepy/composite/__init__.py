""" Composite submodule in PorePy. Contains classes representing components and phases.
Define compositional flow models using available substances.
"""

__all__ = []

from . import (
    composition,
    model_fluids,
    model_phases,
    model_solids,
    phase,
    physical_domain,
    substance,
)

from .composition import *
from .model_fluids import *
from .model_phases import *
from .model_solids import *
from .phase import *
from .physical_domain import *
from .substance import *
from ._composite_utils import COMPUTATIONAL_VARIABLES, IDEAL_GAS_CONSTANT

__all__.extend(composition.__all__)
__all__.extend(model_fluids.__all__)
__all__.extend(model_phases.__all__)
__all__.extend(model_solids.__all__)
__all__.extend(phase.__all__)
__all__.extend(physical_domain.__all__)
__all__.extend(substance.__all__)
