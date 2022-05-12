""" Composite submodule in PorePy. Contains classes representing components and phases.
Define compositional flow models using available substances.
"""

__all__ = []

from . import (
    composition,
    fluid,
    material_subdomain,
    phase,
    solid,
    substance,
    unit_substance,
)
from .composition import *
from .fluid import *
from .material_subdomain import *
from .phase import *
from .solid import *
from .substance import *
from .unit_substance import *

__all__.extend(composition.__all__)
__all__.extend(fluid.__all__)
__all__.extend(material_subdomain.__all__)
__all__.extend(phase.__all__)
__all__.extend(solid.__all__)
__all__.extend(substance.__all__)
__all__.extend(unit_substance.__all__)
