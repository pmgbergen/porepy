""" Composite submodule in PorePy. Contains classes representing components and phases.
Define compositional flow models using available substances.
"""

__all__ = []

from . import (
    compositional_domain,
    decorators,
    fluid,
    material_subdomain,
    phase,
    solid,
    substance,
    unit_substance
)

from .compositional_domain import *
from .decorators import *
from .fluid import *
from .material_subdomain import *
from .phase import *
from .solid import *
from .substance import *
from .unit_substance import *

__all__.extend(compositional_domain.__all__)
__all__.extend(decorators.__all__)
__all__.extend(fluid.__all__)
__all__.extend(material_subdomain.__all__)
__all__.extend(phase.__all__)
__all__.extend(solid.__all__)
__all__.extend(substance.__all__)
__all__.extend(unit_substance.__all__)
