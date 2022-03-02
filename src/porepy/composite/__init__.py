""" Composite submodule in PorePy. Contains classes representing components and phases.
Define compositional flow models using available substances.
"""

""" List of components to be included for in the testing framework. 
Include the name of your Component child class here to have it tested by the porepy unit tests and other.
"""
_COMPONENT_TEST_ARRAY = [ "UnitIncompressibleFluid",
                          "UnitIdealFluid",
                          "UnitSolid"
                          ]

from .substance import Substance, FluidSubstance, SolidSubstance
from .phase import Phase

from ._composite_utils import (
    COMPUTATIONAL_VARIABLES,
    create_merged_variable,
    create_merged_mortar_variable
    )

from .unit_substance import UnitIncompressibleFluid, UnitIdealFluid, UnitSolid

from .material_subdomain import MaterialSubdomain
from .computational_domain import ComputationalDomain

#------------------------------------------------------------------------------
### IMPORT concrete Component children below
#------------------------------------------------------------------------------