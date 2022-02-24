""" Composite submodule in PorePy. Contains classes representing components and phases.
Define compositional flow models using available substances.
"""

""" List of components to be included for in the testing framework. 
Include the name of your Component child class here to have it tested by the porepy unit tests and other.
"""
_COMPONENT_TEST_ARRAY = [ "UnitIncompressibleFluid",
                          "UnitIdealFluid"
                          ]

from .component import Component, FluidComponent, SolidSkeletonComponent
from .phase import Phase

from ._composite_utils import create_merged_variable_on_gb

from .unit_fluid import UnitIncompressibleFluid
from .unit_fluid import UnitIdealFluid

#------------------------------------------------------------------------------
### IMPORT concrete Component children below
#------------------------------------------------------------------------------