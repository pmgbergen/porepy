import porepy as pp

#from . import constit_library, fluid_mass_balance

from constit_library import (
    ConstantRock,
    DensityFromPressure,
    ConstantDensity,
    ConstantViscosity,
    UnitFluid,
    UnitRock,
)
from fluid_mass_balance import (
    ConstitutiveEquationsCompressibleFlow,
    MassBalanceEquations,
    SolutionStrategyCompressibleFlow,
    VariablesSinglePhaseFlow,
)
from geometry import ModelGeometry

# Example 1


class CompressibleCombined(
    ModelGeometry,
    MassBalanceEquations,
    ConstitutiveEquationsCompressibleFlow,
    VariablesSinglePhaseFlow,
    SolutionStrategyCompressibleFlow,
):
    pass


"""
Assuming a setting of rock and fluid components (defaulting to pp.UnitRock and pp.UnitFluid),
usage is:
"""

m = CompressibleCombined()
#pp.run_time_dependent_model(m)

# Define your favourite CO_2 combination to be plugged in in multiple run scripts.


class cooConstit(ConstantRock, DensityFromPressure):
    pass


class CombinedConstit(cooConstit, ConstitutiveEquationsCompressibleFlow):

    pass


"""
This class is combined with variables, solution strategy etc. as mixins/using inheritance.
The colossus will also be assigned materials, which could e.g. be pp.CO_2 for the fluid and
EiriksWeirdRock for the rock:
"""


class EiriksCombinedMassBalance(
    ModelGeometry,
    ConstantRock,
    ConstantViscosity,
    ConstantDensity,
    VariablesSinglePhaseFlow,
    MassBalanceEquations,
    ConstitutiveEquationsCompressibleFlow,
    SolutionStrategyCompressibleFlow,
):
    pass




params = {
    "materials": {"fluid": UnitFluid, "rock": UnitRock},
    "linear_solver": "pypardiso",  # Example showing solution strategy params may go here.
}
m = EiriksCombinedMassBalance(params)
pp.run_time_dependent_model(m, params)
