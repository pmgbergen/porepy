""" Contains concrete substances for the fluid phases in the porous medium.
In the current setting, we expect these substances to only appear in liquid or gaseous form.
i.e. they are associated with the flow.
"""

from typing import List

import iapws

from .substance import FluidSubstance
from ._composite_utils import IDEAL_GAS_CONSTANT

__all__: List[str] = ["SimpleFluid", "H20_iapws"]


class SimpleFluid(FluidSubstance):
    """
    Represents the academic example fluid with all properties constant and unitary.

    The liquid and solid density are constant and unitary.
    The gas density is implemented according to the Ideal Gas Law.

    Intended usage is testing, debugging and demonstration.

    For a proper documentation of all properties, see parent class.
    """

    @staticmethod
    def molar_mass() -> float:
        return 1.0

    def molar_density(
        self, state_of_matter: str, pressure: float, temperature: float, *args, **kwargs
    ) -> float:

        if state_of_matter in ("liquid", "solid"):
            return 1.0
        elif state_of_matter == "gas":
            return pressure / (IDEAL_GAS_CONSTANT * temperature)
        else:
            raise ValueError("Unsupported state of matter: '%s'" % (state_of_matter))

    def Fick_diffusivity(self, *args, **kwargs) -> float:
        return 1.0

    def thermal_conductivity(self, *args, **kwargs) -> float:
        return 1.0

    def dynamic_viscosity(self, *args, **kwargs) -> float:
        return 1.0


class H20_iapws(FluidSubstance):
    pass
