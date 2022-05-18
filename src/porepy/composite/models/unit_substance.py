""" Contains the SimpleFluid and UnitSolid substances. """

from typing import List

from .._composite_utils import IDEAL_GAS_CONSTANT
from ..substance import FluidSubstance, SolidSubstance

__all__: List[str] = ["SimpleFluid", "UnitSolid"]


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


class UnitSolid(SolidSubstance):
    """
    Represent the academic unit solid, with constant unitary properties.
    Intended usage is testing, debugging and demonstration.

    For a proper documentation of all properties, see parent class.
    """

    @staticmethod
    def molar_mass() -> float:
        return 1.0

    def molar_density(self, *args, **kwargs) -> float:
        return 1.0

    @staticmethod
    def base_porosity() -> float:
        return 1.0

    @staticmethod
    def base_permeability() -> float:
        return 1.0

    def Fick_diffusivity(self, *args, **kwargs) -> float:
        return 1.0

    def thermal_conductivity(self, *args, **kwargs) -> float:
        return 1.0
