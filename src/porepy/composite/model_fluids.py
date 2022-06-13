""" Contains concrete substances for the fluid phases in the porous medium.
In the current setting, we expect these substances to only appear in liquid or gaseous form.
i.e. they are associated with the flow.
"""

from typing import List

from iapws import IAPWS95

from ._composite_utils import IDEAL_GAS_CONSTANT
from .substance import FluidSubstance

__all__: List[str] = ["IdealFluid", "H2O"]


class IdealFluid(FluidSubstance):
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

    def molar_density(self, pressure: float, temperature: float) -> float:
        return pressure / (IDEAL_GAS_CONSTANT * temperature)

    def Fick_diffusivity(self, *args, **kwargs) -> float:
        return 1.0

    def thermal_conductivity(self, *args, **kwargs) -> float:
        return 1.0

    def dynamic_viscosity(self, *args, **kwargs) -> float:
        return 1.0

    def molar_heat_capacity(self, pressure: float, temperature: float) -> float:
        return 1.0


class H2O(FluidSubstance):
    @staticmethod
    def molar_mass() -> float:
        """Taken from https://pubchem.ncbi.nlm.nih.gov/compound/water ."""
        return 0.01801528

    def molar_density(self, pressure: float, temperature: float) -> float:
        water = IAPWS95(P=pressure, T=temperature)
        return water.rho / self.molar_mass()

    def Fick_diffusivity(self, pressure: float, temperature: float) -> float:
        return 0.42  # TODO fix this, this is random

    def thermal_conductivity(self, pressure: float, temperature: float) -> float:
        water = IAPWS95(P=pressure, T=temperature)
        return water.k

    def dynamic_viscosity(self, pressure: float, temperature: float) -> float:
        water = IAPWS95(P=pressure, T=temperature)
        return water.mu
