""" Contains concrete substances for the fluid phases in the porous medium.
In the current setting, we expect these substances to only appear in liquid or gaseous form.
i.e. they are associated with the flow.
"""

from typing import List

from iapws import IAPWS95

from ._composite_utils import R_IDEAL
from .component import FluidComponent

__all__: List[str] = ["IdealFluid", "H2O"]


class IdealFluid(FluidComponent):
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

    def density(self, p: float, T: float) -> float:
        return p / (R_IDEAL * T)

    def Fick_diffusivity(self, p: float, T: float) -> float:
        return 1.0

    def thermal_conductivity(self, p: float, T: float) -> float:
        return 1.0

    def dynamic_viscosity(self, p: float, T: float) -> float:
        return 1.0

    def molar_heat_capacity(self, p: float, T: float) -> float:
        return 1.0


class H2O(FluidComponent):
    
    @staticmethod
    def molar_mass() -> float:
        """Taken from https://pubchem.ncbi.nlm.nih.gov/compound/water ."""
        return 0.01801528

    def density(self, p: float, T: float) -> float:
        water = IAPWS95(P=p, T=T)
        return water.rho / self.molar_mass()

    def Fick_diffusivity(self, p: float, T: float) -> float:
        return 0.42  # TODO fix this, this is random

    def thermal_conductivity(self, p: float, T: float) -> float:
        water = IAPWS95(P=p, T=T)
        return water.k

    def dynamic_viscosity(self, p: float, T: float) -> float:
        water = IAPWS95(P=p, T=T)
        return water.mu
