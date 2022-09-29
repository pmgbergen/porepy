""" Contains concrete substances for the fluid phases in the porous medium.
In the current setting, we expect these substances to only appear in liquid or gaseous form.
i.e. they are associated with the flow.
"""
from iapws import IAPWS95

from ._composite_utils import R_IDEAL
from .component import FluidComponent

__all__ = ["IdealFluid", "H2O"]


class IdealFluid(FluidComponent):
    """
    Represents the academic example fluid with all properties constant and unitary.

    The liquid and solid density are constant and unitary.
    The gas density is implemented according to the Ideal Gas Law.

    Intended usage is testing, debugging and demonstration.

    For a proper documentation of all properties, see parent class.
    """

    @staticmethod
    def molar_mass():
        return 1.0

    def density(self, p, T):
        return p / (R_IDEAL * T)

    def Fick_diffusivity(self, p, T):
        return 1.0

    def thermal_conductivity(self, p, T):
        return 1.0

    def dynamic_viscosity(self, p, T):
        return 1.0

    def molar_heat_capacity(self, p, T):
        return 1.0


class H2O(FluidComponent):
    # TODO decorate methods and turn them into AD methods

    @staticmethod
    def molar_mass():
        """Taken from https://en.wikipedia.org/wiki/Water_vapor ."""
        return 0.0180152833

    def density(self, p, T):
        water = IAPWS95(P=p, T=T)
        return water.rho / self.molar_mass()

    def Fick_diffusivity(self, p, T):
        return 0.42  # TODO fix this, this is random

    def thermal_conductivity(self, p, T):
        water = IAPWS95(P=p, T=T)
        return water.k

    def dynamic_viscosity(self, p, T):
        water = IAPWS95(P=p, T=T)
        return water.mu

    @staticmethod
    def critical_pressure():
        """Taken from https://en.wikipedia.org/wiki/Critical_point_(thermodynamics) ."""
        return 22064

    @staticmethod
    def critical_temperature():
        """Taken from https://en.wikipedia.org/wiki/Critical_point_(thermodynamics) ."""
        return 647.096

    @staticmethod
    def triple_point_pressure():
        """Taken from https://en.wikipedia.org/wiki/Triple_point#Triple_point_of_water"""
        return 0.611657

    @staticmethod
    def triple_point_temperature():
        """Taken from https://en.wikipedia.org/wiki/Triple_point#Triple_point_of_water"""
        return 273.1600
