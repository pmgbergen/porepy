""" Contains concrete substances for the fluid phases in the porous medium.
In the current setting, we expect these substances to only appear in liquid or gaseous form.
i.e. they are associated with the flow.
"""
from iapws import IAPWS95

from .component import FluidComponent

__all__ = ["H2O"]


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
