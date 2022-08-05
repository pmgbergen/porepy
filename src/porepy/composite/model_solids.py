""" Contains concrete substances for the solid skeleton of the porous medium.
In the current setting, we expect these substances to only appear in the solid, immobile phase,
"""

from .component import SolidComponent

__all__ = ["UnitSolid", "NaCl"]


class UnitSolid(SolidComponent):
    """
    Represent the academic unit solid, with constant unitary properties.
    Intended usage is testing, debugging and demonstration.

    For a proper documentation of all properties, see parent class.
    """

    @staticmethod
    def molar_mass() -> float:
        return 1.0

    def density(self, p: float, T: float) -> float:
        return 1.0

    def Fick_diffusivity(self, p: float, T: float) -> float:
        return 1.0

    def thermal_conductivity(self, p: float, T: float) -> float:
        return 1.0

    @staticmethod
    def base_porosity() -> float:
        return 1.0

    @staticmethod
    def base_permeability() -> float:
        return 1.0


class NaCl(SolidComponent):
    """A mix of
    https://en.wikipedia.org/wiki/Sodium_chloride
    and
    https://www.researchgate.net/post/Does-anyone-know-of-porosity-and-permeability-measurements-of-rock-salt
    """

    @staticmethod
    def molar_mass() -> float:
        return 0.058443

    def density(self, p: float, T: float) -> float:
        kg_pro_m3 = 0.00217 * 1e6
        return kg_pro_m3 / self.molar_mass()

    def Fick_diffusivity(self, p: float, T: float) -> float:
        return 0.42  # TODO fix this, this is random

    def thermal_conductivity(self, p: float, T: float) -> float:
        return -0.636688 * T + 206.820032

    @staticmethod
    def base_porosity() -> float:
        return 0.001

    @staticmethod
    def base_permeability() -> float:
        return 2.23e-20

    @staticmethod
    def compressibility() -> float:
        return 4.5e-5
