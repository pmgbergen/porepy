""" Contains concrete substances for the solid skeleton of the porous medium.
In the current setting, we expect these substances to only appear in the solid, immobile phase,
"""

from typing import List

from .substance import SolidSubstance

__all__: List[str] = ["UnitSolid", "NaCl_simple"]


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


class NaCl_simple(SolidSubstance):
    pass
