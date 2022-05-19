""" Contains concrete substances for the solid skeleton of the porous medium.
In the current setting, we expect these substances to only appear in the solid, immobile phase,
"""

from typing import List, Optional

from .substance import SolidSubstance

__all__: List[str] = ["UnitSolid", "Salt_simple"]


class UnitSolid(SolidSubstance):
    """
    Represent the academic unit solid, with constant unitary properties.
    Intended usage is testing, debugging and demonstration.

    For a proper documentation of all properties, see parent class.
    """

    @staticmethod
    def molar_mass() -> float:
        return 1.

    def molar_density(
        self, pressure: float, enthalpy: float, temperature: Optional[float] = None
    ) -> float:
        return 1.

    @staticmethod
    def base_porosity() -> float:
        return 1.

    @staticmethod
    def base_permeability() -> float:
        return 1.
    
    @staticmethod
    def poro_reference_pressure() -> float:
        return 1.

    def Fick_diffusivity(
        self, pressure: float, enthalpy: float, temperature: Optional[float] = None
    ) -> float:
        return 1.

    def thermal_conductivity(
        self, pressure: float, enthalpy: float, temperature: Optional[float] = None
    ) -> float:
        return 1.


class Salt_simple(SolidSubstance):
    """ A mix of
    https://en.wikipedia.org/wiki/Sodium_chloride 
    and
    https://www.researchgate.net/post/Does-anyone-know-of-porosity-and-permeability-measurements-of-rock-salt
    """

    @staticmethod
    def molar_mass() -> float:
        return 0.058443

    def molar_density(
        self, pressure: float, enthalpy: float, temperature: Optional[float] = None
        ) -> float:
        kg_pro_m3 = 0.00217* 1e6
        return kg_pro_m3 / self.molar_mass()
    
    def Fick_diffusivity(
        self, pressure: float, enthalpy: float, temperature: Optional[float] = None
        ) -> float:
        return 0.42 # TODO fix this, this is random

    def thermal_conductivity(
        self, pressure: float, enthalpy: float, temperature: Optional[float] = None
        ) -> float:
        if temperature:
            # conduct(8 Kelvin) = 203 and conduct(314 Kelvin) = 6.9
            return -0.636688*temperature + 206.820032
        else:
            # heat capacity c=50.5 J / k / mol -> T = (h-p)/c linearized
            return -0.636688 / 50.5 * (enthalpy - pressure) + 206.820032

    @staticmethod
    def base_porosity() -> float:
        return 0.001

    @staticmethod
    def base_permeability() -> float:
        return 2.23e-20

    @staticmethod
    def poro_reference_pressure() -> float:
        return 1.

    @staticmethod
    def compressibility() -> float:
        return 4.5e-5