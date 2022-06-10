"""Contains concrete implementation of phases."""
from __future__ import annotations

from typing import List, Optional, Union

from iapws import IAPWS95, SeaWater

import numpy as np
import porepy as pp

from .model_fluids import H2O
from .model_solids import NaCl
from .phase import PhaseField

__all__: List[str] = ["SaltWater", "WaterVapor"]


class SaltWater(PhaseField):

    # source wikipedia
    molar_heat_capacity = 0.0042  # kJ / mol / K
    
    def __init__(self, name: str, gb: pp.GridBucket) -> None:
        super().__init__(name, gb)
        # saving external reference for simplicity
        self.water = H2O(gb)
        self.salt = NaCl(gb)
        # adding 'internally' to use parent class functions
        self.add_substance(self.water)
        self.add_substance(self.salt)

    def molar_density(
        self,
        pressure: pp.ad.MergedVariable,
        enthalpy: pp.ad.MergedVariable,
        temperature: Optional[Union[pp.ad.MergedVariable, None]] = None,
    ) -> pp.ad.Operator:
        density = 998.21 / H2O.molar_mass()
        return pp.ad.Array(density * np.ones(self.gb.num_cells()))

    def enthalpy(
        self,
        pressure: pp.ad.MergedVariable,
        enthalpy: pp.ad.MergedVariable,
        temperature: Optional[Union[pp.ad.MergedVariable, None]] = None,
    ) -> pp.ad.Operator:
        if temperature:
            return (1. + self.molar_heat_capacity) * temperature
        else:
            # TODO get physical
            return enthalpy / 2.
        # return pp.ad.Array(np.ones(self.gb.num_cells()))

    def dynamic_viscosity(
        self, pressure: pp.ad.MergedVariable, enthalpy: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        return pp.ad.Array(np.ones(self.gb.num_cells()))  # 0.001

    def thermal_conductivity(self, pressure: float, enthalpy: float) -> float:
        return pp.ad.Array(np.ones(self.gb.num_cells()))


class WaterVapor(PhaseField):
    """Values found on Wikipedia..."""

    # https://en.wikipedia.org/wiki/Water_vapor see specific gas constant (mass)
    specific_molar_gas_constant = 0.4615 * H2O.molar_mass()  # kJ / mol / K
    molar_heat_capacity = 1.864 * H2O.molar_mass()  # kJ / mol / K

    def __init__(self, name: str, gb: pp.GridBucket) -> None:
        super().__init__(name, gb)
        # saving external reference for simplicity
        self.water = H2O(gb)
        # adding 'internally' to use parent class functions
        self.add_substance(self.water)

    def molar_density(
        self,
        pressure: pp.ad.MergedVariable,
        enthalpy: pp.ad.MergedVariable,
        temperature: Optional[Union[pp.ad.MergedVariable, None]] = None,
    ) -> pp.ad.Operator:
        val = 0.756182 / H2O.molar_mass()
        return pp.ad.Array( val * np.ones(self.gb.num_cells()))
        # if temperature:
        #     # ideal gas law
        #     return pressure / temperature / self.specific_molar_gas_constant
        # else:
        #     # linearized internal energy
        #     # rho h = rho T - p -> rho = p / (h-T)
        #     return pressure / (enthalpy)

    def enthalpy(
        self,
        pressure: pp.ad.MergedVariable,
        enthalpy: pp.ad.MergedVariable,
        temperature: Optional[Union[pp.ad.MergedVariable, None]] = None,
    ) -> pp.ad.Operator:
        if temperature:
            return (self.molar_heat_capacity + self.specific_molar_gas_constant) * temperature
        else:
            # TODO get physical
            return enthalpy / 2.

    def dynamic_viscosity(
        self, pressure: pp.ad.MergedVariable, enthalpy: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        return pp.ad.Array(np.ones(self.gb.num_cells()))  # 0.0003

    def thermal_conductivity(
        self, pressure: pp.ad.MergedVariable, enthalpy: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        return pp.ad.Array(np.ones(self.gb.num_cells()))  # 0.05
