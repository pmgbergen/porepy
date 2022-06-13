"""Contains concrete implementation of phases."""
from __future__ import annotations

from typing import List

import numpy as np
from iapws import IAPWS95, SeaWater

import porepy as pp

from .model_fluids import H2O
from .model_solids import NaCl
from .phase import PhaseField

__all__: List[str] = ["SaltWater", "WaterVapor"]


class SaltWater(PhaseField):

    # https://en.wikipedia.org/wiki/Table_of_specific_heat_capacities
    molar_heat_capacity = 0.075327  # kJ / mol / K

    def __init__(self, name: str, gb: pp.GridBucket) -> None:
        super().__init__(name, gb)
        # saving external reference for simplicity
        self.water = H2O(gb)
        self.salt = NaCl(gb)
        # adding 'internally' to use parent class functions
        self.add_substance(self.water)
        self.add_substance(self.salt)

    def molar_density(
        self, pressure: pp.ad.MergedVariable, temperature: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        density = 998.21 / H2O.molar_mass()
        return pp.ad.Array(density * np.ones(self.gb.num_cells()))

    def specific_molar_enthalpy(
        self, pressure: pp.ad.MergedVariable, temperature: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        return (
            pressure / self.molar_density(pressure, temperature)
            + temperature * self.molar_heat_capacity
        )

    def dynamic_viscosity(
        self, pressure: pp.ad.MergedVariable, temperature: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        return pp.ad.Array(np.ones(self.gb.num_cells()))  # 0.001

    def thermal_conductivity(self, pressure: float, temperature: float) -> float:
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
        self, pressure: pp.ad.MergedVariable, temperature: pp.ad.MergedVariable
    ) -> pp.ad.Operator:

        return pressure / (temperature * self.specific_molar_gas_constant)

    def specific_molar_enthalpy(
        self, pressure: pp.ad.MergedVariable, temperature: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        return (
            pressure / self.molar_density(pressure, temperature)
            + temperature * self.molar_heat_capacity
        )

    def dynamic_viscosity(
        self, pressure: pp.ad.MergedVariable, temperature: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        return pp.ad.Array(np.ones(self.gb.num_cells()))  # 0.0003

    def thermal_conductivity(
        self, pressure: pp.ad.MergedVariable, temperature: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        return pp.ad.Array(np.ones(self.gb.num_cells()))  # 0.05
