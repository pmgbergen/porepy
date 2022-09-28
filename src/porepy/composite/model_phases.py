"""Contains concrete implementation of phases."""
from __future__ import annotations

from typing import List

import numpy as np
from iapws import IAPWS95, SeaWater

import porepy as pp

from .model_fluids import H2O
from .model_solids import NaCl
from .phase import Phase

__all__: List[str] = ["SaltWater", "WaterVapor"]


class SaltWater(Phase):

    # https://en.wikipedia.org/wiki/Table_of_specific_heat_capacities
    molar_heat_capacity = 0.075327  # kJ / mol / K

    def __init__(self, name: str, mdg: pp.MixedDimensionalGrid) -> None:
        super().__init__(name, mdg)
        # saving external reference for simplicity
        self.water = H2O(mdg)
        self.salt = NaCl(mdg)
        # adding 'internally' to use parent class functions
        self.add_component(self.water)
        self.add_component(self.salt)

        self._nc = self.ad_system.dof_manager.mdg.num_subdomain_cells()

    def density(
        self, p: pp.ad.MergedVariable, T: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        density = 998.21 / H2O.molar_mass()
        return pp.ad.Array(density * np.ones(self._nc))

    def specific_enthalpy(
        self, p: pp.ad.MergedVariable, T: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        return (
            p / self.density(p, T)
            + T * self.molar_heat_capacity
        )

    def dynamic_viscosity(
        self, p: pp.ad.MergedVariable, T: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        return pp.ad.Array(np.ones(self._nc))  # 0.001

    def thermal_conductivity(self, p: float, T: float) -> float:
        return pp.ad.Array(np.ones(self._nc))


class WaterVapor(Phase):
    """Values found on Wikipedia..."""

    # https://en.wikipedia.org/wiki/Water_vapor see specific gas constant (mass)
    specific_molar_gas_constant = 0.4615 * H2O.molar_mass()  # kJ / mol / K
    molar_heat_capacity = 1.864 * H2O.molar_mass()  # kJ / mol / K

    def __init__(self, name: str, mdg: pp.MixedDimensionalGrid) -> None:
        super().__init__(name, mdg)
        # saving external reference for simplicity
        self.water = H2O(mdg)
        # adding 'internally' to use parent class functions
        self.add_component(self.water)

        self._nc = self.ad_system.dof_manager.mdg.num_subdomain_cells()

    def density(
        self, p: pp.ad.MergedVariable, T: pp.ad.MergedVariable
    ) -> pp.ad.Operator:

        return p / (T * self.specific_molar_gas_constant)

    def specific_enthalpy(
        self, p: pp.ad.MergedVariable, T: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        return (
            p / self.density(p, T)
            + T * self.molar_heat_capacity
        )

    def dynamic_viscosity(
        self, p: pp.ad.MergedVariable, T: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        return pp.ad.Array(np.ones(self._nc))  # 0.0003

    def thermal_conductivity(
        self, p: pp.ad.MergedVariable, T: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        return pp.ad.Array(np.ones(self._nc))  # 0.05
