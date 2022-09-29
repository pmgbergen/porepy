"""Contains concrete implementation of phases."""
from __future__ import annotations

from typing import Optional

import numpy as np

import porepy as pp

from .model_fluids import H2O
from .model_solids import NaCl
from .phase import Phase
from ._composite_utils import R_IDEAL

__all__ = ["SaltWater", "WaterVapor"]


class SaltWater(Phase):

    # https://en.wikipedia.org/wiki/Table_of_specific_heat_capacities
    molar_heat_capacity = 0.075327  # kJ / mol / K

    def __init__(
        self, name: str, ad_system: Optional[pp.ad.ADSystemManager] = None
    ) -> None:
        super().__init__(name, ad_system)
        # saving external reference for simplicity
        self.water = H2O(ad_system)
        self.salt = NaCl(ad_system)
        # adding 'internally' to use parent class functions
        self.add_component(self.water)
        self.add_component(self.salt)

        if ad_system:
            self._nc = self.ad_system.dof_manager.mdg.num_subdomain_cells()
        else:
            self._nc = 1

    def density(self, p, T):
        # https://www.usgs.gov/special-topics/water-science-school/science/water-density
        # converted to kg / m^3 then to mol / m^3
        density = 958.65 / H2O.molar_mass()
        return pp.ad.Array(density * np.ones(self._nc))

    def specific_enthalpy(self, p, T):
        return p / self.density(p, T) + T * self.molar_heat_capacity

    def dynamic_viscosity(self, p, T):
        return pp.ad.Array(np.ones(self._nc))  # 0.001

    def thermal_conductivity(self, p, T):
        return pp.ad.Array(np.ones(self._nc))


class WaterVapor(Phase):
    """Values found on Wikipedia..."""

    # https://en.wikipedia.org/wiki/Table_of_specific_heat_capacities (steam)
    molar_heat_capacity = 0.03747  # kJ / mol / K

    def __init__(
        self, name: str, ad_system: Optional[pp.ad.ADSystemManager] = None
    ) -> None:
        super().__init__(name, ad_system)
        # saving external reference for simplicity
        self.water = H2O(ad_system)
        # adding 'internally' to use parent class functions
        self.add_component(self.water)

    def density(self, p, T):
        return p / (T * R_IDEAL)

    def specific_enthalpy(self, p, T):
        return p / self.density(p, T) + T * self.molar_heat_capacity

    def dynamic_viscosity(self, p, T):
        if self.ad_system:
            nc = self.ad_system.dof_manager.mdg.num_subdomain_cells()
            return pp.ad.Array(np.ones(nc))  # 0.0003
        else:
            return 1.

    def thermal_conductivity(self, p, T):
        if self.ad_system:
            nc = self.ad_system.dof_manager.mdg.num_subdomain_cells()
            return pp.ad.Array(np.ones(nc))  # 0.0003
        else:
            return 1.
