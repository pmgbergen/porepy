"""Contains concrete implementation of phases."""

from typing import List, Optional, Union

from iapws import IAPWS95, SeaWater

import numpy as np
import porepy as pp

from .model_fluids import H2O
from .model_solids import NaCl
from .phase import PhaseField

__all__: List[str] = ["SaltWater", "Water"]


class SaltWater(PhaseField):
    def __init__(self, name: str, gb: "pp.GridBucket") -> None:
        super().__init__(name, gb)
        # saving external reference for simplicity
        self.water = H2O(gb)
        self.salt = NaCl(gb)
        # adding 'internally' to use parent class functions
        self.add_substance(self.water)
        self.add_substance(self.salt)

    def molar_density(
        self,
        pressure: "pp.ad.MergedVariable",
        enthalpy: "pp.ad.MergedVariable",
        temperature: Optional[Union["pp.ad.MergedVariable", None]] = None,
    ) -> "pp.ad.Operator":
        return pp.ad.Array(np.ones(self.gb.num_cells()))

    def enthalpy(
        self,
        pressure: "pp.ad.MergedVariable",
        enthalpy: "pp.ad.MergedVariable",
        temperature: Optional[Union["pp.ad.MergedVariable", None]] = None,
    ) -> "pp.ad.Operator":
        return pp.ad.Array(np.ones(self.gb.num_cells()))

    def dynamic_viscosity(
        self, pressure: "pp.ad.MergedVariable", enthalpy: "pp.ad.MergedVariable"
    ) -> "pp.ad.Operator":
        return pp.ad.Array(np.ones(self.gb.num_cells()))  # 0.001

    def thermal_conductivity(self, pressure: float, enthalpy: float) -> float:
        return pp.ad.Array(np.ones(self.gb.num_cells()))


class Water(PhaseField):
    """Values found on Wikipedia..."""

    def __init__(self, name: str, gb: "pp.GridBucket") -> None:
        super().__init__(name, gb)
        # saving external reference for simplicity
        self.water = H2O(gb)
        # adding 'internally' to use parent class functions
        self.add_substance(self.water)

    def molar_density(
        self,
        pressure: "pp.ad.MergedVariable",
        enthalpy: "pp.ad.MergedVariable",
        temperature: Optional[Union["pp.ad.MergedVariable", None]] = None,
    ) -> "pp.ad.Operator":
        return pp.ad.Array(np.ones(self.gb.num_cells()))

    def enthalpy(
        self,
        pressure: "pp.ad.MergedVariable",
        enthalpy: "pp.ad.MergedVariable",
        temperature: Optional[Union["pp.ad.MergedVariable", None]] = None,
    ) -> "pp.ad.Operator":
        return pp.ad.Array(np.ones(self.gb.num_cells()))

    def dynamic_viscosity(
        self, pressure: "pp.ad.MergedVariable", enthalpy: "pp.ad.MergedVariable"
    ) -> "pp.ad.Operator":
        return pp.ad.Array(np.ones(self.gb.num_cells()))  # 0.0003

    def thermal_conductivity(
        self, pressure: "pp.ad.MergedVariable", enthalpy: "pp.ad.MergedVariable"
    ) -> "pp.ad.Operator":
        return pp.ad.Array(np.ones(self.gb.num_cells()))  # 0.05
