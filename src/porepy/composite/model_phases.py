"""Contains concrete implementation of phases."""

from typing import List, Optional, Union

from iapws import IAPWS95, SeaWater

import porepy as pp

from .model_fluids import H20_iapws
from .model_solids import Salt_simple
from .phase import PhaseField

__all__: List[str] = ["SaltWaterPhase_iapws", "WaterPhase_iapws"]


class SaltWaterPhase_iapws(PhaseField):
    
    def __init__(self, name: str, gb: "pp.GridBucket") -> None:
        super().__init__(name, gb)
        # saving external reference for simplicity
        self.salt = Salt_simple(gb)
        self.water = H20_iapws(gb)
        # adding 'internally' to use parent class functions
        self.add_substance(self.water)
        self.add_substance(self.salt)
    
    @pp.ad.ADWrapper
    def molar_density(
        self, pressure: "pp.ad.Operator", enthalpy: "pp.ad.Operator"
    ) -> "pp.ad.Operator":
        return self.water.molar_density(pressure, enthalpy)
    
    def enthalpy(
        self,
        pressure: "pp.ad.Operator",
        composit_enthalpy: "pp.ad.Operator",
        temperature: Optional[Union["pp.ad.Operator", None]] = None
    ) -> "pp.ad.Operator":
        return self.saturation * composit_enthalpy

class WaterPhase_iapws(PhaseField):
    pass
