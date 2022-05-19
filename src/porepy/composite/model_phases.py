"""Contains concrete implementation of phases."""

from typing import List

import iapws

from .phase import PhaseField
from .model_fluids import H20_iapws
from .model_solids import NaCl_simple

__all__: List[str] = ["SaltWater_iapws", "WaterVapor_iapws"]


class SaltWater_iapws(PhaseField):
    pass


class WaterVapor_iapws(PhaseField):
    pass
