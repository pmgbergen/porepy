"""Contains concrete implementation of phases."""

import iapws

from typing import List

from .phase import PhaseField
from .fluid import H20_iapws
from .solid import NaCl_simple

__all__: List[str] = [
    "SaltWater_iapws",
    "WaterVapor_iapws"
]


class SaltWater_iapws(PhaseField):
    pass


class WaterVapor_iapws(PhaseField):
    pass