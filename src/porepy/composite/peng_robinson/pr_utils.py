"""This module contains utility functionality and parameters for the
Peng-Robinson EoS."""
from __future__ import annotations

from typing import Any

import numpy as np

import porepy as pp

__all__ = [
    "A_CRIT",
    "B_CRIT",
    "Z_CRIT",
]

# AD functions used throughout PR, instantiated only once here
_power = pp.ad.Function(pp.ad.power, "power")
_exp = pp.ad.Function(pp.ad.exp, "exp")
_sqrt = pp.ad.Function(pp.ad.sqrt, "sqrt")
_log = pp.ad.Function(pp.ad.log, "ln")


A_CRIT: float = (
    1
    / 512
    * (
        -59
        + 3 * np.cbrt(276231 - 192512 * np.sqrt(2))
        + 3 * np.cbrt(276231 + 192512 * np.sqrt(2))
    )
)
"""Critical attraction value in the Peng-Robinson EoS, ~ 0.457235529."""

B_CRIT: float = (
    1
    / 32
    * (-1 - 3 * np.cbrt(16 * np.sqrt(2) - 13) + 3 * np.cbrt(16 * np.sqrt(2) + 13))
)
"""Critical co-volume in the Peng-Robinson EoS, ~ 0.077796073."""

Z_CRIT: float = (
    1 / 32 * (11 + np.cbrt(16 * np.sqrt(2) - 13) - np.cbrt(16 * np.sqrt(2) + 13))
)
"""Critical compressibility factor in the Peng-Robinson EoS, ~ 0.307401308."""


class Leaf(pp.ad.Operator):
    """A leaf-operator returning an assigned value whenever it is parsed."""

    def __init__(self, name: str = "") -> None:
        super().__init__(name=name)

        self.value: Any = 0.0
        """The numerical value of this operator."""

    def parse(self, mdg: pp.MixedDimensionalGrid) -> Any:
        """Returns the value assigned to this operator."""
        self.value
