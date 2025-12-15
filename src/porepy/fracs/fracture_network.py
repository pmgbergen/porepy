from __future__ import annotations

import warnings
from typing import Optional, Union, cast, TYPE_CHECKING

import porepy as pp

# Custom typings
FractureList = Optional[
    list[pp.LineFracture] | list[pp.PlaneFracture | pp.EllipticFracture]
]
