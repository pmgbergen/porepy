"""
Defines types commonly used in PorePy.
"""
from typing import Union

import porepy as pp

all = ["number", "GridLike"]
GridLike = Union["pp.Grid", "pp.MortarGrid"]
"""Type for grids and mortar grids."""

number = Union[float, int]
"""Type for numbers."""
