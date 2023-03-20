"""
Defines types commonly used in PorePy.
"""
from typing import Union

import porepy as pp

all = ["number", "GridLike", "discretization_type"]
GridLike = Union["pp.Grid", "pp.MortarGrid"]
"""Type for grids and mortar grids."""

number = Union[float, int]
"""Type for numbers."""

discretization_type = Union[
    "pp.numerics.discretization.Discretization",
    "pp.numerics.interface_laws.abstract_interface_law.AbstractInterfaceLaw",
]
