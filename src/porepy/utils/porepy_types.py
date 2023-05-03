"""
Defines types commonly used in PorePy.
"""
from typing import Union

import porepy as pp

all = ["number", "GridLike", "discretization_type", "fracture_network"]

GridLike = Union["pp.Grid", "pp.MortarGrid"]
"""Type for grids and mortar grids."""

number = Union[float, int]
"""Type for numbers."""

discretization_type = Union[
    "pp.numerics.discretization.Discretization",
    "pp.numerics.interface_laws.abstract_interface_law.AbstractInterfaceLaw",
]

fracture_network = Union[
    "pp.fracs.fracture_network_2d.FractureNetwork2d",
    "pp.fracs.fracture_network_3d.FractureNetwork3d",
]
