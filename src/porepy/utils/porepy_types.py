"""
Defines types commonly used in PorePy.
"""

from typing import Sequence, Union

import porepy as pp

__all__ = [
    "number",
    "GridLike",
    "GridLikeSequence",
    "SubdomainsOrBoundaries",
    "discretization_type",
    "fracture_network",
]

GridLike = Union["pp.Grid", "pp.MortarGrid", "pp.BoundaryGrid"]
"""Type for grids and mortar grids."""
SubdomainsOrBoundaries = Sequence["pp.Grid"] | Sequence["pp.BoundaryGrid"]
"""Type for sequence of subdomains or sequence of boundary grids."""

GridLikeSequence = SubdomainsOrBoundaries | Sequence["pp.MortarGrid"]
"""Type for sequence of any kind of grids, but not a mixture of them."""

number = Union[float, int]
"""Type for numbers."""

discretization_type = Union[
    "pp.numerics.discretization.Discretization",
    "pp.numerics.discretization.InterfaceDiscretization",
]

fracture_network = Union[
    "pp.fracs.fracture_network_2d.FractureNetwork2d",
    "pp.fracs.fracture_network_3d.FractureNetwork3d",
]
