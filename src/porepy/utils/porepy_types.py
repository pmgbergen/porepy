"""
Defines types commonly used in PorePy.
"""

from typing import Callable, Sequence, Union

import porepy as pp

__all__ = [
    "number",
    "GridLike",
    "GridLikeSequence",
    "SubdomainsOrBoundaries",
    "discretization_type",
    "fracture_network",
    "DomainFunctionType",
    "ExtendedDomainFunctionType",
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

DomainFunctionType = Callable[[SubdomainsOrBoundaries], "pp.ad.Operator"]
"""Type alias to denote thermodynamic properties and variables which are defined on
subdomains or boundaries and return an AD-compatible representation.

Motivated by PorePy's modelling framework, terms appearing in model equations are
defined on some domain and represented as an AD operator.

Notes:
    1. Boundaries are included because the various terms can indeed be called with
       boundary grids in the advective part.
    2. Interfaces (mortar grids) are explicitly excluded, since this is part of the
       constitutive modelling in mD and requires separate solutions.

"""

ExtendedDomainFunctionType = Union[DomainFunctionType, "pp.ad.SurrogateFactory"]
"""Extending :data:`DomainFunctionType` to include primarely phase properties, which
can be given by :class:`~porepy.numerics.ad.surrogate_operator.SurrogateFactory` to
accomodate externalized computations."""
