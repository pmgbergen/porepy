"""
For any discretization class compatible with PorePy, wrap_discretization associates
a discretization with all attributes of the class' attributes that end with
'_matrix_key'.


Example:
    # Generate grid
    >>> g = pp.CartGrid([2, 2])
    # Associate an Ad representation of an Mpfa method, aimed this grid
    >>> discr = MpfaAd(keyword='flow', grids=[g])
    # The flux discretization of Mpfa can now be accesed by
    >>> discr.flux
    # While the discretization of boundary conditions is available by
    >>> discr.bound_flux.

    The representation of different discretization objects can be combined with other
    Ad objects into an operator tree, using lazy evaluation.

    It is assumed that the actual action of discretization (creation of the
    discretization matrices) is performed before the operator tree is parsed.
"""
import abc
from typing import List, Tuple, Union

import numpy as np

import porepy as pp

from ._ad_utils import MergedOperator, wrap_discretization

__all__ = [
    "Discretization",
    "BiotAd",
    "MpsaAd",
    "GradPAd",
    "DivUAd",
    "BiotStabilizationAd",
    "ColoumbContactAd",
    "MpfaAd",
    "TpfaAd",
    "MassMatrixAd",
    "UpwindAd",
    "RobinCouplingAd",
    "WellCouplingAd",
    "UpwindCouplingAd",
]

Edge = Tuple[pp.Grid, pp.Grid]


class Discretization(abc.ABC):
    """General/utility methods for AD discretization classes.

    The init of the children classes below typically calls wrap_discretization
    and has arguments including grids or edges and keywords for parameter and
    possibly matrix storage.

    """

    def __init__(self):
        """"""

        self._discretization: Union[
            "pp.numerics.discretization.Discretization",
            "pp.numerics.interface_laws.abstract_interface_law.AbstractInterfaceLaw",
        ]
        self.mat_dict_key: str
        self.keyword = str

        # Get the name of this discretization.
        self._name: str
        self.grids: List[pp.Grid]
        self.edges: List[Edge]

    def __repr__(self) -> str:
        s = f"Ad discretization of type {self._name}. Defined on {len(self.grids)} grids"
        return s

    def __str__(self) -> str:
        return f"{self._name}({self.keyword})"


### Mechanics related discretizations


class BiotAd(Discretization):
    """Ad wrapper around the Biot discretization class.

    For description of the method, we refer to the standard Biot class.

    """

    def __init__(self, keyword: str, grids: List[pp.Grid]) -> None:
        self.grids = grids
        self._discretization = pp.Biot(keyword)
        self._name = "BiotMpsa"

        self.keyword = keyword

        # Declear attributes, these will be initialized by the below call to the
        # discretization wrapper.

        self.stress: MergedOperator
        self.bound_stress: MergedOperator
        self.bound_displacement_cell: MergedOperator
        self.bound_displacement_face: MergedOperator

        self.div_u: MergedOperator
        self.bound_div_u: MergedOperator
        self.grad_p: MergedOperator
        self.stabilization: MergedOperator
        self.bound_pressure: MergedOperator

        wrap_discretization(
            obj=self, discr=self._discretization, grids=grids, mat_dict_key=self.keyword
        )


class MpsaAd(Discretization):
    def __init__(self, keyword: str, grids: List[pp.Grid]) -> None:
        self.grids = grids
        self._discretization = pp.Mpsa(keyword)
        self._name = "Mpsa"

        self.keyword = keyword

        # Declear attributes, these will be initialized by the below call to the
        # discretization wrapper.

        self.stress: MergedOperator
        self.bound_stress: MergedOperator
        self.bound_displacement_cell: MergedOperator
        self.bound_displacement_face: MergedOperator

        wrap_discretization(self, self._discretization, grids=grids)


class GradPAd(Discretization):
    def __init__(self, keyword: str, grids: List[pp.Grid]) -> None:
        self.grids = grids
        self._discretization = pp.GradP(keyword)
        self._name = "GradP from Biot"
        self.keyword = keyword

        self.grad_p: MergedOperator

        wrap_discretization(self, self._discretization, grids=grids)


class DivUAd(Discretization):
    def __init__(
        self, keyword: str, grids: List[pp.Grid], mat_dict_keyword: str
    ) -> None:
        self.grids = grids
        self._discretization = pp.DivU(keyword, mat_dict_keyword)

        self._name = "DivU from Biot"
        self.keyword = mat_dict_keyword

        self.div_u: MergedOperator
        self.bound_div_u: MergedOperator

        wrap_discretization(
            self, self._discretization, grids=grids, mat_dict_key=mat_dict_keyword
        )


class BiotStabilizationAd(Discretization):
    def __init__(self, keyword: str, grids: List[pp.Grid]) -> None:
        self.grids = grids
        self._discretization = pp.BiotStabilization(keyword)
        self._name = "Biot stabilization term"
        self.keyword = keyword

        self.stabilization: MergedOperator

        wrap_discretization(self, self._discretization, grids=grids)


class ColoumbContactAd(Discretization):
    def __init__(self, keyword: str, edges: List[Edge]) -> None:
        self.edges = edges

        # Special treatment is needed to cover the case when the edge list happens to
        # be empty.
        if len(edges) > 0:
            dim = np.unique([e[0].dim for e in edges])

            low_dim_grids = [e[1] for e in edges]
            if not dim.size == 1:
                raise ValueError(
                    "Expected unique dimension of grids with contact problems"
                )
        else:
            # The assigned dimension value should never be used for anything, so we
            # set a negative value to indicate this (not sure how the parameter is used)
            # in the real contact discretization.
            dim = [-1]
            low_dim_grids = []

        self._discretization = pp.ColoumbContact(
            keyword, ambient_dimension=dim[0], discr_h=pp.Mpsa(keyword)
        )
        self._name = "Biot stabilization term"
        self.keyword = keyword

        self.traction: MergedOperator
        self.displacement: MergedOperator
        self.rhs: MergedOperator
        wrap_discretization(
            self, self._discretization, edges=edges, mat_dict_grids=low_dim_grids
        )


## Flow related


class MpfaAd(Discretization):
    def __init__(self, keyword: str, grids: List[pp.Grid]) -> None:
        self.grids = grids
        self._discretization = pp.Mpfa(keyword)
        self._name = "Mpfa"
        self.keyword = keyword

        self.flux: MergedOperator
        self.bound_flux: MergedOperator
        self.bound_pressure_cell: MergedOperator
        self.bound_pressure_face: MergedOperator
        self.vector_source: MergedOperator
        self.bound_pressure_vector_source: MergedOperator

        wrap_discretization(self, self._discretization, grids=grids)


class TpfaAd(Discretization):
    def __init__(self, keyword: str, grids: List[pp.Grid]) -> None:
        self.grids = grids
        self._discretization = pp.Tpfa(keyword)
        self._name = "Tpfa"
        self.keyword = keyword

        self.flux: MergedOperator
        self.bound_flux: MergedOperator
        self.bound_pressure_cell: MergedOperator
        self.bound_pressure_face: MergedOperator
        self.vector_source: MergedOperator
        self.bound_pressure_vector_source: MergedOperator

        wrap_discretization(self, self._discretization, grids=grids)


class MassMatrixAd(Discretization):
    def __init__(self, keyword: str, grids: List[pp.Grid]) -> None:
        self.grids = grids
        self._discretization = pp.MassMatrix(keyword)
        self._name = "Mass matrix"
        self.keyword = keyword

        self.mass: MergedOperator
        wrap_discretization(self, self._discretization, grids=grids)


class UpwindAd(Discretization):
    def __init__(self, keyword: str, grids: List[pp.Grid]) -> None:
        self.grids = grids
        self._discretization = pp.Upwind(keyword)
        self._name = "Upwind"
        self.keyword = keyword

        self.upwind: MergedOperator
        self.bound_transport_dir: MergedOperator
        self.bound_transport_neu: MergedOperator
        wrap_discretization(self, self._discretization, grids=grids)


## Interface coupling discretizations


class WellCouplingAd(Discretization):
    def __init__(self, keyword: str, edges: List[Edge]) -> None:
        self.edges = edges
        self._discretization = pp.WellCoupling(keyword, primary_keyword=keyword)
        self._name = "Well interface coupling"
        self.keyword = keyword

        self.well_discr: MergedOperator
        self.well_vector_source: MergedOperator
        wrap_discretization(self, self._discretization, edges=edges)

    def __repr__(self) -> str:
        s = (
            f"Ad discretization of type {self._name}."
            f"Defined on {len(self.edges)} mortar grids."
        )
        return s


class RobinCouplingAd(Discretization):
    def __init__(self, keyword: str, edges: List[Edge]) -> None:
        self.edges = edges
        self._discretization = pp.RobinCoupling(keyword, primary_keyword=keyword)
        self._name = "Robin interface coupling"
        self.keyword = keyword

        self.mortar_discr: MergedOperator
        self.mortar_vector_source: MergedOperator
        wrap_discretization(self, self._discretization, edges=edges)

    def __repr__(self) -> str:
        s = (
            f"Ad discretization of type {self._name}."
            f"Defined on {len(self.edges)} mortar grids."
        )
        return s


class UpwindCouplingAd(Discretization):
    def __init__(self, keyword: str, edges: List[Edge]) -> None:
        self.edges = edges
        self._discretization = pp.UpwindCoupling(keyword)
        self._name = "Upwind coupling"
        self.keyword = keyword

        # UpwindCoupling also has discretization matrices for (inverse) trace.
        # These are not needed for Ad version since ad.Trace should be used instead
        self.mortar_discr: MergedOperator
        self.flux: MergedOperator
        self.upwind_primary: MergedOperator
        self.upwind_secondary: MergedOperator
        wrap_discretization(self, self._discretization, edges=edges)

    def __repr__(self) -> str:
        s = (
            f"Ad discretization of type {self._name}."
            f"Defined on {len(self.edges)} mortar grids."
        )
        return s
