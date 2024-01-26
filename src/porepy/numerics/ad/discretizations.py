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

from __future__ import annotations

import abc

import porepy as pp
from porepy.utils.porepy_types import discretization_type

from ._ad_utils import MergedOperator, wrap_discretization

__all__ = [
    "Discretization",
    "BiotAd",
    "MpsaAd",
    "GradPAd",
    "DivUAd",
    "BiotStabilizationAd",
    "MpfaAd",
    "TpfaAd",
    "UpwindAd",
    "UpwindCouplingAd",
]


class Discretization(abc.ABC):
    """General/utility methods for AD discretization classes.

    The init of the children classes below typically calls wrap_discretization
    and has arguments including subdomains or interfaces and keywords for parameter and
    possibly matrix storage.

    """

    def __init__(self) -> None:
        self._discretization: discretization_type
        """The discretization object, which is wrapped by this class."""
        self.mat_dict_key: str
        """Keyword for matrix storage in the data dictionary."""
        self.keyword: str
        """Keyword for parameter storage in the data dictionary."""

        self.subdomains: list[pp.Grid]
        """List of grids on which the discretization is defined."""
        self.interfaces: list[pp.MortarGrid]
        """List of interfaces on which the discretization is defined."""

        self._name: str

    def __repr__(self) -> str:
        s = f"""
        Ad discretization of type {self._name}. Defined on {len(self.subdomains)}
        subdomains and {len(self.interfaces)} interfaces.
        """
        return s

    def __str__(self) -> str:
        return f"{self._name}({self.keyword})"

    @property
    def name(self) -> str:
        """Name of the discretization."""
        return self._name


# Mechanics related discretizations


class BiotAd(Discretization):
    """Ad wrapper around the Biot discretization class.

    For description of the method, we refer to the standard Biot class.

    """

    def __init__(
        self, keyword: str, subdomains: list[pp.Grid], coupling_keywords: list[str]
    ) -> None:
        self.subdomains = subdomains
        self._discretization = pp.Biot(coupling_keywords, keyword)
        self._name = "BiotMpsa"

        self.keyword = keyword

        # Declare attributes, these will be initialized by the below call to the
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

        # The following are keywords used to identify the coupling terms constructed
        # by a Biot discretization.
        coupling_terms = ['div_u', 'bound_div_u', 'grad_p', 'bound_pressure', 'stabilization']

        wrap_discretization(
            obj=self,
            discr=self._discretization,
            subdomains=subdomains,
            coupling_terms=coupling_terms,
        )


class MpsaAd(Discretization):
    def __init__(self, keyword: str, subdomains: list[pp.Grid]) -> None:
        self.subdomains = subdomains
        self._discretization = pp.Mpsa(keyword)
        self._name = "Mpsa"

        self.keyword = keyword

        # Declare attributes, these will be initialized by the below call to the
        # discretization wrapper.

        self.stress: MergedOperator
        self.bound_stress: MergedOperator
        self.bound_displacement_cell: MergedOperator
        self.bound_displacement_face: MergedOperator

        wrap_discretization(self, self._discretization, subdomains=subdomains)


class GradPAd(Discretization):
    def __init__(self, keyword: str, subdomains: list[pp.Grid]) -> None:
        self.subdomains = subdomains
        self._discretization = pp.GradP(keyword)
        self._name = "GradP from Biot"
        self.keyword = keyword

        self.grad_p: MergedOperator

        wrap_discretization(self, self._discretization, subdomains=subdomains)


class DivUAd(Discretization):
    def __init__(
        self, keyword: str, subdomains: list[pp.Grid], mat_dict_keyword: str
    ) -> None:
        self.subdomains = subdomains
        self._discretization = pp.DivU(keyword, mat_dict_keyword)

        self._name = "DivU from Biot"
        self.keyword = mat_dict_keyword

        self.div_u: MergedOperator
        self.bound_div_u: MergedOperator

        wrap_discretization(
            self,
            self._discretization,
            subdomains=subdomains,
            mat_dict_key=mat_dict_keyword,
        )


class BiotStabilizationAd(Discretization):
    def __init__(self, keyword: str, subdomains: list[pp.Grid]) -> None:
        self.subdomains = subdomains
        self._discretization = pp.BiotStabilization(keyword)
        self._name = "Biot stabilization term"
        self.keyword = keyword

        self.stabilization: MergedOperator

        wrap_discretization(self, self._discretization, subdomains=subdomains)


## Flow related


class MpfaAd(Discretization):
    def __init__(self, keyword: str, subdomains: list[pp.Grid]) -> None:
        self.subdomains = subdomains
        self._discretization = pp.Mpfa(keyword)
        self._name = "Mpfa"
        self.keyword = keyword

        self.flux: MergedOperator
        self.bound_flux: MergedOperator
        self.bound_pressure_cell: MergedOperator
        self.bound_pressure_face: MergedOperator
        self.vector_source: MergedOperator
        self.bound_pressure_vector_source: MergedOperator

        wrap_discretization(self, self._discretization, subdomains=subdomains)


class TpfaAd(Discretization):
    def __init__(self, keyword: str, subdomains: list[pp.Grid]) -> None:
        self.subdomains = subdomains
        self._discretization = pp.Tpfa(keyword)
        self._name = "Tpfa"
        self.keyword = keyword

        self.flux: MergedOperator
        self.bound_flux: MergedOperator
        self.bound_pressure_cell: MergedOperator
        self.bound_pressure_face: MergedOperator
        self.vector_source: MergedOperator
        self.bound_pressure_vector_source: MergedOperator

        wrap_discretization(self, self._discretization, subdomains=subdomains)


class UpwindAd(Discretization):
    def __init__(self, keyword: str, subdomains: list[pp.Grid]) -> None:
        self.subdomains = subdomains
        self._discretization = pp.Upwind(keyword)
        self._name = "Upwind"
        self.keyword = keyword

        self.upwind: MergedOperator
        self.bound_transport_dir: MergedOperator
        self.bound_transport_neu: MergedOperator
        wrap_discretization(self, self._discretization, subdomains=subdomains)


class UpwindCouplingAd(Discretization):
    def __init__(self, keyword: str, interfaces: list[pp.MortarGrid]) -> None:
        self.interfaces = interfaces
        self._discretization = pp.UpwindCoupling(keyword)
        self._name = "Upwind coupling"
        self.keyword = keyword

        # UpwindCoupling also has discretization matrices for (inverse) trace.
        # These are not needed for Ad version since ad.Trace should be used instead
        self.mortar_discr: MergedOperator
        self.flux: MergedOperator
        self.upwind_primary: MergedOperator
        self.upwind_secondary: MergedOperator
        wrap_discretization(self, self._discretization, interfaces=interfaces)

    def __repr__(self) -> str:
        s = (
            f"Ad discretization of type {self._name}."
            f"Defined on {len(self.interfaces)} interfaces."
        )
        return s
