"""
Utility functions for the AD package.

Functions:
    concatenate_ad_arrays: Concatenates a sequence of AD arrays into a single AD Array
        along a specified axis.

    wrap_discretization: Convert a discretization to its ad equivalent.

    uniquify_discretization_list: Define a unique list of discretization-keyword
        combinations from a list of AD discretizations.

    discretize_from_list: Perform the actual discretization for a list of AD
        discretizations.

Classes:
    MergedOperator: Representation of specific discretization fields for an AD
        discretization or a set of AD discretizations.

"""

from __future__ import annotations

import warnings
from abc import ABCMeta
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.discretization import Discretization, InterfaceDiscretization

from . import operators

__all__ = [
    "concatenate_ad_arrays",
    "wrap_discretization",
    "uniquify_discretization_list",
    "discretize_from_list",
    "MergedOperator",
]


def concatenate_ad_arrays(ad_arrays: list[pp.ad.AdArray], axis=0):
    """Concatenates a sequence of AD arrays into a single AD Array along a specified
    axis."""
    msg = "This functionality is deprecated and will be removed in a future version"
    warnings.warn(msg, DeprecationWarning)
    vals = [var.val for var in ad_arrays]
    jacs = np.array([var.jac for var in ad_arrays])

    vals_stacked = np.concatenate(vals, axis=axis)
    jacs_stacked = sps.vstack(jacs)

    return pp.ad.AdArray(vals_stacked, jacs_stacked)


def wrap_discretization(
    obj: pp.ad.DiscretizationAd,
    discr: Discretization | InterfaceDiscretization,
    subdomains: Optional[list[pp.Grid]] = None,
    interfaces: Optional[list[pp.MortarGrid]] = None,
    coupling_terms: Optional[list[str]] = None,
):
    """Convert a discretization to its AD equivalent.

    For a (non-ad) discretization object ``D`` of type ``discr``, this function will
    identify all attributes of the form ``"foo_matrix_key"`` and create a corresponding
    method "foo" in the AD discretization ``obj``. Thus, after the call to this method,
    ``obj.foo()`` will represent the discretization matrix for the term ``foo``.

    For example: If ``D`` is an instance of ``Mpfa`` (which has an attribute
    ``flux_matrix_key``), then ``obj`` will be an instance of ``MpfaAd``, and this
    function equips ``obj`` with the method ``obj.flux()``. This method will return the
    discretization matrix for the flux term, for the parameter keyword associated with
    ``obj``. NOTE: For discretizations that involve coupling terms (the only known
    example is Biot), the coupling terms are treated differently, see description of the
    coupling_keywords and coupling_terms arguments below.

    Parameters:
        obj: An AD discretization object. discr: A non-AD discretization object.
        subdomains: List of grids on which the discretization is defined. interfaces:
        List of interfaces on which the discretization is defined.

        Either subdomains or interfaces must be provided, but not both.

        coupling_terms: List of (multiphysics) coupling terms provided by this
            discretization. For instance, for a Biot discretization, this would be
            ['displacement_divergence', 'bound_displacement_divergence',
            'bound_pressure', 'consistency', 'scalar_gradient'].

        The coupling keywords and coupling terms are combined in this wrapper, so that
        if ``obj`` has coupling terms ``foo`` and ``bar``, with a coupling keywords
        ``baz`` and ``qux``, then:
            * ``obj.foo('baz')`` and ``obj.bar('baz')`` will be instances of
                ``MergedOperator``, referring to the discretization matrices for ``foo``
                and ``bar``, for the ``baz`` physics.
            * ``obj.foo('baz')`` and ``obj.foo('qux')`` will be *separate* instances of
                ``MergedOperator``, referring to the discretization matrices for ``foo``
                for the ``baz`` and ``qux`` physics, respectively.
        Good luck digesting that - rather see how this is used in constitutive laws
        (search for BiotAd).

    """
    # The purpose of this function is to create a set of MergedOperator instances
    # that represent the discretization matrices for the discretization ``discr``. To
    # that end, we first identify all attributes of the form "foo_matrix_key" in the
    # discretization class, and create a MergerOperator instance for each of these
    # (the MergedOperator is a wrapper around the discretization matrices for a domain,
    # that is, a set of grids). Next, we assign these MergedOperator instances to
    # ``obj`` in the form of methods, so that ``obj.foo()`` will return the
    # MergedOperator instance for the discretization matrix for the term ``foo``.
    # Accounting for coupling terms makes each of these steps a bit more complicated.

    # Process the domains
    domains: pp.GridLikeSequence
    if subdomains is None:
        # This is an interface discretization
        if interfaces is None:
            raise ValueError("Either subdomains or interfaces must be provided")
        if not isinstance(interfaces, list):
            raise ValueError("Interfaces must be a list")

        domains = interfaces
    elif interfaces is None:
        # This is a subdomain discretization
        if not isinstance(subdomains, list):
            raise ValueError("Subdomains must be a list")
        domains = subdomains
    else:
        raise ValueError("Either subdomains or interfaces must be provided, not both")

    if coupling_terms is None:
        coupling_terms = []

    # Loop over all discretizations, identify all attributes that ends with
    # "_matrix_key". These will be taken as discretizations (they are discretization
    # matrices for specific terms, to be).
    discretization_term_key = []
    for s in dir(discr):
        if s.endswith("_matrix_key"):
            key = s[:-11]
            discretization_term_key.append(key)

    # Storage for which MergedOperator instances are associated with which terms *and*
    # (for coupling terms) which physics keywords.
    operators: dict[str, dict[str, MergedOperator]] = {}

    # Loop over all identified terms, assign a MergedOperator to non-coupling terms,
    # while postponing the treatment of coupling terms.
    for discretization_key in discretization_term_key:
        operators[discretization_key] = {}

        # Fetch all physics keywords associated with this discretization term. The
        # default option is that the only keyword is that of the base discretization
        # class.
        if discretization_key in coupling_terms:
            # This is a coupling term, which will receive special treatment below.
            continue
        else:
            # Create the merged operator that represents this discretization matrix
            op = MergedOperator(
                discr=discr,
                discretization_matrix_key=discretization_key,
                physics_key=discr.keyword,
                domains=domains,
            )
            # Store the new
            operators[discretization_key].update({discr.keyword: op})

    def from_single(discr_list):
        # Helper function for creating methods for a standard term.
        def get_discr():
            # From the list of discretizations, return the one corresponding to the
            # provided keyword.
            return list(discr_list.values())[0]

        return get_discr

    def get_merged_operator(discr_keyword):
        # Helper function for creating a merged operator for a coupling term.
        def get_discr(inner_physics_key):
            # Return the discretization matrix for the provided physics keyword.
            op = MergedOperator(
                discr=discr,
                discretization_matrix_key=discr_keyword,
                physics_key=discr.keyword,
                inner_physics_key=inner_physics_key,
                domains=domains,
            )
            return op

        return get_discr

    for key, discretization_list in operators.items():
        if key in coupling_terms:
            # This is a coupling term, we need to create a method for this term that
            # returns the discretization for the provided physics keyword.
            func = get_merged_operator(key)
        else:
            # This is a standard term. It turned out that it was necessary to assign the
            # discretization as a method to the object via a function.
            func = from_single(discretization_list)

        # Assign the discretization as a method to the object.
        setattr(obj, key, func)


def uniquify_discretization_list(
    all_discr: list[MergedOperator],
) -> dict[pp.discretization_type, list[pp.GridLike]]:
    """From a list of Ad discretizations (in an Operator), define a unique list
    of discretization-keyword combinations.

    The intention is to avoid that what is essentially the same discretization
    operation is executed twice. For instance, if the list all_discr contains
    elements

        Mpfa(key1).flux, Mpfa(key2).flux and Mpfa(key1).bound_flux,

    where key1 and key2 are different parameter keywords, the function will
    register Mpfa(key1) and Mpfa(key2) (since these use data specified by different
    parameter keywords) but ignore the second instance Mpfa(key1), since this
    discretization is already registered.

    """
    unique_discr_grids: dict[pp.discretization_type, list[pp.GridLike]] = dict()

    # Mapping from discretization classes to the discretization.
    # We needed this for some reason.
    cls_obj_map: dict[ABCMeta, pp.discretization_type] = {}
    # List of all combinations of discretizations and parameter keywords covered.
    cls_key_covered = []
    for discr in all_discr:
        # Get the class of the underlying discretization, so MpfaAd will return Mpfa.
        cls = discr._discr.__class__
        # Parameter keyword for this discretization
        param_keyword = discr._discr.keyword

        # This discretization-keyword combination
        key = (cls, param_keyword)

        if key in cls_key_covered:
            # If this has been encountered before, we add grids not earlier
            # associated with this discretization to the existing list.
            # Map from discretization class to Ad discretization
            d = cls_obj_map[cls]
            for g in discr.domains:
                if g not in unique_discr_grids[d]:
                    unique_discr_grids[d].append(g)
        else:
            # Take note we have now encountered this discretization and parameter
            # keyword.
            cls_obj_map[cls] = discr._discr
            cls_key_covered.append(key)

            # Add new discretization with associated list of grids.
            # Need a copy here to avoid assigning additional grids to this
            # discretization (if not copy, this may happen if
            # the key-discr combination is encountered a second time and the
            # code enters the if part of this if-else).
            grid_likes = discr.domains.copy()
            unique_discr_grids[discr._discr] = grid_likes

    return unique_discr_grids


def discretize_from_list(
    discretizations: dict,
    mdg: pp.MixedDimensionalGrid,
) -> None:
    """For a list of (ideally uniquified) discretizations, perform the actual
    discretization.
    """
    for discr in discretizations:
        # discr is a discretization (on node or interface in the MixedDimensionalGrid
        # sense)

        # Loop over all subdomains (or MixedDimensionalGrid edges), do discretization.
        for grid in discretizations[discr]:
            if isinstance(grid, pp.MortarGrid):
                data = mdg.interface_data(grid)  # type:ignore
                g_primary, g_secondary = mdg.interface_to_subdomain_pair(grid)
                d_primary = mdg.subdomain_data(g_primary)
                d_secondary = mdg.subdomain_data(g_secondary)
                discr.discretize(
                    g_primary, g_secondary, grid, d_primary, d_secondary, data
                )
            else:
                data = mdg.subdomain_data(grid)
                try:
                    discr.discretize(grid, data)
                except NotImplementedError:
                    # This will likely be GradP and other Biot discretizations
                    pass


class MergedOperator(operators.Operator):
    """Representation of specific discretization fields for an Ad discretization.

    This is the bridge between the representation of discretization classes, implemented
    in Discretization, and the matrices resulting from a discretization.

    Objects of this class should not be access directly, but rather through the
    Discretization class.

    """

    def __init__(
        self,
        discr: pp.discretization_type,
        discretization_matrix_key: str,
        physics_key: str,
        inner_physics_key: Optional[str] = None,
        domains: Optional[pp.GridLikeSequence] = None,
    ) -> None:
        """Initiate a merged discretization.

        Parameters:
            discr: Mapping between subdomains, or interfaces, where the discretization
                is applied, and the actual Discretization objects.
            discretization_matrix_key: Keyword that identifies this discretization
                matrix, e.g. for a class with an attribute foo_matrix_key, the key
                will be foo.
            physics_key: Keyword used to access discretization matrices.
            inner_physics_key: For nested matrix dicts, the inner key.
            domains: Domains on which the discretization is defined.

        """
        name = discr.__class__.__name__
        self._merged_domains: list[pp.GridLike] = list(domains) if domains else []

        # Infer operator source (column space) and target (row space) from the
        # discretization.
        op_source: Optional[operators.OperatorSpace] = None
        op_target: Optional[operators.OperatorSpace] = None
        if domains:
            domain_list = list(domains)
            nd = (
                domain_list[0].dim
                if domain_list and hasattr(domain_list[0], "dim")
                else 1
            )
            row_dof = discr.get_row_dof_info(discretization_matrix_key, nd=nd)
            col_dof = discr.get_col_dof_info(discretization_matrix_key, nd=nd)

            if col_dof:
                op_source = operators.OperatorSpace.from_domains(list(domains), col_dof)
            if row_dof:
                op_target = operators.OperatorSpace.from_domains(list(domains), row_dof)

        super().__init__(name=name, source=op_source, target=op_target)

        self._discretization_matrix_key = discretization_matrix_key
        self._discr = discr

        self._physics_key = physics_key
        self._inner_physics_key = inner_physics_key

    @property
    def domains(self) -> list[pp.GridLike]:
        return list(self._merged_domains)

    def __repr__(self) -> str:
        domain_label = (
            self.target.domain_type.value
            if self.target and self.target.domain_type
            else "unknown"
        )
        s = (
            f"Operator with key {self._discretization_matrix_key} defined on "
            f"{len(self.domains)} {domain_label}"
        )
        return s

    def __str__(self) -> str:
        return f"{self._name}({self._physics_key}).{self._discretization_matrix_key}"

    def _key(self) -> str:
        # Mypy occasionally (but not always, sigh) complains that it cannot
        # self._cached_key determine the type of self._cached_key, despite it being
        # decleared as an Optional[str] in the class definition.
        if self._cached_key is None:  # type: ignore[has-type]
            domain_ids = [domain.id for domain in self.domains]
            s = f"(Merged_operator, name={self.name}, domains={domain_ids})"
            s += f", discretization_matrix_key={self._discretization_matrix_key}"
            s += f", physics_key={self._physics_key}"
            if self._inner_physics_key is not None:
                s += f", inner_physics_key={self._inner_physics_key}"

            self._cached_key = s
        return self._cached_key

    def parse(self, mdg: pp.MixedDimensionalGrid) -> sps.spmatrix:
        """Convert a merged operator into a sparse matrix by concatenating
        discretization matrices.

        Parameters:
            mdg: Mixed-dimensional grid.

        Returns:
            sps.spmatrix: The merged discretization matrices for the associated matrix.

        """

        # Data structure for matrices.
        mat = []

        if len(self.domains) == 0:
            # The underlying discretization is constructed on an empty grid list, for
            # instance on a mixed-dimensional grid containing no mortar subdomains. We
            # can return an empty matrix.
            return sps.csc_matrix((0, 0))

        # Loop over all grid-discretization combinations, get hold of the discretization
        # matrix for this grid quantity.
        for grid in self.domains:
            # Get data dictionary for either grid or interface
            if isinstance(grid, pp.MortarGrid):
                data = mdg.interface_data(grid)
            elif isinstance(grid, pp.Grid):
                data = mdg.subdomain_data(grid)
            else:
                s = "Did not expect a discretization defined on a BoundaryGrid."
                raise ValueError(s)

            mat_dict: dict[str, sps.spmatrix] = data[  # type: ignore
                pp.DISCRETIZATION_MATRICES
            ][self._physics_key]

            # Get the submatrix for the right discretization.
            key = self._discretization_matrix_key
            mat_key = getattr(self._discr, key + "_matrix_key")
            if self._inner_physics_key is not None:
                local_mat = mat_dict[mat_key][self._inner_physics_key]
            else:
                local_mat = mat_dict[mat_key]
            mat.append(local_mat)

        if all([isinstance(m, np.ndarray) for m in mat]):
            # EK is almost sure this never happens, but leave this check for now.
            raise NotImplementedError("")

        else:
            # This is a standard discretization; wrap it in a diagonal sparse matrix.

            if all([m.format == "dia" for m in mat]):
                # If all matrices are of dia-format, we can try to use a special method
                # for forming a sparse dia matrix from the blocks. This is more
                # efficient than the below csr-based method. However, the matrices can
                # only be nonzero along their main diagonal. This condition does not
                # hold true for all dia-matrices, so there is a chance a ValueError may
                # be raised. If so, we let it pass and fall back to the csr-based
                # method.
                try:
                    return pp.matrix_operations.sparse_dia_from_sparse_blocks(mat)
                except ValueError:
                    # Use the csr-based method below.
                    pass

            return pp.matrix_operations.csr_matrix_from_sparse_blocks(mat)
