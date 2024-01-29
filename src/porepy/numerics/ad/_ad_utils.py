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

    set_solution_values: Set values to the data dictionary providing parameter name and
        time step/iterate index.

    get_solution_values: Get values from the data dictionary providing parameter name
        and time step/iterate index.

Classes:
    MergedOperator: Representation of specific discretization fields for an AD
        discretization or a set of AD discretizations.

"""

from __future__ import annotations

from abc import ABCMeta
from typing import Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp

from . import operators
from .forward_mode import AdArray


def concatenate_ad_arrays(ad_arrays: list[AdArray], axis=0):
    """Concatenates a sequence of AD arrays into a single AD Array along a specified axis."""
    vals = [var.val for var in ad_arrays]
    jacs = np.array([var.jac for var in ad_arrays])

    vals_stacked = np.concatenate(vals, axis=axis)
    jacs_stacked = sps.vstack(jacs)

    return AdArray(vals_stacked, jacs_stacked)


def wrap_discretization(
    obj: pp.ad.Discretization,
    discr: pp.numerics.disrcetization.Discretization,
    subdomains: Optional[list[pp.Grid]] = None,
    interfaces: Optional[list[pp.MortarGrid]] = None,
    coupling_terms: Optional[list[str]] = None,
):
    """Convert a discretization to its AD equivalent.
    
    For a (non-ad) discretization object ``D`` of type ``discr``, this function will
    identify all attributes of the form ``"foo_matrix_key"`` and create a corresponding
    attribute "foo" in the AD discretization ``obj``. Thus, after the call to this
    method, ``obj.foo`` will represent the discretization matrix for the term ``foo``.
    
    For example: If ``D`` is an instance of ``Mpfa`` (which has an attribute
    ``flux_matrix_key``), then ``obj.flux`` will be an instance of ``MpfaAd``, and
    this function equips ``obj`` with the method ``obj.flux()``. This method will
    return the discretization matrix for the flux term, for the parameter keyword
    associated with ``obj``.
    NOTE: For discretizations that involve coupling terms (the only known example is
    Biot), the coupling terms are treated differently, see description of the
    coupling_keywords and coupling_terms arguments below.

    Parameters:
        obj: An AD discretization object.
        discr: A non-AD discretization object.
        subdomains: List of grids on which the discretization is defined.
        interfaces: List of interfaces on which the discretization is defined.

        Either subdomains or interfaces must be provided, but not both.

        coupling_terms: List of (multiphysics) coupling terms provided by this
            discretization. For instance, for a Biot discretization, this would be
            ['div_u', 'bound_div_u', 'bound_pressure', 'stabilization', 'grad_p'].
        
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
    operators: list[str, list[str]] = {}

    # Loop over all identified terms, identify relevant physics keywords and create a
    # MergedOperator instance for each combination of term and physics keyword.
    for discretization_key in discretization_term_key:

        operators[discretization_key] = {}

        # Fetch all physics keywords associated with this discretization term. The
        # default option is that the only keyword is that of the base discretization
        # class.
        if discretization_key in coupling_terms:
            continue
            physics_keys = coupling_keywords
        else:
            physics_keys = [discr.keyword]

        # Create the merged operator that represents this discretization matrix
        for phys_key in physics_keys:
            op = MergedOperator(
                discr=discr,
                discretization_matrix_key=discretization_key,
                physics_key=phys_key,
                domains=domains,
            )
            # Store the new 
            operators[discretization_key].update({phys_key: op})


    def from_single(discr_list):
        # Helper function for creating methods for coupling terms - it turned out that
        # this step was necessary to assign the discretization as a method to the
        # object.
        def set_discr():
            # From the list of discretizations, return the one corresponding to the
            # provided keyword.
            return list(discr_list.values())[0]
        return set_discr

    def from_coupled(discr_list):
        # Helper function for creating methods for coupling terms - it turned out that
        # this step was necessary to assign the discretization as a method to the
        # object.
        def set_discr(keyword):
            # From the list of discretizations, return the one corresponding to the
            # provided keyword.
            return discr_list[keyword]
        return set_discr

    def get_merged_operator(discr_keyword):
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
            #func = outer(discretization_list, key)
            func = get_merged_operator(key)
            debug = []
        else:
            # This is a standard term, we can just return the discretization for the
            # main physics keyword.
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
        if isinstance(discr._discr, pp.Biot):
            debug = []

        # This discretization-keyword combination
        key = (cls, param_keyword)

        if key in cls_key_covered:
            # If this has been encountered before, we add subdomains not earlier associated
            # with this discretization to the existing list.
            # of subdomains.
            # Map from discretization class to Ad discretization
            d = cls_obj_map[cls]
            for g in discr.subdomains:
                if g not in unique_discr_grids[d]:
                    unique_discr_grids[d].append(g)
            for e in discr.interfaces:
                if e not in unique_discr_grids[d]:
                    unique_discr_grids[d].append(e)
        else:
            # Take note we have now encountered this discretization and parameter keyword.
            cls_obj_map[cls] = discr._discr
            cls_key_covered.append(key)

            # Add new discretization with associated list of subdomains.
            # Need a copy here to avoid assigning additional subdomains to this
            # discretization (if not copy, this may happen if
            # the key-discr combination is encountered a second time and the
            # code enters the if part of this if-else).
            grid_likes = discr.subdomains.copy() + discr.interfaces.copy()
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
        # discr is a discretization (on node or interface in the MixedDimensionalGrid sense)

        # Loop over all subdomains (or MixedDimensionalGrid edges), do discretization.
        for g in discretizations[discr]:
            if isinstance(g, pp.MortarGrid):
                data = mdg.interface_data(g)  # type:ignore
                g_primary, g_secondary = mdg.interface_to_subdomain_pair(g)
                d_primary = mdg.subdomain_data(g_primary)
                d_secondary = mdg.subdomain_data(g_secondary)
                discr.discretize(
                    g_primary, g_secondary, g, d_primary, d_secondary, data
                )
            else:
                data = mdg.subdomain_data(g)
                try:
                    discr.discretize(g, data)
                except NotImplementedError:
                    # This will likely be GradP and other Biot discretizations
                    pass


def set_solution_values(
    name: str,
    values: np.ndarray,
    data: dict,
    time_step_index: Optional[int] = None,
    iterate_index: Optional[int] = None,
    additive: bool = False,
) -> None:
    """Function for setting values in the data dictionary.

    Parameters:
        name: Name of the quantity that is to be assigned values.
        values: The values that are set in the data dictionary.
        data: Data dictionary corresponding to the subdomain or interface in question.
        time_step_index (optional): Determines the key of where `values` are to be
            stored in `data[pp.TIME_STEP_SOLUTIONS][name]`. 0 means it is the most
            recent set of values, 1 means the one time step back in time, 2 is two time
            steps back, and so on.
        iterate_index (optional): Determines the key of where `values` are to be
            stored in `data[pp.ITERATE_SOLUTIONS][name]`. 0 means it is the most
            recent set of values, 1 means the values one iteration back in time, 2 is
            two iterations back, and so on.
        additive: Flag to decide whether the values already stored in the data
            dictionary should be added to or overwritten.

    Raises:
        ValueError: If neither of `time_step_index` or `iterate_index` have been
            assigned a non-None value.

    """
    if time_step_index is None and iterate_index is None:
        raise ValueError(
            "At least one of time_step_index and iterate_index needs to be different"
            " from None."
        )

    if not additive:
        if time_step_index is not None:
            if pp.TIME_STEP_SOLUTIONS not in data:
                data[pp.TIME_STEP_SOLUTIONS] = {}
            if name not in data[pp.TIME_STEP_SOLUTIONS]:
                data[pp.TIME_STEP_SOLUTIONS][name] = {}
            data[pp.TIME_STEP_SOLUTIONS][name][time_step_index] = values.copy()

        if iterate_index is not None:
            if pp.ITERATE_SOLUTIONS not in data:
                data[pp.ITERATE_SOLUTIONS] = {}
            if name not in data[pp.ITERATE_SOLUTIONS]:
                data[pp.ITERATE_SOLUTIONS][name] = {}
            data[pp.ITERATE_SOLUTIONS][name][iterate_index] = values.copy()
    else:
        if time_step_index is not None:
            data[pp.TIME_STEP_SOLUTIONS][name][time_step_index] += values

        if iterate_index is not None:
            data[pp.ITERATE_SOLUTIONS][name][iterate_index] += values


def get_solution_values(
    name: str,
    data: dict,
    time_step_index: Optional[int] = None,
    iterate_index: Optional[int] = None,
) -> np.ndarray:
    """Function for fetching values stored in the data dictionary.

    This function should be used for obtaining solution values that are not related to a
    variable. This is to avoid the cumbersome alternative of writing e.g.:
    `data["solution_name"][pp.TIME_STEP_SOLUTION/pp.ITERATE_SOLUTION][0]`.

    Parameters:
        name: Name of the parameter whose values we are interested in.
        data: The data dictionary.
        time_step_index: Which time step we want to get values for. 0 is current, 1 is
            one time step back in time. This is only limited by how many time steps are
            stored from before.
        iterate_index: Which iterate we want to get values for. 0 is current, 1 is one
            iterate back in time. This is only limited by how many iterates are stored
            from before.

    Raises:
        ValueError: If both time_step_index and iterate_index are None.

        ValueErorr: If both time_step_index and iterate_index are assigned a value.

        KeyError: If there are no data values assigned to the provided name.

        KeyError: If there are no data values assigned to the time step/iterate index.

    Returns:
        An array containing the solution values.

    """
    if time_step_index is None and iterate_index is None:
        raise ValueError("Both time_step_index and iterate_index cannot be None.")

    if time_step_index is not None and iterate_index is not None:
        raise ValueError(
            "Both time_step_index and iterate_index cannot be assigned a value."
        )

    if time_step_index is not None:
        if name not in data[pp.TIME_STEP_SOLUTIONS].keys():
            raise KeyError(f"There are no values related the parameter name {name}.")

        if time_step_index not in data[pp.TIME_STEP_SOLUTIONS][name].keys():
            raise KeyError(
                f"There are no values stored for time step index {time_step_index}."
            )
        return data[pp.TIME_STEP_SOLUTIONS][name][time_step_index].copy()

    else:
        if name not in data[pp.ITERATE_SOLUTIONS].keys():
            raise KeyError(f"There are no values related the parameter name {name}.")

        if iterate_index not in data[pp.ITERATE_SOLUTIONS][name].keys():
            raise KeyError(
                f"There are no values stored for iterate index {iterate_index}."
            )
        return data[pp.ITERATE_SOLUTIONS][name][iterate_index].copy()


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
            discr: Mapping between subdomains, or interfaces, where the
                discretization is applied, and the actual Discretization objects.
            key: Keyword that identifies this discretization matrix, e.g.
                for a class with an attribute foo_matrix_key, the key will be foo.
            mat_dict_key: Keyword used to access discretization matrices.
            domains: Domains on which the discretization is defined.

        """
        name = discr.__class__.__name__
        super().__init__(name=name, domains=domains)

        self._discretization_matrix_key = discretization_matrix_key
        self._discr = discr
        
        self._physics_key = physics_key
        self._inner_physics_key = inner_physics_key
        self.domain = domains

    def __repr__(self) -> str:
        if len(self.interfaces) == 0:
            s = f"Operator with key {self._discretization_matrix_key} defined on {len(self.subdomains)} subdomains"
        else:
            s = f"Operator with key {self._discretization_matrix_key} defined on {len(self.interfaces)} edges"
        return s

    def __str__(self) -> str:
        return f"{self._name}({self._physics_key}).{self._discretization_matrix_key}"

    def parse(self, mdg: pp.MixedDimensionalGrid) -> sps.spmatrix:
        """Convert a merged operator into a sparse matrix by concatenating
        discretization matrices.

        Parameters:
            mdg: Mixed-dimensional grid.

        Returns:
            sps.spmatrix: The merged discretization matrices for the associated matrix.

        """

        # Data structure for matrices
        mat = []

        if len(self.domains) == 0:
            # The underlying discretization is constructed on an empty grid list, for
            # instance on a mixed-dimensional grid containing no mortar subdomains. We
            # can return an empty matrix.
            return sps.csc_matrix((0, 0))

        # Loop over all grid-discretization combinations, get hold of the discretization
        # matrix for this grid quantity
        for g in self.domains:
            # Get data dictionary for either grid or interface
            if isinstance(g, pp.MortarGrid):
                data = mdg.interface_data(g)
            else:
                data = mdg.subdomain_data(g)

            mat_dict: dict[str, sps.spmatrix] = data[  # type: ignore
                pp.DISCRETIZATION_MATRICES
            ][self._physics_key]

            # Get the submatrix for the right discretization
            key = self._discretization_matrix_key
            mat_key = getattr(self._discr, key + "_matrix_key")
            if self._inner_physics_key is not None:
                local_mat = mat_dict[mat_key][self._inner_physics_key]
            else:
                local_mat = mat_dict[mat_key]
            mat.append(local_mat)

        if all([isinstance(m, np.ndarray) for m in mat]):
            # TODO: EK is almost sure this never happens, but leave this check for now
            raise NotImplementedError("")

        else:
            # This is a standard discretization; wrap it in a diagonal sparse matrix
            return sps.block_diag(mat, format="csr")
