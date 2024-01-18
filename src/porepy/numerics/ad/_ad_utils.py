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
    mat_dict_key: Optional[str] = None,
):
    """Convert a discretization to its AD equivalent.
    
    For a (non-ad) discretization object ``D`` of type ``discr``, this function will
    identify all attributes of the form ``"foo_matrix_key"`` and create a corresponding
    attribute "foo" in the AD discretization ``obj``. Thus, after the call to this
    method, ``obj.foo`` will represent the discretization matrix for the term ``foo``.
    
    For example: If ``D`` is an instance of ``Mpfa`` (which has an attribute
    ``flux_matrix_key``), then ``obj.flux`` will be an instance of ``MpfaAd``, and
    this function equips ``obj`` with the attribute ``obj.flux``.

    Parameters:
        obj: An AD discretization object.
        discr: A non-AD discretization object.
        subdomains: List of grids on which the discretization is defined.
        interfaces: List of interfaces on which the discretization is defined.

        Either subdomains or interfaces must be provided, but not both.

        mat_dict_key: Keyword for matrix storage in the data dictionary. If not
            specified, the keyword of the discretization (the one used to identify
            where parameters are stored) will be used.

    """
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

    if mat_dict_key is None:
        mat_dict_key = discr.keyword

    # Loop over all discretizations, identify all attributes that ends with
    # "_matrix_key". These will be taken as discretizations (they are discretization
    # matrices for specific terms, to be).
    key_set = []
    for s in dir(discr):
        if s.endswith("_matrix_key"):
            key = s[:-11]
            key_set.append(key)

    # Make a merged discretization for each of the identified terms. 
    for key in key_set:
        # Create the merged operator that represents this discretization matrix
        op = MergedOperator(
            discr=discr,
            key=key,
            mat_dict_key=mat_dict_key,
            domains=domains,
        )
        # Set the keyword for this discretization - this is where the parse method will
        # go into a data dictionary and fetch the discretization matrix
        setattr(op, "keyword", mat_dict_key)
        # Set the merged operator as an attribute of the AD discretization
        setattr(obj, key, op)


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
        cls = discr.discr.__class__
        # Parameter keyword for this discretization
        param_keyword = discr.keyword

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
            cls_obj_map[cls] = discr.discr
            cls_key_covered.append(key)

            # Add new discretization with associated list of subdomains.
            # Need a copy here to avoid assigning additional subdomains to this
            # discretization (if not copy, this may happen if
            # the key-discr combination is encountered a second time and the
            # code enters the if part of this if-else).
            grid_likes = discr.subdomains.copy() + discr.interfaces.copy()
            unique_discr_grids[discr.discr] = grid_likes

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
        key: str,
        mat_dict_key: str,
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

        self.key = key
        self.discr = discr

        # Special field to access matrix dictionary for Biot
        self.mat_dict_key = mat_dict_key
        self.domain = domains

        self.keyword: Optional[str] = None

    def __repr__(self) -> str:
        if len(self.interfaces) == 0:
            s = f"Operator with key {self.key} defined on {len(self.subdomains)} subdomains"
        else:
            s = f"Operator with key {self.key} defined on {len(self.interfaces)} edges"
        return s

    def __str__(self) -> str:
        return f"{self._name}({self.mat_dict_key}).{self.key}"

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
            ][self.mat_dict_key]

            # Get the submatrix for the right discretization
            key = self.key
            mat_key = getattr(self.discr, key + "_matrix_key")
            mat.append(mat_dict[mat_key])

        if all([isinstance(m, np.ndarray) for m in mat]):
            # TODO: EK is almost sure this never happens, but leave this check for now
            raise NotImplementedError("")

        else:
            # This is a standard discretization; wrap it in a diagonal sparse matrix
            return sps.block_diag(mat, format="csr")
