"""
Utility functions for the AD package.

Functions:
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

from typing import Optional, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

from . import operators
from .forward_mode import Ad_array


def concatenate_ad_arrays(ad_arrays: list[Ad_array], axis=0):
    """Concatenates a sequence of AD arrays into a single AD Array along a specified axis."""
    vals = [var.val for var in ad_arrays]
    jacs = np.array([var.jac for var in ad_arrays])

    vals_stacked = np.concatenate(vals, axis=axis)
    jacs_stacked = sps.vstack(jacs)

    return Ad_array(vals_stacked, jacs_stacked)


def wrap_discretization(
    obj,
    discr,
    subdomains: Optional[list[pp.Grid]] = None,
    interfaces: Optional[list[pp.MortarGrid]] = None,
    mat_dict_key: Optional[str] = None,
    mat_dict_grids=None,
):
    """Convert a discretization to its AD equivalent."""
    if subdomains is None:
        assert isinstance(interfaces, list)
    else:
        assert isinstance(subdomains, list)
        assert interfaces is None

    key_set = []
    # Loop over all discretizations, identify all attributes that ends with
    # "_matrix_key". These will be taken as discretizations (they are discretization
    # matrices for specific terms, to be).

    if mat_dict_key is None:
        mat_dict_key = discr.keyword

    if mat_dict_grids is None:
        if interfaces is None:
            mat_dict_grids = subdomains
        else:
            mat_dict_grids = interfaces

    for s in dir(discr):
        if s.endswith("_matrix_key"):
            key = s[:-11]
            key_set.append(key)

    # Make a merged discretization for each of the identified terms.
    # If some keys are not shared by all values in grid_discr, errors will result.
    for key in key_set:
        op = MergedOperator(
            discr, key, mat_dict_key, mat_dict_grids, subdomains, interfaces
        )
        setattr(op, "keyword", mat_dict_key)
        setattr(obj, key, op)


def uniquify_discretization_list(all_discr):
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
    discr_type = Union["pp.Discretization", "pp.AbstractInterfaceLaw"]
    unique_discr_grids: dict[discr_type, list[pp.Grid] | list[pp.MortarGrid]] = dict()

    # Mapping from discretization classes to the discretization.
    # We needed this for some reason.
    cls_obj_map = {}
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


class MergedOperator(operators.Operator):
    """Representation of specific discretization fields for an Ad discretization.

    This is the bridge between the representation of discretization classes, implemented
    in Discretization, and the matrices resulting from a discretization.

    Objects of this class should not be access directly, but rather through the
    Discretization class.

    """

    def __init__(
        self,
        discr: "pp.ad.Discretization",
        key: str,
        mat_dict_key: str,
        mat_dict_grids,
        subdomains: Optional[list[pp.Grid]] = None,
        interfaces: Optional[list[pp.MortarGrid]] = None,
    ) -> None:
        """Initiate a merged discretization.

        Parameters:
            discr (dict): Mapping between subdomains, or interfaces, where the
                discretization is applied, and the actual Discretization objects.
            key (str): Keyword that identifies this discretization matrix, e.g.
                for a class with an attribute foo_matrix_key, the key will be foo.
            mat_dict_key (str): Keyword used to access discretization matrices, if this
                is not the same as the keyword of the discretization. The only known
                case where this is necessary is for Mpfa applied to Biot's equations.
            mat_dict_grids: EK, could this be a list of grid-likes?

        """
        name = discr.__class__.__name__
        super().__init__(name=name)

        self.key = key
        self.discr = discr
        if subdomains is None:
            assert isinstance(interfaces, list)
            self.subdomains = []
            self.interfaces = list(interfaces)
        else:
            assert isinstance(subdomains, list)
            assert interfaces is None
            self.subdomains = list(subdomains)
            self.interfaces = []

        # Special field to access matrix dictionary for Biot
        self.mat_dict_key = mat_dict_key
        self.mat_dict_grids = mat_dict_grids

    def __repr__(self) -> str:
        if len(self.interfaces) == 0:
            s = f"Operator with key {self.key} defined on {len(self.subdomains)} subdomains"
        else:
            s = f"Operator with key {self.key} defined on {len(self.interfaces)} edges"
        return s

    def __str__(self) -> str:
        return f"{self._name}({self.mat_dict_key}).{self.key}"

    def parse(self, mdg):
        """Convert a merged operator into a sparse matrix by concatenating
        discretization matrices.

        Parameters:
            mdg (pp.MixedDimensionalGrid): Mixed-dimensional grid.

        Returns:
            sps.spmatrix: The merged discretization matrices for the associated matrix.

        """

        # Data structure for matrices
        mat = []

        if len(self.mat_dict_grids) == 0:
            # The underlying discretization is constructed on an empty grid list, quite
            # likely on a mixed-dimensional grid containing no mortar subdomains.
            # We can return an empty matrix.
            return sps.csc_matrix((0, 0))

        # Loop over all grid-discretization combinations, get hold of the discretization
        # matrix for this grid quantity
        for g in self.mat_dict_grids:

            # Get data dictionary for either grid or interface
            if isinstance(g, pp.MortarGrid):
                data = mdg.interface_data(g)
            else:
                data = mdg.subdomain_data(g)

            mat_dict: dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
                self.mat_dict_key
            ]

            # Get the submatrix for the right discretization
            key = self.key
            mat_key = getattr(self.discr, key + "_matrix_key")
            mat.append(mat_dict[mat_key])

        if all([isinstance(m, np.ndarray) for m in mat]):
            if all([m.ndim == 1 for m in mat]):
                # This is a vector (may happen e.g. for right-hand side terms that are
                # stored as discretization matrices, as may happen for non-linear terms)
                return np.hstack(mat)
            elif all([m.ndim == 2 for m in mat]):
                # Variant of the first case
                if all([m.shape[0] == 1 for m in mat]):
                    return np.hstack(mat).ravel()

                elif all([m.shape[1] == 1 for m in mat]):
                    return np.vstack(mat).ravel()
            else:
                # This should correspond to a 2d array. In principle, it should be
                # possible to concatenate arrays in the right direction, provided they
                # have coinciding shapes. However, the use case is not clear, so we
                # raise an error, and rethink if we ever get here.
                raise NotImplementedError("")
        else:
            # This is a standard term; wrap it in a diagonal sparse matrix
            return sps.block_diag(mat, format="csr")
