"""
Utility functions for the AD package.

Functions:
    concatenate - EK, please document.

    wrap_discretization: Convert a discretization to its ad equivalent.

    uniquify_discretization_list: Define a unique list of discretization-keyword
        combinations from a list of AD discretizations.

    discretize_from_list: Perform the actual discretization for a list of AD
        discretizations.

Classes:
    MergedOperator: Representation of specific discretization fields for an AD
        discretization or a set of AD discretizations.

"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

from . import operators
from .forward_mode import Ad_array

Edge = Tuple[pp.Grid, pp.Grid]
GridList = List[pp.Grid]
EdgeList = List[Edge]


def concatenate(variables, axis=0):
    # NOTE: This function is related to an outdated approach to equation assembly
    # based on forward Ad. For now, it is kept for legacy reasons, however,
    # the newer approach based on Operators and the EquationManager is
    # highly recommended.
    vals = [var.val for var in variables]
    jacs = np.array([var.jac for var in variables])

    vals_stacked = np.concatenate(vals, axis=axis)
    jacs_stacked = []
    jacs_stacked = sps.vstack(jacs)

    return Ad_array(vals_stacked, jacs_stacked)


def wrap_discretization(
    obj,
    discr,
    grids: Optional[GridList] = None,
    edges: Optional[EdgeList] = None,
    mat_dict_key: Optional[str] = None,
    mat_dict_grids=None,
):
    """
    Please add documentation, @EK.
    It is assumed that grids or edges is None
    """
    if grids is None:
        assert isinstance(edges, list)
    else:
        assert isinstance(grids, list)
        assert edges is None

    key_set = []
    # Loop over all discretizations, identify all attributes that ends with
    # "_matrix_key". These will be taken as discretizations (they are discretization
    # matrices for specific terms, to be).

    if mat_dict_key is None:
        mat_dict_key = discr.keyword

    if mat_dict_grids is None:
        if edges is None:
            mat_dict_grids = grids
        else:
            mat_dict_grids = edges

    for s in dir(discr):
        if s.endswith("_matrix_key"):
            key = s[:-11]
            key_set.append(key)

    # Make a merged discretization for each of the identified terms.
    # If some keys are not shared by all values in grid_discr, errors will result.
    for key in key_set:
        op = MergedOperator(discr, key, mat_dict_key, mat_dict_grids, grids, edges)
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
    unique_discr_grids: Dict[discr_type, Union[GridList, EdgeList]] = {}

    # Mapping from discretization classes to the discretization.
    # We needed this for some reason..
    cls_obj_map = {}
    # List of all combinations of discretizations and parameter keywords covered.
    cls_key_covered = []
    for discr in all_discr:
        # Get the class of the underlying dicsretization, so MpfaAd will return Mpfa.
        cls = discr.discr.__class__
        # Parameter keyword for this discretization
        param_keyword = discr.keyword

        # This discretization-keyword combination
        key = (cls, param_keyword)

        if key in cls_key_covered:
            # If this has been encountered before, we add grids not earlier associated
            # with this discretization to the existing list.
            # of grids.
            # Map from discretization class to Ad discretization
            d = cls_obj_map[cls]
            for g in discr.grids:
                if g not in unique_discr_grids[d]:
                    unique_discr_grids[d].append(g)
            for e in discr.edges:
                if e not in unique_discr_grids[d]:
                    unique_discr_grids[d].append(e)
        else:
            # Take note we have now encountered this discretization and parameter keyword.
            cls_obj_map[cls] = discr.discr
            cls_key_covered.append(key)

            # Add new discretization with associated list of grids.
            # Need a copy here to avoid assigning additional grids to this
            # discretization (if not copy, this may happen if
            # the key-discr combination is encountered a second time and the
            # code enters the if part of this if-else).
            grid_likes = discr.grids.copy() + discr.edges.copy()
            unique_discr_grids[discr.discr] = grid_likes

    return unique_discr_grids


def discretize_from_list(
    discretizations: Dict,
    gb: pp.GridBucket,
) -> None:
    """For a list of (ideally uniquified) discretizations, perform the actual
    discretization.
    """
    for discr in discretizations:
        # discr is a discretization (on node or interface in the GridBucket sense)

        # Loop over all grids (or GridBucket edges), do discretization.
        for g in discretizations[discr]:
            if isinstance(g, tuple):
                data = gb.edge_props(g)  # type:ignore
                g_primary, g_secondary = g
                d_primary = gb.node_props(g_primary)
                d_secondary = gb.node_props(g_secondary)
                discr.discretize(g_primary, g_secondary, d_primary, d_secondary, data)
            else:
                data = gb.node_props(g)
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
        grids: Optional[GridList] = None,
        edges: Optional[EdgeList] = None,
    ) -> None:
        """Initiate a merged discretization.

        Parameters:
            grid_discr (dict): Mapping between grids, or interfaces, where the
                discretization is applied, and the actual Discretization objects.
            key (str): Keyword that identifies this discretization matrix, e.g.
                for a class with an attribute foo_matrix_key, the key will be foo.
            mat_dict_key (str): Keyword used to access discretization matrices, if this
                is not the same as the keyword of the discretization. The only known
                case where this is necessary is for Mpfa applied to Biot's equations.
            mat_dict_grids: EK, could this be a list of grid-likes?

        """
        self._name = discr.__class__.__name__
        self._set_grids_or_edges(grids, edges)
        self.key = key
        self.discr = discr

        # Special field to access matrix dictionary for Biot
        self.mat_dict_key = mat_dict_key
        self.mat_dict_grids = mat_dict_grids

        self._set_tree(None)

    def __repr__(self) -> str:
        if len(self.edges) == 0:
            s = f"Operator with key {self.key} defined on {len(self.grids)} grids"
        else:
            s = f"Operator with key {self.key} defined on {len(self.edges)} edges"
        return s

    def __str__(self) -> str:
        return f"{self._name}({self.mat_dict_key}).{self.key}"

    def parse(self, gb):
        """Convert a merged operator into a sparse matrix by concatenating
        discretization matrices.

        Pameteres:
            gb (pp.GridBucket): Mixed-dimensional grid. Not used, but it is needed as
                input to be compatible with parse methods for other operators.

        Returns:
            sps.spmatrix: The merged discretization matrices for the associated matrix.

        """

        # Data structure for matrices
        mat = []

        if len(self.mat_dict_grids) == 0:
            # The underlying discretization is constructed on an empty grid list, quite
            # likely on a mixed-dimensional grid containing no mortar grids.
            # We can return an empty matrix.
            return sps.csc_matrix((0, 0))

        # Loop over all grid-discretization combinations, get hold of the discretization
        # matrix for this grid quantity
        for g in self.mat_dict_grids:

            # Get data dictionary for either grid or interface
            if isinstance(g, pp.Grid):
                data = gb.node_props(g)
            else:
                data = gb.edge_props(g)

            mat_dict: Dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
                self.mat_dict_key
            ]

            # Get the submatrix for the right discretization
            key = self.key
            mat_key = getattr(self.discr, key + "_matrix_key")
            mat.append(mat_dict[mat_key])

        if all([isinstance(m, np.ndarray) for m in mat]):
            if all([m.ndim == 1 for m in mat]):
                # This is a vector (may happen e.g. for right hand side terms that are
                # stored as discretization matrices, as may happen for non-linear terms)
                return np.hstack(mat)
            elif all([m.ndim == 2 for m in mat]):
                # Variant of the first case
                if all([m.shape[0] == 1 for m in mat]):
                    return np.hstack(mat).ravel()

                elif all([m.shape[1] == 1 for m in mat]):
                    return np.vstack(mat).ravel()
            else:
                # This should correspond to a 2d array. In prinicple it should be
                # possible to concatenate arrays in the right direction, provided they
                # have coinciding shapes. However, the use case is not clear, so we
                # raise an error, and rethink if we ever get here.
                raise NotImplementedError("")
        else:
            # This is a standard term; wrap it in a diagonal sparse matrix
            return sps.block_diag(mat, format="csr")
