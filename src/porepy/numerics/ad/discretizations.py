"""

"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

from .operators import Operator

__all__ = [
    "Discretization",
    "BiotAd",
    "MpsaAd",
    "GradPAd",
    "DivUAd",
    "BiotStabilizationAd",
    "ColoumbContactAd",
    "MpfaAd",
    "MassMatrixAd",
    "UpwindAd",
    "RobinCouplingAd",
]


class Discretization:
    """Wrapper to make a PorePy discretization compatible with the Ad framework.

    For any discretization class compatible with PorePy, the wrapper associates
    a discretization with all attributes of the class' attributes that ends with
    '_matrix_key'.

    Before wrapping a standard discretization by this class, you should check if
    a wrapper has already been implemented, e.g. pp.ad.MpfaAd etc.

    Example:
        # Generate grid
        >>> g = pp.CartGrid([2, 2])
        # Associate an Ad representation of an Mpfa method, aimed this grid
        >>> discr = Discretization({g: pp.Mpfa('flow')})
        # The flux discretization of Mpfa can now be accesed by
        >>> discr.flux
        # While the discretization of boundary conditions is available by
        >>> discr.bound_flux.

    The representation of different discretization objects can be combined with other
    Ad objects into an operator tree, using lazy evaluation.

    It is assumed that the actual action of discretization (creation of the
    discretization matrices) is performed before the operator tree is parsed.

    """

    def __init__(
        self,
        grids: Union[List[pp.Grid], List[Tuple[pp.Grid, pp.Grid]]],
        discretization: "pp.numerics.discretization.Discretization",
        mat_dict_key: Optional[str] = None,
    ):
        """Construct a wrapper around a Discretization object for a set of grids.

        Different grids may be associated with different discetization classes, but they
        should have the same keywords to access discretization matrices (Tpfa and Mpfa
        are compatible in this sense).

        Parameters:
            grid_discr (dict): Mapping between grids, or interfaces, where the
                discretization is applied, and the actual Discretization objects.
            name (str): Name of the wrapper.
            mat_dict_key (str): Keyword used to access discretization matrices, if this
                is not the same as the keyword of the discretization. The only known
                case where this is necessary is for Mpfa applied to Biot's equations.

        """
        self.grids = grids
        self._discretization = discretization
        self.mat_dict_key = mat_dict_key

        # Get the name of this discretization.
        self._name = discretization.__class__.__name__

        _wrap_discretization(self, self._discretization, self.grids, self.mat_dict_key)

    def __repr__(self) -> str:
        s = f"Ad discretization of type {self._name}. Defined on {len(self.grids)} grids"
        return s


### Mechanics related discretizations


class BiotAd:
    """Ad wrapper around the Biot discretization class.

    For description of the method, we refer to the standard Biot class.

    """

    def __init__(self, keyword: str, grids: Union[pp.Grid, List[pp.Grid]]) -> None:
        if isinstance(grids, list):
            self._grids = grids
        else:
            self._grids = [grids]
        self._discretization = pp.Biot(keyword)
        self._name = "BiotMpsa"

        self.keyword = keyword

        # Declear attributes, these will be initialized by the below call to the
        # discretization wrapper.

        self.stress: _MergedOperator
        self.bound_stress: _MergedOperator
        self.bound_displacement_cell: _MergedOperator
        self.bound_displacement_face: _MergedOperator

        self.div_u: _MergedOperator
        self.bound_div_u: _MergedOperator
        self.grad_p: _MergedOperator
        self.stabilization: _MergedOperator
        self.bound_pressure: _MergedOperator

        _wrap_discretization(
            obj=self, discr=self._discretization, grids=grids, mat_dict_key=self.keyword
        )

    def __repr__(self) -> str:
        s = f"Ad discretization of type {self._name}. Defined on {len(self._grids)} grids"
        return s


class MpsaAd:
    def __init__(self, keyword, grids):
        if isinstance(grids, list):
            self._grids = grids
        else:
            self._grids = [grids]
        self._discretization = pp.Mpsa(keyword)
        self._name = "Mpsa"

        self.keyword = keyword

        # Declear attributes, these will be initialized by the below call to the
        # discretization wrapper.

        self.stress: _MergedOperator
        self.bound_stress: _MergedOperator
        self.bound_displacement_cell: _MergedOperator
        self.bound_displacement_face: _MergedOperator

        _wrap_discretization(self, self._discretization, grids)

    def __repr__(self) -> str:
        s = f"Ad discretization of type {self._name}. Defined on {len(self._grids)} grids"
        return s


class GradPAd:
    def __init__(self, keyword, grids):
        if isinstance(grids, list):
            self._grids = grids
        else:
            self._grids = [grids]
        self._discretization = pp.GradP(keyword)
        self._name = "GradP from Biot"
        self.keyword = keyword

        self.grad_p: _MergedOperator

        _wrap_discretization(self, self._discretization, grids)

    def __repr__(self) -> str:
        s = f"Ad discretization of type {self._name}. Defined on {len(self._grids)} grids"
        return s


class DivUAd:
    def __init__(self, keyword, grids, mat_dict_keyword):
        if isinstance(grids, list):
            self._grids = grids
        else:
            self._grids = [grids]
        self._discretization = pp.DivU(keyword, mat_dict_keyword)

        self._name = "DivU from Biot"
        self.keyword = mat_dict_keyword

        self.div_u: _MergedOperator
        self.bound_div_u: _MergedOperator

        _wrap_discretization(self, self._discretization, grids, mat_dict_keyword)

    def __repr__(self) -> str:
        s = f"Ad discretization of type {self._name}. Defined on {len(self._grids)} grids"
        return s


class BiotStabilizationAd:
    def __init__(self, keyword, grids):
        if isinstance(grids, list):
            self._grids = grids
        else:
            self._grids = [grids]
        self._discretization = pp.BiotStabilization(keyword)
        self._name = "Biot stabilization term"
        self.keyword = keyword

        self.stabilization: _MergedOperator

        _wrap_discretization(self, self._discretization, grids)

    def __repr__(self) -> str:
        s = f"Ad discretization of type {self._name}. Defined on {len(self._grids)} grids"
        return s


class ColoumbContactAd:
    def __init__(self, keyword, grids):

        if isinstance(grids, list):
            self._grids = grids
        else:
            self._grids = [grids]

        dim = np.unique([g[0].dim for g in grids])

        low_dim_grids = [g[1] for g in grids]
        if not dim.size == 1:
            raise ValueError("Expected unique dimension of grids with contact problems")

        self._discretization = pp.ColoumbContact(
            keyword, ambient_dimension=dim[0], discr_h=pp.Mpsa(keyword)
        )
        self._name = "Biot stabilization term"
        self.keyword = keyword

        self.traction: _MergedOperator
        self.displacement: _MergedOperator
        self.rhs: _MergedOperator

        _wrap_discretization(
            self, self._discretization, grids, mat_dict_grids=low_dim_grids
        )

    def __repr__(self) -> str:
        s = f"Ad discretization of type {self._name}. Defined on {len(self._grids)} grids"
        return s


## Flow related


class MpfaAd:
    def __init__(self, keyword, grids):

        if isinstance(grids, list):
            self._grids = grids
        else:
            self._grids = [grids]
        self._discretization = pp.Mpfa(keyword)
        self._name = "Mpfa"
        self.keyword = keyword

        self.flux: _MergedOperator
        self.bound_flux: _MergedOperator
        self.bound_pressure_cell: _MergedOperator
        self.bound_pressure_face: _MergedOperator
        self.vector_source: _MergedOperator
        self.bound_pressure_vector_source: _MergedOperator

        _wrap_discretization(self, self._discretization, grids)

    def __repr__(self) -> str:
        s = f"Ad discretization of type {self._name}. Defined on {len(self._grids)} grids"
        return s


class MassMatrixAd:
    def __init__(self, keyword, grids):
        if isinstance(grids, list):
            self._grids = grids
        else:
            self._grids = [grids]
        self._discretization = pp.MassMatrix(keyword)
        self._name = "Mass matrix"
        self.keyword = keyword

        self.mass: _MergedOperator
        _wrap_discretization(self, self._discretization, grids)

    def __repr__(self) -> str:
        s = f"Ad discretization of type {self._name}. Defined on {len(self._grids)} grids"
        return s


class UpwindAd:
    def __init__(self, keyword, grids):
        if isinstance(grids, list):
            self._grids = grids
        else:
            self._grids = [grids]
        self._discretization = pp.Upwind(keyword)
        self._name = "Upwind"
        self.keyword = keyword

        self.upwind: _MergedOperator
        self.rhs: _MergedOperator
        self.outflow_neumann: _MergedOperator
        _wrap_discretization(self, self._discretization, grids)

    def __repr__(self) -> str:
        s = f"Ad discretization of type {self._name}. Defined on {len(self._grids)} grids"
        return s


class RobinCouplingAd:
    def __init__(self, keyword, edges):
        if isinstance(edges, list):
            self._edges = edges
        else:
            self._edges = [edges]
        self._discretization = pp.RobinCoupling(keyword, primary_keyword=keyword)
        self._name = "Robin interface coupling"
        self.keyword = keyword

        self.mortar_scaling: _MergedOperator
        self.mortar_discr: _MergedOperator
        _wrap_discretization(self, self._discretization, edges)

    def __repr__(self) -> str:
        s = f"Ad discretization of type {self._name}. Defined on {len(self._edges)} grids"
        return s


class _MergedOperator(Operator):
    """Representation of specific discretization fields for an Ad discretization.

    This is the bridge between the representation of discretization classes, implemented
    in Discretization, and the matrices resulting from a discretization.

    Objects of this class should not be access directly, but rather through the
    Discretization class.

    """

    def __init__(
        self,
        grids: Union[List[pp.Grid], List[Tuple[pp.Grid, pp.Grid]]],
        discr: "pp.numerics.discretization.Discretization",
        key: str,
        mat_dict_key: str,
        mat_dict_grids: Optional[Union[pp.Grid, Tuple[pp.Grid, pp.Grid]]],
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

        """
        self.grids = grids
        self.key = key
        self.discr = discr

        # Special field to access matrix dictionary for Biot
        self.mat_dict_key = mat_dict_key
        self.mat_dict_grids = mat_dict_grids

        self._set_tree(None)

    def __repr__(self) -> str:
        return f"Operator with key {self.key} defined on {len(self.grids)} grids"

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
            return sps.block_diag(mat)


def _wrap_discretization(
    obj, discr, grids, mat_dict_key: Optional[str] = None, mat_dict_grids=None
):
    key_set = []
    # Loop over all discretizations, identify all attributes that ends with
    # "_matrix_key". These will be taken as discretizations (they are discretization
    # matrices for specific terms, to be).
    if not isinstance(grids, list):
        grids = [grids]

    if mat_dict_key is None:
        mat_dict_key = discr.keyword

    if mat_dict_grids is None:
        mat_dict_grids = grids

    for s in dir(discr):
        if s.endswith("_matrix_key"):
            key = s[:-11]
            key_set.append(key)

    # Make a merged discretization for each of the identified terms.
    # If some keys are not shared by all values in grid_discr, errors will result.
    for key in key_set:
        op = _MergedOperator(grids, discr, key, mat_dict_key, mat_dict_grids)
        setattr(op, "keyword", mat_dict_key)
        setattr(obj, key, op)
