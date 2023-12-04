import numpy as np
import scipy as sp
import porepy as pp

import pdb


heaviside_operators = pp.ad.Function(pp.ad.heaviside, "heviside_for_operators")


def expansion_matrix(sd: pp.Grid) -> sp.sparse.spmatrix:
    """
    from internal faces set to all faces set
    TODO: change the name
    """
    data = np.ones(sd.get_internal_faces().shape[0])
    rows = sd.get_internal_faces()
    cols = np.arange(sd.get_internal_faces().shape[0])

    expansion = sp.sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(sd.num_faces, sd.get_internal_faces().shape[0]),
    )
    return expansion


def restriction_matrices_left_right(sd: pp.Grid) -> sp.sparse.spmatrix:
    """
    remark: with ad you have to use matrices instead of working with indices
    TODO: PAY ATTENTION: there are two logical operation in the same function (improve it):
            get internal set of faces and compute left and right restriction of internal set
    TODO: this function was essentially copied from email. Improve it if possible.
    TODO: use one matrix like: restriction = restriction_left + restiction_right

    var_left = left_restriction @ var
    var_left.shape = (len(sd.get_internal_faces()),)
    var.shape = (sd.num_cells)
    """

    internal_faces = sd.get_internal_faces()

    cell_left_internal_id = sd.cell_face_as_dense()[
        0, internal_faces
    ]  # left cells id of the internal faces subset
    cell_right_internal_id = sd.cell_face_as_dense()[
        1, internal_faces
    ]  # right cells id of the internal faces subset

    data_l = np.ones(cell_left_internal_id.size)
    rows_l = np.arange(cell_left_internal_id.size)
    cols_l = cell_left_internal_id

    left_restriction = sp.sparse.coo_matrix(
        (data_l, (rows_l, cols_l)), shape=(cell_left_internal_id.size, sd.num_cells)
    )

    data_r = np.ones(cell_right_internal_id.size)
    rows_r = np.arange(cell_right_internal_id.size)
    cols_r = cell_right_internal_id

    right_restriction = sp.sparse.coo_matrix(
        (data_r, (rows_r, cols_r)), shape=(cell_left_internal_id.size, sd.num_cells)
    )

    return left_restriction, right_restriction


def density_internal_faces(
    saturation: pp.ad.AdArray,
    density: pp.ad.AdArray,
    left_restriction: sp.sparse.spmatrix,
    right_restriction: sp.sparse.spmatrix,
) -> pp.ad.AdArray:
    """ """
    s_rho = saturation * density

    density_internal_faces = (left_restriction @ s_rho + right_restriction @ s_rho) / (
        left_restriction @ saturation + right_restriction @ saturation + 1e-10
    )  # added epsilon to avoid division by zero
    return density_internal_faces


def g_internal_faces(
    z: np.ndarray,
    density_faces: pp.ad.AdArray,
    gravity_value: float,
    left_restriction: sp.sparse.spmatrix,
    right_restriction: sp.sparse.spmatrix,
) -> pp.ad.AdArray:
    """ """

    g_faces = (
        density_faces * gravity_value * (left_restriction @ z - right_restriction @ z)
    )
    return g_faces


def compute_transmissibility_tpfa(sd: pp.Grid, data: dict, keyword="flow") -> None:
    """ """
    discr = pp.Tpfa(keyword)
    discr.discretize(sd, data)


def get_transmissibility_tpfa(
    sd: pp.Grid, data: dict, keyword="flow"
) -> (np.ndarray, np.ndarray):
    """TODO: use the matrix of transmissibilities, not the vector"""
    matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][keyword]
    div_transmissibility = matrix_dictionary["flux"]  # TODO: "flux"
    transmissibility = matrix_dictionary["transmissibility"]
    transmissibility_internal = transmissibility[sd.get_internal_faces()]

    return transmissibility, transmissibility_internal


def var_upwinded_faces(
    var: pp.ad.AdArray,
    upwind_directions: pp.ad.AdArray,
    left_restriction: sp.sparse.spmatrix,
    right_restriction: sp.sparse.spmatrix,
) -> pp.ad.AdArray:
    """
    - works for both ad and non ad
    - var defined at cell centers
    - remark: no derivative wrt upwind direction => we use only the val if ad or real part if complex step
    - you can use pp upwind
    """

    var_left = left_restriction @ var
    var_right = right_restriction @ var

    if isinstance(upwind_directions, pp.ad.AdArray):
        upwind_directions = upwind_directions.val

    upwind_left = np.maximum(
        0, np.heaviside(np.real(upwind_directions), 1)
    )  # attention (if complex step is used): I'm using only the real part

    upwind_right = (
        np.ones(upwind_directions.shape) - upwind_left
    )  # what's not left is right (here!)

    upwind_left_matrix = sp.sparse.diags(upwind_left)
    upwind_right_matrix = sp.sparse.diags(upwind_right)

    var_upwinded = upwind_left_matrix @ var_left + upwind_right_matrix @ var_right

    return var_upwinded
