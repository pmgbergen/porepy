"""Module contains various utility functions for working with grids."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

import porepy as pp
from porepy.geometry.half_space import half_space_interior_point
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data

if TYPE_CHECKING:  # Avoid importing heavyish modules at runtime purely for typing.
    from porepy.grids.grid import Grid

logger = logging.getLogger(__name__)


def switch_sign_if_inwards_normal(
    g: Grid, nd: int, faces: NDArray[np.int_]
) -> sps.dia_matrix:
    """Construct a matrix that changes sign of quantities on faces with a normal that
    points into the grid.

    Parameters:
        g: Grid.
        nd: Number of quantities per face; this will for instance be the number of
            components in a face-vector.
        faces: Index for which faces to be considered. Should only contain boundary
            faces.

    Returns:
        sps.dia_matrix: Diagonal matrix which switches the sign of faces if the normal
        vector of the face points into the grid g. Faces not considered will have a 0
        diagonal term. If nd > 1, the first nd rows are associated with the first face,
        then nd elements of the second face etc.

    """

    faces = np.asarray(faces)

    # Find out whether the boundary faces have outwards pointing normal vectors.
    # Negative sign implies that the normal vector points inwards.
    sgn, _ = g.signs_and_cells_of_boundary_faces(faces)

    # Create vector with the sign in the places of faces under consideration,
    # zeros otherwise.
    sgn_mat = np.zeros(g.num_faces)
    sgn_mat[faces] = sgn
    # Duplicate the numbers, the operator is intended for vector quantities.
    sgn_mat = np.tile(sgn_mat, (nd, 1)).ravel(order="F")

    # Create the diagonal matrix.
    return sps.dia_matrix((sgn_mat, 0), shape=(sgn_mat.size, sgn_mat.size))


def star_shape_cell_centers(g: Grid, as_nan: bool = False) -> NDArray[np.float64]:
    """For a given grid compute the star shape center for each cell.

    The algorithm computes the half space intersections of the spaces defined by the
    cell faces and the face normals. This is a wrapper method that operates on a grid.

    Parameters:
        g: The grid.
        as_nan: Decide whether to return nan as the new center for cells which are not
            star-shaped. Otherwise, an exception is raised (default behaviour).

    Returns:
        Array containing the new cell centers.

    """
    # Nothing to do for 1d or 0d grids.
    if g.dim < 2:
        return g.cell_centers

    # Retrieve the faces and nodes.
    faces, _, sgn = sparse_array_to_row_col_data(g.cell_faces)
    nodes, _, _ = sparse_array_to_row_col_data(g.face_nodes)

    # Shift the nodes close to the origin to avoid numerical problems when coordinates
    # are too big.
    xn = g.nodes.copy()
    xn_shift = np.average(xn, axis=1)
    xn -= np.tile(xn_shift, (xn.shape[1], 1)).T

    # Compute the star shape cell centers by constructing the half spaces of each cell
    # given by its faces and related normals.
    cell_centers = np.zeros((3, g.num_cells))
    for c in np.arange(g.num_cells):
        loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
        faces_loc = faces[loc]
        loc_n = g.face_nodes.indptr[faces_loc]
        # Make the normals coherent.
        normal = np.multiply(
            sgn[loc], np.divide(g.face_normals[:, faces_loc], g.face_areas[faces_loc])
        )

        x0, x1 = xn[:, nodes[loc_n]], xn[:, nodes[loc_n + 1]]
        coords = np.concatenate((x0, x1), axis=1)
        # Compute a point in the half space intersection of all cell faces.
        try:
            cell_centers[:, c] = half_space_interior_point(
                normal, (x1 + x0) / 2.0, coords
            )
        except ValueError:
            # The cell is not star-shaped.
            if as_nan:
                cell_centers[:, c] = np.array([np.nan, np.nan, np.nan])
            else:
                raise ValueError(
                    "Cell not star-shaped; impossible to compute the center."
                )

    # Shift back the computed cell centers and return them.
    return cast(
        NDArray[np.float64],
        cell_centers + np.tile(xn_shift, (g.num_cells, 1)).T,
    )


def compute_circumcenters(
    sd: pp.TriangleGrid | pp.TetrahedralGrid,
    threshold: float = 0.95,
    tol: float = 1e-14,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Compute circumcenters for simplicial grids in 2 and 3 dimensions.

    If the circumcenter lies outside of the cell, or too close to the boundary, a
    partial movement from barycenter towards circumcenter is applied.
    The movement is such that it goes at most as far as ``threshold`` times
    distance from barycenter to face between barycenter and circumcenter
    (exit distance).
    I.e., the new center is not the barycenter and not the circumcenter, but a point
    on the line between them, which is sufficiently far in the interior of the domain.

    For simplices with right or almost right angles, the circumcenter lies exactly or
    almost on a face. I.e., the shift from barycenter to circumcenter is capped by
    ``threshold`` times the exit distance.

    For simplices with obtuse angles, the circumcenter lies outside the simplex.
    The shift is capped analogously.

    All cases, where the circumcenter would violate the threshold, are logged as bad
    cells.

    Paremters:
        sd: A simplex grid.
        threshold: ``default=0.95``

            Fraction of distance between barycenter and exit face, denoting the maximal
            admissible shift. Must be in ``(0,1)`` for the computed cell center to be
            strictly in the interior.

        tol: ``default=1e-14``

            Absolute tolerance to detect degenerate cells and exactness of computations.

    Raises:
        ValueError: If the threshold is not strictly in ``(0,1)``.
        ValueError: If the grid is not a simplex of dimension 2 or 3.
        ValueError: If any cell is degenerate.

    Returns:
        A 3-tuple containing

        1. the new cell centers of shape ``sd.cell_centers.shape``.
        2. Shift values of shape ``(sd.num_cells,)``.
        3. a boolean array of shape ``(sd.num_cells,)`` indicating where numerically
           relevant changes in cell centers occurred.

        The shift values contain the scale of the vector going from barycenter
        to new cell center per cell. A value of 1 indicates the circumcenter is strictly
        in the interior of the triangle and within the threshold. Values below that
        indicate that the real circumcenter would violate the threshold distance, i.e.
        the returned, new cell center is not the actual circumcenter.

    """
    if not (0 < threshold < 1):
        raise ValueError(f"Threshold must be in (0, 1), got {threshold}.")

    dim = sd.dim
    numc = sd.num_cells
    dim1p = dim + 1

    if (type(sd), dim) not in [(pp.TriangleGrid, 2), (pp.TetrahedralGrid, 3)]:
        raise ValueError(f"Unsupported grid type {type(sd)} of dimension {dim}.")

    # Extract nodes.
    c2n = sd.cell_nodes().tocsc()
    n_idx = c2n.indices.reshape((dim1p, numc), order="F")
    x, y, z = sd.nodes[0], sd.nodes[1], sd.nodes[2]

    # Let A be the matrix of edges (columnwise) pointing away from node n_0. It has
    # shape (3, dim) per cell. The barycentric coordinates with respect to n_0 are given
    # by a vector b of shape (dim,), and it holds that the circumcenter c = n_0 + Ab.
    # For the circumcenter it holds that ||c - n_i||^2 = ||c - n_0||^2, i = 1..dim, i.e.
    # it is equidistant to all nodes. Plugging in the barycentric representation we get
    # ||Ab + n_0 - n_i||^2 = ||Ab||^2, i = 1 ..dim, a dim x dim system of nonlinear
    # equations. Let e_i = n_i - n_0 be the i-th edge. Then we can expand the above
    # system to
    # ||Ab - e_i||^2 = ||Ab||^2
    # (Ab)T (Ab) - 2 (Ab)T e_i + ||e_i||^2 = (Ab)T (Ab)
    # (Ab)T e_i = ||e_i||^2 / 2
    # Stacking the dim equations for edges e_i, i=1..dim gives
    # (A)T Ab = v / 2,
    # with v being the array of ||e_i||^2.
    # Hence the barycentric coordinates are given by b = 1 / 2 ((A)T A)^-1 v.
    # This formula holds for all simplices, 2D, 3D and for 2D embedded in 3D.
    # The matrix M = (A)T A is positive definite and invertible, if the simplex is
    # not degenerate, or vice versa if its determinant is zero, the simplex is
    # degenerate.

    # For batched computations, we stack the matrices A and the rhs v/2 along a new
    # axis, and let numpy solve it efficiently. The vectorized computations are
    # triggered for the first axis, hence we move the per-cell axis to 0.

    # nodes (dim + 1, 3, nc)
    n = np.array([(x[n_idx[i]], y[n_idx[i]], z[n_idx[i]]) for i in range(dim1p)])
    # row-wise edges (dim, 3, nc) -  matrix (A)T
    At: NDArray[np.float64] = n[1:] - n[0][None, :, :]
    # Edge norms squared (dim, nc) - rhs v / 2
    v: NDArray[np.float64] = 0.5 * np.sum(At**2, axis=1)
    # Matrix product of edge matrices (dim, dim, nc) - Gram matrix M = (A)T A
    # NOTE with modern numpy and BLAS, matmul is significantly faster then einsum for
    # more than 100k cells.
    # M: NDArray[np.float64] = np.einsum("ikn,jkn->ijn", At, At)
    At_batch = np.moveaxis(At, -1, 0)
    M: NDArray[np.float64] = np.moveaxis(
        np.matmul(At_batch, At_batch.transpose(0, 2, 1)),
        0,
        -1,
    )
    assert M.shape == (dim, dim, numc)
    assert v.shape == (dim, numc)

    # Solving system to get barycentric coordinates. Moving batch axis (cells) to the
    # front to trigger numpy's vectorization.
    M_batch = np.moveaxis(M, -1, 0)
    v_batch = v.T

    det: NDArray[np.float64] = np.linalg.det(M_batch)
    assert det.shape == (numc,)
    # Degeneracy check. M should be positive definite if simplices are not degenerate.
    if not np.all(det > tol):
        raise ValueError("Degenerate simplex with near-zero volume encountered.")

    # Solve system and obtain barycentric coordinates.
    # Move batch axis (cells) again to the back
    b_r = np.linalg.solve(M_batch, v_batch[..., None])[..., 0].T
    assert b_r.shape == (dim, numc)

    # The circumcenter is given by c = n_0 + Ab.
    # NOTE Here we use einsum, because it's a batched matrix-vector product. Code
    # more transparent and faster then moving axes around and doing the product using
    # matmul.
    ccs: NDArray[np.float64] = n[0] + np.einsum("dkn,dn->kn", At, b_r)
    assert ccs.shape == (3, numc)

    # SANITY CHECK: Circumcenter equidistant to nodes.
    dist = np.array([np.linalg.norm(ccs - n[i, :, :], axis=0) for i in range(dim1p)])
    max_dist = np.maximum.reduce(dist, axis=0)
    min_dist = np.minimum.reduce(dist, axis=0)
    assert np.max(np.abs(max_dist - min_dist)) <= tol, (
        "Circumcenters not equidistant from all nodes."
    )

    # Starting point for movement: barycenters (3, nc)
    bcs = np.mean(n, axis=0)
    # Full shift vector from barycenters to circumcenters.
    shift_vec = ccs - bcs

    # Catching cases where the circumcenter is close to boundary or outside of triangle.
    # The exit face is the face passed when going from barycenter in direction
    # circumcenter. It is the one face where the dot product of outwards normal and
    # shift vector is the largest. The exit point is the intersection of the respective
    # face with line spanned by barycenter and circumcenter.
    # The distance between barycenter and exitpoint, scaled with threshold, is the
    # maximal admissible movement along the bary-to-circum ray.

    # Cell-face connectivity.
    c2f = sd.cell_faces.tocsc()
    face_idx = c2f.indices.reshape((dim1p, numc), order="F")  # (dim + 1, nc)
    face_sgn = c2f.data.reshape((dim1p, numc), order="F")  # (dim + 1, nc)

    # Normalized outwards face normals (3, dim + 1, nc).
    fn_out = sd.face_normals[:, face_idx] * face_sgn[None, :, :]
    fn_out = fn_out / np.linalg.norm(fn_out, axis=0, keepdims=True)

    F = sd.face_centers[:, face_idx]  # Face centers (3, dim + 1, nc).

    # Dot products of all face outwards normals with the shift vector (dim + 1, nc).
    dots: NDArray[np.float64] = np.einsum("ijk,ik->jk", fn_out, shift_vec)
    assert dots.shape == (dim + 1, numc), (
        "Inconsistent dot products of shift vector with normals."
    )
    face_id = np.argmax(dots, axis=0)  # (nc,)

    idx = np.arange(numc)
    fn = fn_out[:, face_id, idx]  # (3, nc)
    pof = F[:, face_id, idx]  # point on face (3, nc)

    # Plane (face) and line intersection.
    denom = np.maximum(np.einsum("jk,jk->k", shift_vec, fn), tol)  # avoid /0 zero.
    num = np.einsum("jk,jk->k", pof - bcs, fn)
    t = num / denom

    epts = bcs + t * shift_vec  # exit points (3, nc)

    # Maximally allowed shift: distance barycenter to exitpoint times threshold.
    max_shift = np.linalg.norm(epts - bcs, axis=0) * threshold
    # Actual shift: distance barycenter to calculated circumcenter.
    act_shift = np.maximum(np.linalg.norm(shift_vec, axis=0), tol)
    # Shift fully, where potential shift does not violate maximal shift, or apply
    # maximal shift (normalized with length of shift vector).
    shift = np.where(act_shift <= max_shift, np.ones(numc), max_shift / act_shift)

    # Correct shift values in case barycenters and circumcenters coincide.
    shift[act_shift <= tol] = 0.0
    # Compute new cell center by applying shift to barycenter.
    nccs: NDArray[np.float64] = bcs + shift * shift_vec

    # Changes as per specificed tolerance.
    changed: NDArray[np.bool_] = np.linalg.norm(nccs - sd.cell_centers, axis=0) > tol

    # Log total change.
    logger.info(
        "Replaced %d out of %d cell centers on grid %d.",
        int(changed.sum()),
        numc,
        sd.id,
    )
    # Log changes where circumcenter not within threshold.
    logger.info(
        "Circumcenter not in threshold in %d out of %d cells on grid %d (bad cells).",
        int(np.sum(shift < 1)),
        numc,
        sd.id,
    )
    assert nccs.shape == sd.cell_centers.shape, (
        "Inconsistent shape for new cell centers."
    )
    assert shift.shape == (numc,), "Inconsistent shape for shift values."
    assert changed.shape == (numc,), "Inconsistent shape for change indicators."

    return nccs, shift, changed
