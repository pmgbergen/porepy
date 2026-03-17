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
    """Compute circumcenters for simplex grids in 2 and 3 dimensions.

    Paremters:
        sd: A simplex grid.
        threshold: ``default=0.95``

            Enforces the location of the new cell centers to be at at most 95% of the
            distance between barycenters and exit faces. The exit face is determined
            by the direction of the vector barycenter-to-circumcenter.

            Example:

                2D - for right or obtuse triangles the circumcenter lies on a face or
                outside the triangle. The threshold ensures that the new center
                approximates the circumcenter and that the distance to the exit face
                is not zero or too small to cause troubles for discretizations.

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
        to circumcenter per cell. A value of 1 indicates the circumcenter is strictly
        in the interior of the triangle and within the threshold. Values below that
        indicate a fraction of the movement along the vector such that the new center is
        still strictly in the interior and within the threshold.

    """
    if not (0 < threshold < 1):
        raise ValueError(f"Threshold must be in (0, 1), got {threshold}.")

    dim = sd.dim

    if (type(sd), dim) not in [(pp.TriangleGrid, 2), (pp.TetrahedralGrid, 3)]:
        raise ValueError(f"Unsupported grid type {type(sd)} of dimension {dim}.")

    nc = sd.num_cells
    dim1p = dim + 1

    # Extract nodes.
    c2n = sd.cell_nodes().tocsc()
    ni = c2n.indices.reshape((dim1p, sd.num_cells), order="F")
    x, y, z = sd.nodes[0], sd.nodes[1], sd.nodes[2]

    # nodes (dim + 1, 3, nc)
    nds = np.array([(x[ni[i]], y[ni[i]], z[ni[i]]) for i in range(dim1p)])
    # edges (dim, 3, nc)
    eds: NDArray[np.float64] = nds[1:] - nds[0][None, :, :]

    # Starting point for movement: barycenters (3, nc)
    bcs = np.mean(nds, axis=0)

    # Solving system for coordinates of circumcenter c_i.
    # The general system reads something like
    # [node_i - node_0].T * c = (||node_i||^2 - ||node_0||^2) / 2
    # for all nodes with respect to the arbitrarily chosen node 0 (a matrix must be
    # inverted). We assemble the system in a vectorized fashion (vectorization over
    # cells) by introducing an additional array axis for both right-hand sides and
    # matrices and reshuffle them such that numpy recognizes a batch of linear systems
    # to be solved. This way we exploit numpy's vectorization of various ufuncs such as
    # det and solve.

    # NOTE: For 2D, the z-coordinate is redundant and would cause issues. Cut it off.
    mat_batch = np.moveaxis(eds[:, :dim, :], -1, 0)  # (nc, dim, dim)

    # Degeneracy check.
    det = np.linalg.det(mat_batch)  # (nc,)
    if not np.all(np.abs(det) > tol):
        raise ValueError("Degenerate simplex with near-zero volume encountered.")

    sq = np.sum(nds**2, axis=1)  # (dim + 1, nc)
    rhs: NDArray[np.float64] = sq[1 : dim + 1] - sq[0][None, :]  # (dim, nc)
    rhs_batch = 0.5 * rhs.T
    assert mat_batch.shape == (nc, dim, dim)
    assert rhs_batch.shape == (nc, dim)

    # Circumcenters (dim, nc)
    ccs = np.linalg.solve(mat_batch, rhs_batch[..., np.newaxis])[..., 0].T

    # For 2D, stack with original z-coordinate.
    if dim == 2:
        ccs = np.vstack((ccs, sd.cell_centers[2, :]))

    # SANITY CHECK: Circumcenter equidistant to nodes.
    dist = np.array([np.linalg.norm(ccs - nds[i, :, :], axis=0) for i in range(dim1p)])
    max_dist = np.maximum.reduce(dist, axis=0)
    min_dist = np.minimum.reduce(dist, axis=0)
    assert np.max(np.abs(max_dist - min_dist)) <= tol, (
        "Circumcenters not equidistant from all nodes."
    )

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
    face_idx = c2f.indices.reshape((dim1p, sd.num_cells), order="F")  # (dim + 1, nc)
    face_sgn = c2f.data.reshape((dim1p, sd.num_cells), order="F")  # (dim + 1, nc)

    # Normalized outwards face normals (3, dim + 1, nc).
    fn_out = sd.face_normals[:, face_idx] * face_sgn[None, :, :]
    fn_out = fn_out / np.linalg.norm(fn_out, axis=0, keepdims=True)

    F = sd.face_centers[:, face_idx]  # Face centers (3, dim + 1, nc).

    # Dot products of all face outwards normals with the shift vector (dim + 1, nc).
    dots: NDArray[np.float64] = np.einsum("ijk,ik->jk", fn_out, shift_vec)
    assert dots.shape == (dim + 1, nc), (
        "Inconsistent dot products of shift vector with normals."
    )
    face_id = np.argmax(dots, axis=0)  # (nc,)

    idx = np.arange(sd.num_cells)
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
    shift = np.where(act_shift <= max_shift, np.ones(nc), max_shift / act_shift)

    # Compute new cell center by applying shift to barycenter.
    nccs: NDArray[np.float64] = bcs + shift * shift_vec
    # Correct shift values in case barycenters and circumcenters coincide.
    shift[act_shift <= tol] = 0.0

    # SANITY CHECK: New cell centers lie strictly in interior (barycentric coordinates
    # strictly positive).

    mat_batch = np.moveaxis(eds[:, :dim, :].transpose(1, 0, 2), -1, 0)  # (nc, dim, dim)
    rhs = (nccs[:dim, :] - nds[0, :dim, :]).T  # (nc, dim)
    # NOTE: determinants of matrices should be non-zero as per degeneracy check above.
    lambdas_r = np.linalg.solve(mat_batch, rhs[..., None])[..., 0].T  # (dim, nc)

    # Full barycentric coordinates.
    lambdas = np.empty((dim + 1, nc), dtype=np.float64)
    lambdas[0, :] = 1.0 - np.sum(lambdas_r, axis=0)
    lambdas[1:, :] = lambdas_r

    # NOTE: Since threshold is never 1, i.e. points strictly interior, the numerics
    # should be stable enough such that the bary-coordinates are strictly positive.
    assert np.all(lambdas > 0), "New cell centers not strictly in interior."
    assert np.allclose(np.sum(lambdas, axis=0), 1.0, atol=tol), (
        "Barycentric weights do not sum to 1."
    )
    assert np.allclose(np.einsum("ik,ijk->jk", lambdas, nds), nccs, rtol=0, atol=tol), (
        "Improper barycentric coordinates."
    )

    # Changes as per specificed tolerance.
    changed: NDArray[np.bool_] = np.linalg.norm(nccs - sd.cell_centers, axis=0) > tol

    # Log total change.
    logger.info(
        "Replaced %d out of %d cell centers on grid %d.", int(changed.sum()), nc, sd.id
    )
    # Log changes where circumcenter not within threshold.
    logger.info(
        "Circumcenter not in threshold in %d out of %d cells on grid %d (bad cells).",
        int(np.sum(shift < 1)),
        nc,
        sd.id,
    )
    assert nccs.shape == sd.cell_centers.shape, (
        "Inconsistent shape for new cell centers."
    )
    assert shift.shape == (nc,), "Inconsistent shape for shift values."
    assert changed.shape == (nc,), "Inconsistent shape for change indicators."

    return nccs, shift, changed  #
