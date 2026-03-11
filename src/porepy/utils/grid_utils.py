"""Module contains various utility functions for working with grids."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

from porepy.geometry.half_space import half_space_interior_point
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data

if TYPE_CHECKING:  # Avoid importing heavyish modules at runtime purely for typing.
    from porepy.grids.grid import Grid
    from porepy.grids.simplex import TetrahedralGrid, TriangleGrid

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


def compute_circumcenter_2d(
    sd: TriangleGrid,
    threshold: float = 0.95,
    eps: float = 1e-14,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute circumcenters of triangular cells in 2D grid.

    Parameters:
        sd: A 2D structured or unstructured triangular grid.
        threshold: ``default=0.95``

            Used for cases where the circumcenter is not in the interior of the
            triangle. The center is moved 95% of the distance from barycenter to
            triangle boundary, in the direction circumcenter, ensuring the new center
            lies strictly inside the triangle and approximates the circumcenter.
        eps: ``default=1e-14``

            Tolerance for detecting degenerate triangles and changes in circumcenter.

    Raises:
        ValueError: If degenerate triangles are detected.
        ValueError: If ``threshold`` is not in (0, 1).

    Returns:
        A 3-tuple containing

        1. the new cell centers of shape ``sd.cell_centers``.
        2. the shift values of shape ``(sd.num_cells,)``.
        3. a boolean array of shape ``(sd.num_cells,)`` indicating where numerically
           relevant changes in cell center occurred.

        The shift values contain the scale of the vector going from current center
        to circumcenter per cell. A value of 1 indicates the circumcenter is strictly
        in the interior of the triangle. Values below that indicate a fraction of the
        movement along the vector such that the new center is still strictly in the
        interior (in other words, the triangle is not acute and the circumcenter lies
        on the boundary or outside the triangle).

    """
    if not (0 < threshold < 1):
        raise ValueError(f"Threshold must be in (0, 1), got {threshold}.")

    eps_loc = eps * max(sd.dim, 2)  # Account for some dimensionality.

    # Extract node coordinates for all cells.
    cn = sd.cell_nodes().tocsc()
    ni = cn.indices.reshape((3, sd.num_cells), order="F")
    x = sd.nodes[0]
    y = sd.nodes[1]

    # Nodes spanning triangle.
    A = np.vstack((x[ni[0]], y[ni[0]]))
    B = np.vstack((x[ni[1]], y[ni[1]]))
    C = np.vstack((x[ni[2]], y[ni[2]]))

    # Compute circumcenters. First compute determinant D.
    D = 2.0 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
    if not np.all(abs(D) > eps):
        raise ValueError("Degenerate triangle with zero area encountered.")

    # Norm squared of coordinates.
    a2 = np.sum(np.square(A), axis=0)
    b2 = np.sum(np.square(B), axis=0)
    c2 = np.sum(np.square(C), axis=0)
    # Edges.
    A2B = B - A
    B2C = C - B
    C2A = A - C
    # Squared side lengths.
    ab2 = np.sum(np.square(A2B), axis=0)
    bc2 = np.sum(np.square(B2C), axis=0)
    ca2 = np.sum(np.square(C2A), axis=0)

    # Compute circumcenter coordinates.
    CC = np.array(
        (
            -(a2 * B2C[1] + b2 * C2A[1] + c2 * A2B[1]) / D,
            (a2 * B2C[0] + b2 * C2A[0] + c2 * A2B[0]) / D,
        )
    )

    # Mask indicating acute triangles, only triangles where circumcenter is strictly
    # in the interior of the triangle. NOTE: Thales' theorem.
    is_acute = (ab2 + bc2 > ca2) & (ab2 + ca2 > bc2) & (bc2 + ca2 > ab2)

    # New cell centers.
    NCC = sd.cell_centers.copy()
    NCC[:2, is_acute] = CC[:, is_acute]
    shifts = np.ones(sd.num_cells, dtype=np.float64)

    # If any obtuse or right triangle, the new centers are not strictly inside.
    # Instead we compute barycenters, and move in direction circumcenter.
    not_acute = ~is_acute
    if np.any(not_acute):
        # Starting point: barycenters.
        BCC = (A + B + C)[:, not_acute] / 3.0
        # Shift vector from barycenters to circumcenters.
        V = CC[:, not_acute] - BCC

        # Assume counter-clockwise (CCW) order ABC of triangle, A being lower left node.
        # Get sign if not CCW.
        signed_area_2 = A2B[1] * C2A[0] - A2B[0] * C2A[1]
        sign = np.sign(signed_area_2)
        # sign = np.where(sign == 0, 1.0, sign)  # In case almost degenerate

        # Compute outward normal by rotating edges, mind the sign in case not CCW.
        nA2B = np.array((A2B[1] * sign, -A2B[0] * sign))[:, not_acute]
        nB2C = np.array((B2C[1] * sign, -B2C[0] * sign))[:, not_acute]
        nC2A = np.array((C2A[1] * sign, -C2A[0] * sign))[:, not_acute]

        intercept_ba = np.sum(nA2B * V, axis=0) > 0
        intercept_bc = np.sum(nB2C * V, axis=0) > 0
        intercept_ca = np.sum(nC2A * V, axis=0) > 0
        # Sanity check: should be mutually exclusive and cover all cells.
        check = np.vstack((intercept_ba, intercept_bc, intercept_ca)).sum(axis=0)
        assert np.all(check == 1), "Failed to find unique intercepting face."

        # Allocate modified cell centers and shifts.
        MCC = np.full_like(BCC, np.nan)
        mshifts = np.full_like(BCC[0], np.nan)

        to_ = CC[:, not_acute]
        for intercept, P, N in [
            (intercept_ba, A, nA2B),
            (intercept_bc, B, nB2C),
            (intercept_ca, C, nC2A),
        ]:
            if np.any(intercept):
                from_ = BCC[:, intercept]  # from barycenter
                to_i = to_[:, intercept]  # to circumcenter
                pof = P[:, not_acute][:, intercept]  # point on face
                fn = N[:, intercept]  # face normal

                # NOTE: Dot product by design positive, this is a failsafe for numerical
                # robustness.
                denom = np.maximum(np.sum((to_i - from_) * fn, axis=0), eps)
                # Apply threshold to stay in interior.
                t = np.sum((pof - from_) * fn, axis=0) / denom * threshold

                MCC[:, intercept] = from_ + V[:, intercept] * t
                mshifts[intercept] = t

        assert np.all(~np.isnan(MCC)), "Failed to find modified cell centers."
        # This will also raise errors if nans are still present.
        assert np.all(mshifts) >= 0.0, "Shift must be non-negative."
        assert np.all(mshifts) <= 1.0, "Shift must be at most 1."

        NCC[:2, not_acute] = MCC
        shifts[not_acute] = mshifts

    # Sanity check: Computed cell centers are strictly inside triangle using barycentric
    # coordinates.
    v0 = -C2A  # vector from A to C
    v1 = A2B  # vector from A to B
    v2 = NCC[:2] - A  # vector from A to center.
    dot00 = np.sum(np.square(v0), axis=0)
    dot11 = np.sum(np.square(v1), axis=0)
    dot01 = np.sum(v0 * v1, axis=0)
    dot02 = np.sum(v0 * v2, axis=0)
    dot12 = np.sum(v1 * v2, axis=0)

    denom = dot00 * dot11 - np.square(dot01)
    # Should not happen after check above, but nevertheless.
    assert np.all(np.abs(denom) > eps), "Degeneration detected."
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    assert (
        np.all(u > -eps_loc) and np.all(v > -eps_loc) and np.all(u + v < 1 + eps_loc)
    ), "New cell centers not strictly in interior."

    # Compute indicators where center changed (column-wise norm).
    # NOTE: Numerically there is always a change. Furthermore, by default the cell
    # centers are originally at the barycenter. It coincides with the circumcenter if
    # and only if triangle is equilateral. We use simply the distance between new and
    # old centers to indicate numerically relevant change.
    changed = np.sqrt(np.sum(np.square(NCC - sd.cell_centers), axis=0)) > eps_loc

    logger.info(
        "Replaced %d out of %d cell centers.", int(changed.sum()), int(sd.num_cells)
    )

    return NCC, shifts, changed


def compute_circumcenter_3d(
    sd: TetrahedralGrid,
    threshold: float = 0.95,
    eps: float = 1e-14,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Compute circumcenter-based cell centers for tetrahedral cells in 3D.

    Parameters:
        sd: A 3D tetrahedral grid.
        threshold: default=0.95

            Used for cells where the circumcenter is not strictly in the interior of the
            tetrahedron. The new center is placed at `threshold` times the distance from
            barycenter to the boundary, in the direction of the circumcenter, ensuring
            the new center lies strictly inside the tetrahedron.
        eps: default=1e-14

            Tolerance for detecting degenerate tetrahedra and numerically relevant
            changes in cell center.

    Raises:
        ValueError: If degenerate tetrahedra are detected.
        ValueError: If ``threshold`` is not in (0, 1).

    Returns:
        A 3-tuple containing

        1. the new cell centers of shape ``sd.cell_centers``.
        2. the shift values of shape ``(sd.num_cells,)``.
        3. a boolean array of shape ``(sd.num_cells,)`` indicating where numerically
           relevant changes in cell center occurred.

        Shift semantics:
        - 1.0 means the circumcenter was used directly (strictly interior).
        - values below 1.0 mean only part of the move from barycenter to circumcenter
          was used so the new center stays strictly inside the tetrahedron.

    """
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"Threshold must be in (0, 1), got {threshold}.")

    if sd.dim != 3:
        raise ValueError(f"Expected a 3D grid, got dim={sd.dim}.")

    eps_loc = eps * max(sd.dim, 3)

    # Extract tetrahedron nodes for all cells.
    cn = sd.cell_nodes().tocsc()
    ni = cn.indices.reshape((4, sd.num_cells), order="F")

    x, y, z = sd.nodes[0], sd.nodes[1], sd.nodes[2]

    A = np.vstack((x[ni[0]], y[ni[0]], z[ni[0]]))
    B = np.vstack((x[ni[1]], y[ni[1]], z[ni[1]]))
    C = np.vstack((x[ni[2]], y[ni[2]], z[ni[2]]))
    D = np.vstack((x[ni[3]], y[ni[3]], z[ni[3]]))

    # Compute circumcenters by solving a 3x3 linear system per cell.
    # Let c be the circumcenter. Then:
    #   (B-A)·c = (|B|^2 - |A|^2)/2
    #   (C-A)·c = (|C|^2 - |A|^2)/2
    #   (D-A)·c = (|D|^2 - |A|^2)/2
    AB = B - A
    AC = C - A
    AD = D - A

    M = np.empty((3, 3, sd.num_cells), dtype=np.float64)
    M[0, :, :] = AB
    M[1, :, :] = AC
    M[2, :, :] = AD

    a2 = np.sum(np.square(A), axis=0)
    b2 = np.sum(np.square(B), axis=0)
    c2 = np.sum(np.square(C), axis=0)
    d2 = np.sum(np.square(D), axis=0)

    rhs = 0.5 * np.vstack((b2 - a2, c2 - a2, d2 - a2))

    # Degeneracy test: 6 * volume = det([B-A, C-A, D-A]).
    detM = np.linalg.det(np.moveaxis(M, 2, 0))
    if not np.all(np.abs(detM) > eps):
        raise ValueError("Degenerate tetrahedron with near-zero volume encountered.")

    CC = np.empty((3, sd.num_cells), dtype=np.float64)
    for i in range(sd.num_cells):
        CC[:, i] = np.linalg.solve(M[:, :, i], rhs[:, i])

    # Verify circumcenters are equidistant from all four nodes.
    distA = np.linalg.norm(CC - A, axis=0)
    distB = np.linalg.norm(CC - B, axis=0)
    distC = np.linalg.norm(CC - C, axis=0)
    distD = np.linalg.norm(CC - D, axis=0)

    max_dist = np.maximum.reduce((distA, distB, distC, distD))
    min_dist = np.minimum.reduce((distA, distB, distC, distD))
    radius = 0.5 * (max_dist + min_dist) + eps
    if np.max((max_dist - min_dist) / radius) >= 1e-10:
        raise ValueError("Circumcenter not equidistant from all nodes.")

    # Test whether circumcenters are strictly inside using barycentric coordinates.
    # We write:
    #   CC = A + u(B-A) + v(C-A) + w(D-A)
    # Then barycentric coords are:
    #   λ0 = 1-u-v-w, λ1 = u, λ2 = v, λ3 = w
    # Strict interior <=> all λi > 0.
    bary = np.empty((4, sd.num_cells), dtype=np.float64)
    for i in range(sd.num_cells):
        uvw = np.linalg.solve(M[:, :, i].T, CC[:, i] - A[:, i])
        u, v, w = uvw
        bary[:, i] = np.array([1.0 - u - v - w, u, v, w])

    is_interior = np.all(bary > eps_loc, axis=0)

    # Initialize outputs.
    NCC = sd.cell_centers.copy()
    NCC[:3, is_interior] = CC[:, is_interior]
    shifts = np.ones(sd.num_cells, dtype=np.float64)

    # Non-interior circumcenters: start from barycenter and move toward circumcenter,
    # but stop before crossing the boundary.
    # In 2D exactly one face has n·V > 0.
    # In 3D there may be several faces with n·V > 0, so we choose the face with the
    # maximal value of n·V.
    not_interior = ~is_interior
    if np.any(not_interior):
        BCC = (A + B + C + D) / 4.0
        V_all = CC - BCC

        # Cell-face connectivity.
        cf = sd.cell_faces.tocsc()
        face_idx = cf.indices
        face_sgn = cf.data
        face_ptr = cf.indptr

        MCC = np.full((3, np.sum(not_interior)), np.nan, dtype=np.float64)
        mshifts = np.full(np.sum(not_interior), np.nan, dtype=np.float64)

        not_int_cells = np.where(not_interior)[0]

        for local_j, cell in enumerate(not_int_cells):
            from_ = BCC[:, cell]
            V = V_all[:, cell]

            # If circumcenter and barycenter nearly coincide, keep barycenter.
            if np.linalg.norm(V) <= eps_loc:
                MCC[:, local_j] = from_
                mshifts[local_j] = 0.0
                continue

            loc = slice(face_ptr[cell], face_ptr[cell + 1])
            f_loc = face_idx[loc]
            sgn = np.sign(face_sgn[loc])

            if f_loc.size != 4:
                raise ValueError(
                    f"Expected 4 faces for tetrahedral cell {cell}, got {f_loc.size}."
                )

            # Outward face normals.
            normals = (sd.face_normals[:, f_loc] / sd.face_areas[f_loc]) * sgn
            normal_norms: NDArray[np.float64] = np.linalg.norm(normals, axis=0)
            normals /= normal_norms + eps

            # Alignment of motion direction with outward normals.
            dots = normals.T @ V

            # Choose face with maximal positive dot product.
            positive = dots > eps
            if not np.any(positive):
                # Fallback: this should not happen for a strictly interior barycenter
                # and a target outside the tetrahedron, but keep code robust.
                best = int(np.argmax(dots))
            else:
                pos_idx = np.where(positive)[0]
                best = int(pos_idx[np.argmax(dots[pos_idx])])

            n = normals[:, best]
            f = f_loc[best]
            pof = sd.face_centers[:, f]

            denom = max(float(np.dot(V, n)), eps)
            t = float(np.dot(pof - from_, n) / denom) * threshold

            # Robust clipping.
            t = max(0.0, min(1.0, t))

            MCC[:, local_j] = from_ + t * V
            mshifts[local_j] = t

        assert np.all(~np.isnan(MCC)), "Failed to compute modified cell centers."
        assert np.all(mshifts >= 0.0), "Shift must be non-negative."
        assert np.all(mshifts <= 1.0), "Shift must be at most 1."

        NCC[:3, not_interior] = MCC
        shifts[not_interior] = mshifts

    # Final sanity check: all new centers lie inside tetrahedra.
    final_bary = np.empty((4, sd.num_cells), dtype=np.float64)
    for i in range(sd.num_cells):
        uvw = np.linalg.solve(M[:, :, i].T, NCC[:3, i] - A[:, i])
        u, v, w = uvw
        final_bary[:, i] = np.array([1.0 - u - v - w, u, v, w])

    assert np.all(final_bary > -eps_loc), "New cell centers not in interior."

    # Compute indicators where center changed (column-wise norm).
    changed: NDArray[np.bool_] = (
        np.sqrt(np.sum(np.square(NCC[:3] - sd.cell_centers[:3]), axis=0)) > eps_loc
    )

    logger.info(
        "Replaced %d out of %d cell centers.", int(changed.sum()), int(sd.num_cells)
    )

    return NCC, shifts, changed
