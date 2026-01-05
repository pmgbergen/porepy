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
    from porepy.grids.simplex import TriangleGrid

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
    sd: TriangleGrid, threshold_angle: float = np.pi * 0.45
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Compute circumcenters of triangular cells in 2D grid.

    Parameters:
        sd: A 2D structured or unstructured triangular grid.
        threshold_angle: Threshold angle (in radians). The circumcenter will replace the
            cell center only in those triangles where all angles in the triangle are
            below this threshold.

            Note that if the threshold is set larger than 0.5 * pi, cells with
            circumcenters outside the cell will have their cell centers replaced.

    Returns:
        Tuple with:
        - New cell centers where circumcenters have replaced original centers for cells
            with all angles below minimum_angle.
        - A boolean array indicating which cells had their centers replaced.

    Raises:
        ValueError: If degenerate triangles with zero area are encountered.
        ValueError: If computed angles do not sum to pi.
    """
    # Extract node coordinates for all cells.
    cn = sd.cell_nodes().tocsc()
    ni = cn.indices.reshape((3, sd.num_cells), order="F")
    cc = sd.cell_centers.copy()
    x = sd.nodes[0]
    y = sd.nodes[1]
    x0 = x[ni[0]]
    y0 = y[ni[0]]
    x1 = x[ni[1]]
    y1 = y[ni[1]]
    x2 = x[ni[2]]
    y2 = y[ni[2]]
    # Compute circumcenters. First compute determinant D.
    D = 2 * (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))
    if not np.all(D != 0):
        raise ValueError("Degenerate triangle with zero area encountered.")
    # Compute circumcenter coordinates.
    xc = (
        (x0**2 + y0**2) * (y1 - y2)
        + (x1**2 + y1**2) * (y2 - y0)
        + (x2**2 + y2**2) * (y0 - y1)
    ) / D
    yc = (
        -(
            (x0**2 + y0**2) * (x1 - x2)
            + (x1**2 + y1**2) * (x2 - x0)
            + (x2**2 + y2**2) * (x0 - x1)
        )
        / D
    )
    # Compute angles at each node using the law of cosines.
    # Construct vectors between nodes to compute angles.
    d_01 = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    d_12 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    d_20 = np.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2)

    dot_0 = (x1 - x0) * (x2 - x0) + (y1 - y0) * (y2 - y0)
    dot_1 = (x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)
    dot_2 = (x0 - x2) * (x1 - x2) + (y0 - y2) * (y1 - y2)

    # Guard against tiny numerical drift in the acos argument by clipping to [-1, 1]
    # to avoid NaNs for nearly-degenerate or floating-point perturbed inputs. Ignoring
    # invalid value warnings since we handle them via clipping.
    with np.errstate(invalid="ignore"):
        cos0 = np.clip(dot_0 / (d_01 * d_20), -1.0, 1.0)
        cos1 = np.clip(dot_1 / (d_01 * d_12), -1.0, 1.0)
        cos2 = np.clip(dot_2 / (d_12 * d_20), -1.0, 1.0)
    angle_0 = np.arccos(cos0)
    angle_1 = np.arccos(cos1)
    angle_2 = np.arccos(cos2)
    # Verify that angles sum to pi.
    if not np.allclose(angle_0 + angle_1 + angle_2, np.pi):
        raise ValueError("Computed angles do not sum to pi.")
    # Replace cell centers with circumcenters for cells with all angles below
    # threshold_angle.
    replace = np.logical_and.reduce(
        [
            angle_0 < threshold_angle,
            angle_1 < threshold_angle,
            angle_2 < threshold_angle,
        ]
    )
    cc[0, replace] = xc[replace]
    cc[1, replace] = yc[replace]

    return cc, replace


def compute_circumcenter_3d(
    sd: Grid, threshold_angle: float = np.pi * 0.45
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Compute circumcenters of tetrahedral cells in 3D grid.

    Parameters:
        sd: A 3D structured or unstructured grid.
        threshold_angle: Threshold angle (in radians). The circumcenter will replace the
            cell center only in those tetrahedra where all dihedral angles, i.e., the
            angles between faces, are below this threshold.
    Returns:
        Tuple with:
        - New cell centers where circumcenters have replaced original centers for cells
            where all dihedral angles are below ``threshold_angle`` (analogous to the 2D
            version's triangle angles).
        - A boolean array indicating which cells had their centers replaced.

    Raises:
        ValueError: If circumcenters are not equidistant from all nodes.

    """
    # Extract node coordinates for all cells.
    cn = sd.cell_nodes().tocsc()
    ni = cn.indices.reshape((sd.dim + 1, sd.num_cells), order="F")
    x, y, z = sd.nodes[0], sd.nodes[1], sd.nodes[2]
    x0 = x[ni[0]]
    y0 = y[ni[0]]
    z0 = z[ni[0]]
    x1 = x[ni[1]]
    y1 = y[ni[1]]
    z1 = z[ni[1]]
    x2 = x[ni[2]]
    y2 = y[ni[2]]
    z2 = z[ni[2]]
    x3 = x[ni[3]]
    y3 = y[ni[3]]
    z3 = z[ni[3]]

    # Compute matrix A and its inverse for each cell, based on the tetrahedron vertices
    # as described in:
    # https://en.wikipedia.org/wiki/Tetrahedron
    # https://rodolphe-vaillant.fr/entry/127/find-a-tetrahedron-circumcenter
    A = np.array(
        [
            [x1 - x0, y1 - y0, z1 - z0],
            [x2 - x0, y2 - y0, z2 - z0],
            [x3 - x0, y3 - y0, z3 - z0],
        ]
    )

    # Construct right-hand side vector B.
    B = 0.5 * np.array(
        [
            (x1**2 + y1**2 + z1**2) - (x0**2 + y0**2 + z0**2),
            (x2**2 + y2**2 + z2**2) - (x0**2 + y0**2 + z0**2),
            (x3**2 + y3**2 + z3**2) - (x0**2 + y0**2 + z0**2),
        ]
    )
    # Compute circumcenters by solving A c = B for each cell (avoid explicit inverse).
    center: NDArray[np.float64] = np.empty((A.shape[2], 3), dtype=float)
    for i in range(A.shape[2]):
        center[i, :] = np.linalg.solve(A[:, :, i], B[:, i])

    # Check that the circumcenter is equidistant from all nodes.
    distance_node_center = []
    for ind in ni:
        dist = np.sqrt(np.sum((sd.nodes[:, ind] - center.T) ** 2, axis=0))
        distance_node_center.append(dist)

    max_distance = np.max(np.abs(distance_node_center), axis=0)
    min_distance = np.min(np.abs(distance_node_center), axis=0)
    # Use a relative tolerance scaled by the radius to avoid false negatives on
    # large/small cells.
    radius = 0.5 * (max_distance + min_distance) + 1e-15
    if np.max((max_distance - min_distance) / radius) >= 1e-10:
        raise ValueError("Circumcenter not equidistant from all nodes.")

    # Decide replacement using a dihedral-angle criterion analogous to 2D.
    # For each cell, construct outward unit normals for its four faces, then compute
    # the six internal dihedral angles and require all to be below the threshold.
    cf_csc = sd.cell_faces.tocsc()
    faces_idx = cf_csc.indices
    faces_data = cf_csc.data
    faces_ptr = cf_csc.indptr

    replace = np.zeros(sd.num_cells, dtype=bool)
    for c in range(sd.num_cells):
        loc = slice(faces_ptr[c], faces_ptr[c + 1])
        f_loc = faces_idx[loc]
        # Orientation sign per face relative to cell.
        sgn = np.sign(faces_data[loc])
        # Outward unit normals per face.
        n = (sd.face_normals[:, f_loc] / sd.face_areas[f_loc]) * sgn
        # Normalize to guard against numerical drift.
        n = n / (np.linalg.norm(n, axis=0) + 1e-15)

        # Compute all six dihedral angles between face pairs: θ = arccos( - n_i · n_j ).
        # Pairs (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
        angles_list: list[float] = []
        for i in range(4):
            for j in range(i + 1, 4):
                dot = np.dot(n[:, i], n[:, j])
                dot = float(np.clip(dot, -1.0, 1.0))
                angles_list.append(float(np.arccos(-dot)))
        dihedral_angles = np.array(angles_list, dtype=float)
        # Replace iff all dihedral angles are below threshold.
        replace[c] = bool(np.all(dihedral_angles < threshold_angle))

    new_centers = sd.cell_centers.copy()
    if np.any(replace):
        new_centers[:, replace] = center.T[:, replace]

    logger.info(
        "Replaced %d out of %d cell centers.", int(replace.sum()), int(sd.num_cells)
    )
    # Additional verification: For internal faces between two cells that had their
    # centers replaced, the vector between the two circumcenters should be parallel to
    # the face normal for those faces.
    fc = sd.cell_faces_as_dense()
    # Note: fc = -1 (boundary faces) will not be found in replace. Thus, we only
    # consider internal faces here.
    internal_replaced = np.all(np.isin(fc, np.where(replace)[0]), axis=0)
    # Verify that the circumcenter-to-circumcenter vector across internal faces is
    # parallel to the face normal for replaced cells.
    cc_vec = (
        new_centers[:, fc[0, internal_replaced]]
        - new_centers[:, fc[1, internal_replaced]]
    )
    normal = sd.face_normals[:, internal_replaced]
    # Compute cross product between cc_vec and normal.
    cc_vec_cross_normal = np.vstack(
        (
            cc_vec[1] * normal[2] - cc_vec[2] * normal[1],
            cc_vec[2] * normal[0] - cc_vec[0] * normal[2],
            cc_vec[0] * normal[1] - cc_vec[1] * normal[0],
        )
    )
    # Relative colinearity check: ||a x b|| <= tol * ||a|| ||b|| for all internal faces
    if cc_vec_cross_normal.size:
        cross_norm = np.linalg.norm(cc_vec_cross_normal, axis=0)
        denom = np.linalg.norm(cc_vec, axis=0) * np.linalg.norm(normal, axis=0) + 1e-15
        if np.max(cross_norm / denom) >= 1e-10:
            raise ValueError(
                "Circumcenter not aligned with face normals for replaced cells."
            )
    return new_centers, replace
