"""This module contains technical tools used in the treatment of fractures.

This can be thought of as a module for backend utility functions, as opposed to the
frontend functions found in :mod:`~porepy.fracs.utils`.

"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp


def determine_mesh_size(
    pts: np.ndarray,
    lines: np.ndarray,
    pts_on_boundary: Optional[np.ndarray] = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sets the preferred mesh size for geometrical points as specified by gmsh.

    This function assumes a 2D mesh given by points ``pts``.

    See the gmsh manual for further details.

    Parameters:
        pts: ``dtype=float, shape=(2, np)``

            The points which will be passed to gmsh, where ``np`` is the number of
            points.
        lines: ``dtype=int, shape=(4, np)``

            Definition and tags of the boundary and fracture lines. Per column (point),
            the first two elements are indices of points in ``pts``, the last two are
            numerical line tags (enumeration).
        pts_on_boundary: ``dtype=bool``

            A boolean array, indicates which of the pts are constitute the domain
            boundary (corners). Only relevant if ``mesh_size_bound`` is passed as a
            keyword argument.

        **kwargs: The mesh size is determined by the following keywords

            - ``'mesh_size_frac'``: Ideal mesh size. Will be added to all points which
              are sufficiently far away from other points.
            - ``'mesh_size_min'``: Minimal mesh size. No attempts are made to enforce
              even smaller mesh sizes using gmsh.
            - ``'mesh_size_bound'``: Boundary mesh size. Will be added to the points
              defining the boundary. If used, the argument ``pts_on_boundary`` is
              mandatory.

    Returns:
        A 3-tuple containing

        :obj:`~numpy.ndarray`: ``shape=(1, np_mod)``

            An array containing per point (from resulting array below) the minimal
            distance to points which are connected by a line.
        :obj:`~numpy.ndarray`: ``shape=(2, np_mod)``

            A modified array of ``np_mod`` points, corresponding to the
            mesh size defined by the arguments (analogous to argument ``pts``).
        :obj:`~numpy.ndarray`:
            A modified array of lines with the same structure as the argument ``lines``.

    """
    num_pts = pts.shape[1]
    val = kwargs.get("mesh_size_frac", 1)
    val_bound = kwargs.get("mesh_size_bound", None)
    val_min = kwargs.get("mesh_size_min", None)
    tol = kwargs.get("tol", 1e-5)
    # One value for each point to distinguish between val and val_bound.
    vals = val * np.ones(num_pts)
    if val_bound is not None:
        vals[pts_on_boundary] = val_bound
    if val_min is None:
        val_min = 1e-8 * val
    # Compute the length of each pair of points (fractures + domain boundary)
    pts_id = lines[:2, :]
    dist = np.linalg.norm(pts[:, pts_id[0, :]] - pts[:, pts_id[1, :]], axis=0)
    dist_pts = np.tile(np.inf, pts.shape[1])

    # Loop on all the points and consider the minimum between the pairs of points
    # lengths associated to the single point and the value input by the user
    for i, pt_id in enumerate(pts_id.T):
        distances = np.array([dist_pts[pt_id], [dist[i]] * 2, vals[pt_id]])
        dist_pts[pt_id] = np.amin(distances, axis=0)

    num_pts = pts.shape[1]

    # Data structures for storing information on extra points.

    # IMPLEMENTATION NOTE, EK: These variables are gradually appended during
    # iterations. This is somewhat costly (estimated to ~10% of the total runtime)
    # for large sets of lines. However, there turned out to be a lot of special cases
    # to cover, so I ended with not trying to implement the necessary changes.
    pts_extra = np.empty((pts.shape[0], 0))
    dist_extra = np.empty(0)
    pts_id_extra = np.empty(0, dtype=int)
    vals_extra = np.empty(0)

    # For each point we compute the distance between the point and the other pairs of
    # points. We keep the minimum distance between the previously computed point
    # distance and the distance among the other pairs of points. If the latter
    # happens, we introduce a new point (useful to determine the grid size) on the
    # corresponding pair of points with a corresponding distance.

    # Find the bounding box for all lines. We only need to do close comparisons with
    # points that are within this bounding box.
    x_min = np.minimum(pts[0, lines[0]], pts[0, lines[1]])
    x_max = np.maximum(pts[0, lines[0]], pts[0, lines[1]])
    y_min = np.minimum(pts[1, lines[0]], pts[1, lines[1]])
    y_max = np.maximum(pts[1, lines[0]], pts[1, lines[1]])

    # Loop over all lines
    for line_ind, line in enumerate(lines.T):
        # Find start and endpoint of this line
        start, end = pts[:, line[0]], pts[:, line[1]]

        # Size of buffer region for bounding box.
        buff = 1.1 * np.minimum(vals[line[0]], vals[line[1]])

        # Lines which are outside the bounding box
        outside_lines = np.logical_or.reduce(
            (
                x_max < x_min[line_ind] - buff,
                x_min > x_max[line_ind] + buff,
                y_max < y_min[line_ind] - buff,
                y_min > y_max[line_ind] + buff,
            )
        )

        # Index of lines inside the box.
        inside_lines = np.where(np.logical_not(outside_lines))[0].ravel()

        # Points that are inside the bounding box: All points on lines inside the
        # box. Note that we are careful when to add indexing on these points below.
        inside_pts = np.unique(lines[:2, inside_lines])

        # Compute the distance between this line and all points inside the box. This
        # will also include the start and endpoint of this line, but we deal with
        # that later
        dist, cp = pp.distances.points_segments(pts[:, inside_pts], start, end)

        # We know there is a single segment, thus the distance vector can be reduced
        dist = dist[:, 0]
        # Find points that are closer than the target distance
        hit = np.where(
            np.logical_and(
                dist < vals[inside_pts], np.logical_not(np.isclose(dist, 0.0))
            )
        )[0]
        # Update the minimum distance found for these points
        dist_pts[inside_pts[hit]] = np.minimum(dist_pts[inside_pts[hit]], dist[hit])

        # cp[hit, 0] now contains points on the main line that are closest to another
        # point, and that sufficiently close to warrant attention. Compute the
        # distance from cp to the start and endpoint of this line. Transposes are
        # needed here because that is how numpy works.
        dist_start = pp.distances.point_pointset(start, cp[hit, 0].T)
        dist_end = pp.distances.point_pointset(end, cp[hit, 0].T)

        # Now, the cp points are added if they are closer to another point than to
        # the start and end point of its line, and if the distance from the start and
        # end is not smaller than the minimum point. The latter removes lines having
        # their own end points added, and also avoids arbitrarily small segments
        # along the line.
        to_add = np.logical_and.reduce((dist[hit] < dist_start, dist[hit] < dist_end))
        if np.any(to_add):
            dist_extra = np.r_[
                dist_extra, np.minimum(dist[hit[to_add]], vals[hit[to_add]])
            ]
            pts_extra = np.c_[pts_extra, cp[hit[to_add], 0].T]
            pts_id_extra = np.r_[
                pts_id_extra, line[3] * np.ones(to_add.sum(), dtype=int)
            ]
            vals_extra = np.r_[vals_extra, vals[hit[to_add]]]

    old_lines = lines
    old_pts = pts

    # Since the computation was done point by point with the lines, we need to
    # consider all the new points together and remove (from the new points) the
    # useless ones.
    extra_ids, inv_index = np.unique(pts_id_extra, return_inverse=True)

    to_remove = np.empty(0, dtype=int)

    for idx, i in enumerate(extra_ids):
        mask = np.flatnonzero(inv_index == idx)
        if mask.size > 1:
            mesh_matrix = np.tile(dist_extra[mask], (mask.size, 1))
            dist_matrix = np.ones((mask.size, mask.size)) * np.inf

            for pt1_id_loc in np.arange(mask.size):
                for pt2_id_loc in np.arange(pt1_id_loc + 1, mask.size):
                    pt1_id = mask[pt1_id_loc]
                    pt2_id = mask[pt2_id_loc]
                    pt1 = pts_extra[:, pt1_id]
                    pt2 = pts_extra[:, pt2_id]
                    pts_id = np.array([pt1_id, pt2_id])
                    pts_id_loc = np.array([pt1_id_loc, pt2_id_loc])
                    pos_min = np.argmin(dist_extra[pts_id])
                    pos_max = np.argmax(dist_extra[pts_id])
                    dist_matrix[pts_id_loc[pos_min], pts_id_loc[pos_max]] = (
                        np.linalg.norm(pt1 - pt2)
                    )

            to_remove_loc = np.any(dist_matrix < mesh_matrix, axis=0)
            to_remove = np.r_[to_remove, mask[to_remove_loc]]

    # Remove the useless new points
    pts_extra = np.delete(pts_extra, to_remove, axis=1)
    dist_extra = np.delete(dist_extra, to_remove)
    pts_id_extra = np.delete(pts_id_extra, to_remove)
    vals_extra = np.delete(vals_extra, to_remove)

    # Consider all the points
    pts = np.c_[pts, pts_extra]
    dist_pts = np.r_[dist_pts, dist_extra]
    vals = np.r_[vals, vals_extra]
    # Re-create the lines, considering the new introduced points
    seg_ids = np.unique(lines[3, :])
    new_lines = np.empty((4, 0), dtype=int)
    for seg_id in seg_ids:
        mask_bool = lines[3, :] == seg_id
        extra_mask_bool = pts_id_extra == seg_id
        if not np.any(extra_mask_bool):
            # No extra points are considered for the current line
            new_lines = np.c_[new_lines, lines[:, mask_bool]]
        else:
            # New extra point are considered for the current line, they need to
            # be sorted along the line.
            pts_frac_id = np.hstack(
                (
                    lines[0:2, mask_bool].ravel(),
                    np.flatnonzero(extra_mask_bool) + num_pts,
                )
            )
            pts_frac_id = np.unique(pts_frac_id)
            pts_frac = pts[:, pts_frac_id]

            # We will sort points on the line below, but this function requires 3D
            # points
            pts_frac_aug = np.vstack((pts_frac, np.zeros(pts_frac.shape[1])))
            pts_frac_id = pts_frac_id[
                pp.map_geometry.sort_points_on_line(pts_frac_aug, tol)
            ]
            pts_frac_id = np.vstack((pts_frac_id[:-1], pts_frac_id[1:]))
            other_info = np.tile(
                lines[2:, mask_bool][:, 0], (pts_frac_id.shape[1], 1)
            ).T
            new_lines = np.c_[new_lines, np.vstack((pts_frac_id, other_info))]

    # Consider extra points related to the input value, if the fracture is long and,
    # because of val, needs additional points we increase the number of lines.

    # Make bounding boxes for the new segments.
    x_min_new = np.minimum(pts[0, new_lines[0]], pts[0, new_lines[1]])
    x_max_new = np.maximum(pts[0, new_lines[0]], pts[0, new_lines[1]])
    y_min_new = np.minimum(pts[1, new_lines[0]], pts[1, new_lines[1]])
    y_max_new = np.maximum(pts[1, new_lines[0]], pts[1, new_lines[1]])

    relax = kwargs.get("relaxation", 0.8)

    # Make a list of new lines, will convert to numpy array towards the end
    line_list = []

    for seg_ind, seg in enumerate(new_lines.T):
        mesh_size_pt1 = dist_pts[seg[0]]
        mesh_size_pt2 = dist_pts[seg[1]]
        dist = np.linalg.norm(pts[:, seg[0]] - pts[:, seg[1]])
        if (
            mesh_size_pt1 >= relax * vals[seg[0]]
            and mesh_size_pt2 >= relax * vals[seg[1]]
        ) or (relax * dist <= 2 * mesh_size_pt1 and relax * dist <= 2 * mesh_size_pt2):
            line_list.append(seg.tolist())
        else:
            pt_id = pts.shape[1]
            new_pt = 0.5 * (pts[:, seg[0]] + pts[:, seg[1]])
            pts = np.c_[pts, new_pt]

            mesh_size = np.amin(np.r_[vals[seg[:2]], dist / 2.0])

            # Size of buffer region for bounding box. Not sure if this can be made
            # smaller, but it may not matter too much.
            buff = 1.1 * mesh_size

            # Compare the bounding boxes of the old segments with the box for this
            # new segment.
            outside_lines = np.logical_or.reduce(
                (
                    x_max < x_min_new[seg_ind] - buff,
                    x_min > x_max_new[seg_ind] + buff,
                    y_max < y_min_new[seg_ind] - buff,
                    y_min > y_max_new[seg_ind] + buff,
                )
            )

            # Index of lines inside the box.
            inside_lines = np.where(np.logical_not(outside_lines))[0].ravel()

            # Find the closest point of the old lines inside the box.
            if inside_lines.size > 0:
                start_old = old_pts[:, old_lines[0, inside_lines]]
                end_old = old_pts[:, old_lines[1, inside_lines]]

                # Compute the ditsacne between the current point and all old lines
                dist1, _ = pp.distances.points_segments(new_pt, start_old, end_old)

                # Disregard points that lie on the old segment by assigning a value
                # so high that it will not be picked up by the minimum.
                dist1[np.isclose(dist1, 0.0)] = mesh_size * 10

            else:
                # Nothing to do here, assign value that will be disregarded.
                dist1 = np.array([mesh_size * 10])

            # Update the minimum mesh size if any dist1 is less than the current value
            mesh_size = np.minimum(mesh_size, dist1.min())

            dist_pts = np.r_[dist_pts, mesh_size]
            vals = np.r_[vals, mesh_size]
            line_list.append([seg[0], pt_id, seg[2], seg[3]])
            line_list.append([pt_id, seg[1], seg[2], seg[3]])

    lines = np.array(line_list, dtype=np.int32).T

    return dist_pts, pts, lines


def obtain_interdim_mappings(
    g: pp.Grid, fn: sps.spmatrix, n_per_face: int
) -> tuple[np.ndarray, np.ndarray]:
    """Finds mappings between faces in higher dimension and cells in the lower
    dimension.

    Parameters:
        g: Lower dimensional grid.
        fn: Face-node map of the higher-dimensional grid
            (see :data:`~porepy.grids.grid.Grid.face_nodes`).
        n_per_face: Number of nodes per face in the higher-dimensional grid.

    Returns:
        A 2-tuple containing

        :obj:`~numpy.ndarray`:
            An array containing indices of faces in the higher-dimensional grid that
            correspond to a cell in the lower-dimensional grid.
            The indexing is based on **all** cells in the lower-dimensional grid.
        :obj:`~numpy.ndarray`:
            Indices of the corresponding cells in the lower-dimensional grid.

    """
    if g.dim > 0:
        cn_loc = g.cell_nodes().indices.reshape((n_per_face, g.num_cells), order="F")
        cn = g.global_point_ind[cn_loc]
        cn = np.sort(cn, axis=0)
    else:
        cn = np.array([g.global_point_ind])
        # We also know that the higher-dimensional grid has faces of a single node.
        # This sometimes fails, so enforce it.
        if cn.ndim == 1:
            fn = fn.ravel()
    is_mem, cell_2_face = pp.utils.setmembership.ismember_rows(
        cn.astype(np.int32), fn.astype(np.int32), sort=False
    )
    # An element in cell_2_face gives, for all cells in the lower-dimensional grid,
    # the index of the corresponding face in the higher-dimensional structure.
    if not (np.all(is_mem) or np.all(~is_mem)):
        warnings.warn(
            """Found inconsistency between cells and higher dimensional faces.
            Continuing, fingers crossed"""
        )
    low_dim_cell = np.where(is_mem)[0]
    return cell_2_face, low_dim_cell
