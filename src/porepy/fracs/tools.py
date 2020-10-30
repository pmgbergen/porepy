#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Technical tools used in treatment of fractures.

This can be thought of as a module for backend utility functions, as opposed to
the frontend functions found in utils.

"""
import warnings

import numpy as np

import porepy as pp


def determine_mesh_size(pts, pts_on_boundary=None, lines=None, **kwargs):
    """
    Set the preferred mesh size for geometrical points as specified by
    gmsh.
    Parameters:
        pts (float array): The points which will be passed to Gmsh. Array size
            2 x n_pts.
        pts_on_boundary (logical array): Indicates which (True) of the pts are
            constitute the domain boundary (corners). Only relevant if
            mesh_size_bound is defined as a kw (see below).
        lines (integer array): Definition and tags of the boundary and fracture
            lines. Size 4 x n_points, two first are pointers to pts and two
            last are line tags.
    The mesh size is determined through the three parameters (all passed
    as kwargs):
        mesh_size_frac: Ideal mesh size. Will be added to all points that are
            sufficiently far away from other points.
        mesh_size_min: Minimal mesh size; we will make no attempts to enforce
            even smaller mesh sizes upon Gmsh.
        mesh_size_bound: Boundary mesh size. Will be added to the points
            defining the boundary. If included, pts_on_boundary is mandatory.

    See the gmsh manual for further details.

    """
    num_pts = pts.shape[1]
    val = kwargs.get("mesh_size_frac", 1)
    val_bound = kwargs.get("mesh_size_bound", None)
    val_min = kwargs.get("mesh_size_min", None)
    tol = kwargs.get("tol", 1e-5)
    # One value for each point to distinguish betwee val and val_bound.
    vals = val * np.ones(num_pts)
    if val_bound is not None:
        vals[pts_on_boundary] = val_bound
    if val_min is None:
        val_min = 1e-8 * val
    # Compute the lenght of each pair of points (fractures + domain boundary)
    pts_id = lines[:2, :]
    dist = np.linalg.norm(pts[:, pts_id[0, :]] - pts[:, pts_id[1, :]], axis=0)
    dist_pts = np.tile(np.inf, pts.shape[1])

    # Loop on all the points and consider the minimum between the pairs of
    # points lengths associated to the single point and the value input by the
    # user
    for i, pt_id in enumerate(pts_id.T):
        distances = np.array([dist_pts[pt_id], [dist[i]] * 2, vals[pt_id]])
        dist_pts[pt_id] = np.amin(distances, axis=0)

    num_pts = pts.shape[1]
    pts_extra = np.empty((pts.shape[0], 0))
    dist_extra = np.empty(0)
    pts_id_extra = np.empty(0, dtype=np.int)
    vals_extra = np.empty(0)
    # For each point we compute the distance between the point and the other
    # pairs of points. We keep the minimum distance between the previously
    # computed point distance and the distance among the other pairs of points.
    # If the latter happens, we introduce a new point (useful to determine the
    # grid size) on the corresponding pair of points with a corresponding
    # distance.

    # Loop over all lines
    for line in lines.T:
        # Find start and endpoint of this line
        start, end = pts[:, line[0]], pts[:, line[1]]
        # Compute the distacne between this line and all points in the domain
        # This will also include the start and endpoint of this line, but we
        # deal with that later
        dist, cp = pp.distances.points_segments(pts, start, end)

        # We know there is a single segment, thus the distance vector can be reduced
        dist = dist[:, 0]
        # Find points that are closer than the target distance
        hit = np.where(
            np.logical_and(dist < vals, np.logical_not(np.isclose(dist, 0.0)))
        )[0]
        # Update the minimum distance found for these points
        dist_pts[hit] = np.minimum(dist_pts[hit], dist[hit])

        # cp[hit, 0] now contains points on the main line that are closest to
        # another point, and that sufficiently close to warrant attention.
        # Compute the distance from cp to the start and endpoint of this
        # line
        # Transposes are needed here because that is how numpy works
        dist_start = pp.distances.point_pointset(start, cp[hit, 0].T)
        dist_end = pp.distances.point_pointset(end, cp[hit, 0].T)

        # Now, the cp points are added if they are closer to another point than
        # to the start and end point of its line, and if the distance from the
        # start and end is not smaller than the minimum point. The latter removes
        # lines having their own end points added, and also avoids arbitrarily
        # small segments along the line.
        to_add = np.logical_and.reduce((dist[hit] < dist_start, dist[hit] < dist_end))
        if np.any(to_add):
            dist_extra = np.r_[
                dist_extra, np.minimum(dist[hit[to_add]], vals[hit[to_add]])
            ]
            pts_extra = np.c_[pts_extra, cp[hit[to_add], 0].T]
            pts_id_extra = np.r_[
                pts_id_extra, line[3] * np.ones(to_add.sum(), dtype=np.int)
            ]
            vals_extra = np.r_[vals_extra, vals[hit[to_add]]]

    old_lines = lines
    old_pts = pts

    # Since the computation was done point by point with the lines, we need to
    # consider all the new points together and remove (from the new points) the
    # useless ones.
    extra_ids, inv_index = np.unique(pts_id_extra, return_inverse=True)
    to_remove = np.empty(0, dtype=np.int)
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
                    dist_matrix[
                        pts_id_loc[pos_min], pts_id_loc[pos_max]
                    ] = np.linalg.norm(pt1 - pt2)

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
    new_lines = np.empty((4, 0), dtype=np.int)
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

            # We will sort points on the line below, but this function requires
            # 3D points
            pts_frac_aug = np.vstack((pts_frac, np.zeros(pts_frac.shape[1])))
            pts_frac_id = pts_frac_id[
                pp.map_geometry.sort_points_on_line(pts_frac_aug, tol)
            ]
            pts_frac_id = np.vstack((pts_frac_id[:-1], pts_frac_id[1:]))
            other_info = np.tile(
                lines[2:, mask_bool][:, 0], (pts_frac_id.shape[1], 1)
            ).T
            new_lines = np.c_[new_lines, np.vstack((pts_frac_id, other_info))]

    # Consider extra points related to the input value, if the fracture is long
    # and, beacuse of val, needs additional points we increase the number of
    # lines.
    relax = kwargs.get("relaxation", 0.8)
    lines = np.empty((4, 0), dtype=np.int)
    for seg in new_lines.T:
        mesh_size_pt1 = dist_pts[seg[0]]
        mesh_size_pt2 = dist_pts[seg[1]]
        dist = np.linalg.norm(pts[:, seg[0]] - pts[:, seg[1]])
        if (
            mesh_size_pt1 >= relax * vals[seg[0]]
            and mesh_size_pt2 >= relax * vals[seg[1]]
        ) or (relax * dist <= 2 * mesh_size_pt1 and relax * dist <= 2 * mesh_size_pt2):
            lines = np.c_[lines, seg]
        else:
            pt_id = pts.shape[1]
            new_pt = 0.5 * (pts[:, seg[0]] + pts[:, seg[1]])
            pts = np.c_[pts, new_pt]

            mesh_size = np.amin(np.r_[vals[seg[:2]], dist / 2.0])

            start_old = old_pts[:, old_lines[0]]
            end_old = old_pts[:, old_lines[1]]

            # Compute the ditsacne between the current point and all old lines
            dist1, cp = pp.distances.points_segments(new_pt, start_old, end_old)
            # Disregard points that lie on the old segment by assigning a value so high
            # that it will not be picked up by the minimum.
            dist1[np.isclose(dist1, 0.0)] = mesh_size * 10
            # Update the minimum mesh size if any dist1 is less than the current value
            mesh_size = np.minimum(mesh_size, dist1.min())

            dist_pts = np.r_[dist_pts, mesh_size]
            vals = np.r_[vals, mesh_size]
            lines = np.c_[
                lines, [seg[0], pt_id, seg[2], seg[3]], [pt_id, seg[1], seg[2], seg[3]]
            ]

    return dist_pts, pts, lines


def obtain_interdim_mappings(lg, fn, n_per_face):
    """
    Find mappings between faces in higher dimension and cells in the lower
    dimension

    Parameters:
        lg (pp.Grid): Lower dimensional grid.
        fn (pp.Grid): Higher dimensional face-node relation.
        n_per_face (int): Number of nodes per face in the higher-dimensional grid.

    Returns:
        np.array: Index of faces in the higher-dimensional grid that correspond
            to a cell in the lower-dimensional grid.
        np.array: Index of the corresponding cells in the lower-dimensional grid.

    """
    if lg.dim > 0:
        cn_loc = lg.cell_nodes().indices.reshape((n_per_face, lg.num_cells), order="F")
        cn = lg.global_point_ind[cn_loc]
        cn = np.sort(cn, axis=0)
    else:
        cn = np.array([lg.global_point_ind])
        # We also know that the higher-dimensional grid has faces
        # of a single node. This sometimes fails, so enforce it.
        if cn.ndim == 1:
            fn = fn.ravel()
    is_mem, cell_2_face = pp.utils.setmembership.ismember_rows(
        cn.astype(np.int32), fn.astype(np.int32), sort=False
    )
    # An element in cell_2_face gives, for all cells in the
    # lower-dimensional grid, the index of the corresponding face
    # in the higher-dimensional structure.
    if not (np.all(is_mem) or np.all(~is_mem)):
        warnings.warn(
            """Found inconsistency between cells and higher
                      dimensional faces. Continuing, fingers crossed"""
        )
    low_dim_cell = np.where(is_mem)[0]
    return cell_2_face, low_dim_cell
