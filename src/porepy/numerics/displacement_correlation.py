#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimation of stress intensity factors using the displacement correlation
method, see e.g.
        Nejati et al.
        On the use of quarter-point tetrahedral finite elements in linear
        elastic fracture mechanics
        Engineering Fracture Mechanics 144 (2015) 194–221

At present, some unnecessary quantities are computed (and passed around). This
is (at least partly) for purposes of investigation of the method.
"""

import numpy as np
import porepy as pp


# ---------------------propagation criteria----------------------------------#

def faces_to_open(gb, u, critical_sifs, **kw):
    """
    Determine where a fracture should propagate based on the displacement
    solution u using the displacement correlation technique.
    Parameters:
        gb  - grid bucket. For now, contains one higher-dimensional (2D or 3D)
            grid and one lower-dimensional fracture, to be referred to as g_h
            and g_l, respectively. Note that the data corresponding to d_h
            should contain the Young's modulus and Poisson's ratio, both
            assumed (for now) to be constant.
        u (array): solution as computed by FracturedMpsa. One displacement
            vector for each cell center in g_h and one for each of the fracture
            faces. The ordering for a 2D g_h with four cells and four fracture
            faces (two on each side of two g_l fracture cells) is

            [u(c0), v(c0), u(c1), v(c1), u(c2), v(c2), u(c3), v(c3),
             u(f0), v(f0), u(f1), v(f1), u(f2), v(f2), u(f3), v(f3)]

            Here, f0 and f1 are the "left" faces, the original faces before
            splitting of the grid, found in g_h.frac_pairs[0]. f2 and f3 are
            the "right" faces, the faces added by the splitting, found in
            g_h.frac_pairs[1].
        critical_sifs (array): the stress intensity factors at which the
            fracture yields, one per mode (i.e., we assume this rock parameter
            to be homogeneous for now)
        kw: optional keyword arguments, to be explored. For now:
            rm (float): oprimal distance from tip faces to correlation points.
                If not provided, an educated guess is made by estimate_rm().
            use_normal_rm_distance (bool) - if True, the distance from
                correlation point to tip face is used instead of the distance
                between the correlation point and the tip face centroid. These
                may differ for 2d fractures.

    Returns:
        faces_h_to_open: (possibly empty) array of higher-dimensional faces
            to be split
        sifs: the calculated stress intensity factors for each of the
            lower-dimensional tip faces.

    """
    dim_h = gb.dim_max()
    dim_l = gb.dim_min()
    g_h = gb.grids_of_dimension(dim_h)[0]
    g_l = gb.grids_of_dimension(dim_l)[0]
    d_h = gb.node_props(g_h)
    rm = kw.get('rm', estimate_rm(g_l, **kw))
    E = d_h['Young']
    poisson = d_h['Poisson']
    mu = E / (2 * (1 + poisson))
    kappa = 3 - 4 * poisson
    face_cells = gb.edge_props((g_h, g_l), 'face_cells')

    # Obtain the g_h.dim components of the relative displacement vector
    # corresponding to each fracture tip face of the lower dimension.
    cells_l, faces_l, rm_vectors, rm_distances, normal_rm \
        = identify_correlation_points(g_h, g_l, rm, u, face_cells)

    delta_u = relative_displacements(u, face_cells, g_l, cells_l, faces_l, g_h)

    if kw.get('use_normal_rm_distance', False) and g_l.dim > 1:
        rm_distances = normal_rm

    # Use them to compute the SIFs locally.
    sifs = sif_from_delta_u(delta_u, rm_distances, mu, kappa)

    tips_to_propagate = determine_onset(sifs,
                                        np.array(critical_sifs))
    # Find the right direction. For now, only normal extension is allowed.
    faces_h_to_open = identify_faces_to_open(g_h, g_l, tips_to_propagate,
                                             rm_vectors)
    faces_h_to_open = np.unique(faces_h_to_open)
    return faces_h_to_open, sifs


def identify_faces_to_open(g_h, g_l, tips_to_propagate, rm_vectors):
    """
    Identify the faces to open. For now, just pick out the face lying
    immediately outside the existing fracture tip faces which we wish to
    propagate (as close to the fracture plane as possible).
    TODO: Include angle computation.

    Parameters:
        g_h: higher-dimensional grid
        g_l: lower-dimensional grid
        tips_to_propagate (boolean array): Whether or not to propagate the
            fracture at each of the tip faces.
        rm_vectors (array): Vectors pointing away from the fracture tip,
            lying in the fracture plane.

    Returns:
        faces_h (array): The higher-dimensional faces which should be opened.
    """
    faces_l = g_l.tags['tip_faces'].nonzero()[0]
    # Construct points lying just outside the fracture tips
    extended_points = g_l.face_centers[:, faces_l] + rm_vectors / 100
    # Find closest higher-dimensional face.
    faces_h = []
    for i in range(extended_points.shape[1]):
        if tips_to_propagate[i]:
            p = extended_points[:, i]
            distances = pp.cg.dist_point_pointset(p, g_h.face_centers)
            faces_h.append(np.argmin(distances))

    return np.array(faces_h, dtype=int)


def determine_onset(sifs, critical_values):
    """
    For the time being, very crude criterion: K_I > K_I,cricial.
    TODO: Extend to equivalent SIF, taking all three modes into account.

    Parameters:
        sifs (array): stress intensity factors.
        critical_values (array): critical SIF values to which the above are
        compared.

    Returns:
        exceed_critical (array): bool indicating which sifs meet the
            criterion.
    """
    exceed_critical = np.absolute(sifs[0]) > critical_values[0]
    return exceed_critical


def sif_from_delta_u(d_u, rm, mu, kappa):
    """
    Compute the stress intensity factors from the relative displacements

    Parameters:
        d_u (array): relative displacements, g_h.dim x n.
        rm (array): distance from correlation point to fracture tip.

    Returns:
        K (array): the displacement correlation stress intensity factor
        estimates.
    """
    (dim, n_points) = d_u.shape
    K = np.zeros(d_u.shape)

    rm = rm.T
    K[0] = np.sqrt(2 * np.pi / rm) * np.divide(mu, kappa + 1) * d_u[1, :]
    K[1] = np.sqrt(2 * np.pi / rm) * np.divide(mu, kappa + 1) * d_u[0, :]
    if dim == 3:
        K[2] = np.sqrt(2 * np.pi / rm) * np.divide(mu, 4) * d_u[2, :]
    return K


def identify_correlation_points(g_h, g_l, rm, u, face_cells):
    """
    Get the relative displacement for displacement correlation SIF computation.
    For each tip face of the fracture, the displacements are evaluated on the
    two fracture walls, on the (higher-dimensional) face midpoints closest to
    a point p. p is defined as the point lying a distance rm away from the
    (lower-dimensional) face midpoint in the direction normal to the fracture
    boundary.
    TODO: Account for sign (in local coordinates), to ensure positive relative
    displacement for fracture opening.

    Parameters:
        g_h: higher-dimensional grid
        g_l: lower-dimensional grid
        rm (float): optimal distance from fracture front to correlation point.
        u (array): displacement solution.
        face_cells (array): face_cells of the grid pair.

    Returns:
        cells_l (array): Lower-dimensional cells containing the correlation
            point (as its cell center).
        faces_l (array): The tip faces, for which the SIFs are to be estimated.
        rm_vectors (array): Vector between centers of cells_l and faces_l.
        actual_rm (array): Length of the above.
        normal_rm (array): Distance between the cell center and the fracture
            front (defined by the face of the fracture tip). Will differ from
            actual_rm if rm_vectors are non-orthogonal to the tip faces (in
            2d fractures).
    """
    # Find the g_l cell center which is closest to the optimal correlation
    # point for each fracture tip
    faces_l = g_l.tags['tip_faces'].nonzero()[0]
    n_tips = faces_l.size
    ind = g_l.cell_faces[faces_l].nonzero()[1]
    normals_l = np.divide(g_l.face_normals[:, faces_l],
                          g_l.face_areas[faces_l])
    # Get the sign right (outwards normals)
    normals_l = np.multiply(normals_l, g_l.cell_faces[faces_l, ind])
    rm_vectors = np.multiply(normals_l, rm)
    optimal_points = g_l.face_centers[:, faces_l] - rm_vectors

    cells_l = []
    actual_rm = []
    normal_rm = []
    for i in range(n_tips):
        p = optimal_points[:, i]
        distances = pp.cg.dist_point_pointset(p, g_l.cell_centers)
        cell_ind = np.argmin(distances)
        dist = pp.cg.dist_point_pointset(g_l.face_centers[:, faces_l[i]],
                                         g_l.cell_centers[:, cell_ind])

        cells_l.append(cell_ind)
        actual_rm.append(dist)
        if g_l.dim > 1:
            nodes = g_l.face_nodes[:, faces_l[i]].nonzero()[0]
            p0 = g_l.nodes[:, nodes[0]].reshape((3, 1))
            p1 = g_l.nodes[:, nodes[1]].reshape((3, 1))
            normal_dist, _ = pp.cg.dist_points_segments(
                    g_l.cell_centers[:, cell_ind], p0, p1)
            normal_rm.append(normal_dist[0])

    actual_rm = np.array(actual_rm)
    normal_rm = np.array(normal_rm)
    return cells_l, faces_l, rm_vectors, actual_rm, normal_rm


def relative_displacements(u, face_cells, g_l, cells_l, faces_l, g_h):
    """
    Compute the relative displacements between the higher-dimensional faces
    on either side of each correlation point.

    Parameters:
    u (array): displacements on the higher-dimensional grid, as computed by
        FracturedMpsa, g_h.dim x (g_h.num_cells + g_l.num_cells * 2), see e.g.
        displacement_correlation for description.
    face_cells (array): the face_cells connectivity matrix corresponding to g_l
    and g_h.
    g_l and g_h: higher- and lower-dimensional grid.
    cells_l (array): the lower-dimensional cells containing the correlation
        points as their cell centers.
    faces_l (array): the tip faces of the lower dimension, where propagation
        may occur.

    Returns:
        delta_us (array): the relative displacements, g_h.dim x n_tips.
    """
    # Find the two faces used for relative displacement calculations for each
    # of the tip faces
    n_tips = faces_l.size
    faces_h = face_cells[cells_l].nonzero()[1]
    faces_h = faces_h.reshape((2, n_tips), order='F')
    # Extract the displacement differences between pairs of higher-dimensional
    # fracture faces
    i_l = np.arange(g_h.num_cells * g_h.dim,
                    (g_h.num_cells + g_l.num_cells) * g_h.dim)
    i_r = np.arange((g_h.num_cells + g_l.num_cells) * g_h.dim, u.size)
    du_faces = np.reshape(u[i_l] - u[i_r], (g_h.dim, -1), order='F')

    delta_us = np.empty((g_h.dim, 0))
    # For each face center pair, rotate to local coordinate system aligned with
    # x along the normals, y perpendicular to the fracture and z along the
    # fracture tip. Doing this for each point avoids problems with non-planar
    # fractures.
    cell_nodes = g_l.cell_nodes()
    for i in range(n_tips):
        face_l = faces_l[i]
        cell_l = g_l.cell_faces[face_l].nonzero()[1]
        face_pair = faces_h[:, i]
        nodes = cell_nodes[:, cell_l].nonzero()[0]
        pts = g_l.nodes[:, nodes]
        pts -= g_l.cell_centers[:, cell_l].reshape((3, 1))
        normal_h_1 = g_h.face_normals[:, face_pair[1]] \
            * g_h.cell_faces[face_pair[1]].data
        if g_h.dim == 3:
            # Rotate to xz, i.e., normal alignes with y. Pass normal of the
            # second face to ensure that the first one is on top in the local
            # coordinate system

            R1 = pp.cg.project_plane_matrix(pts, normal=normal_h_1,
                                            reference=[0, 1, 0])
            # Rotate so that tangent aligns with z-coordinate
            nodes_l = g_l.face_nodes[:, faces_l[i]].nonzero()[0]
            translated_pts = g_l.nodes[:, nodes_l] \
                - g_l.face_centers[:, face_l].reshape((3,1))
            pts = np.dot(R1, translated_pts)
        else:
            R1 = np.eye(3)

        normal_r = g_l.cell_faces[face_l].data * g_l.face_normals[:, face_l]
        normal_r = np.dot(R1, normal_r) / np.linalg.norm(normal_r)
        # Rotate so that the face normal, on whose line pts lie, aligns
        # with x
        if np.all(np.isclose(normal_r, np.array([1, 0, 0]))):
            R2 = np.eye(3)
        elif np.all(np.isclose(normal_r, np.array([-1, 0, 0]))):
            R2 = - np.eye(3)
            R2[3 - g_l.dim, 3 - g_l.dim] = 1
        else:
            R2 = pp.cg.project_line_matrix(pts, tangent=normal_r,
                                           reference=[1, 0, 0])
        R = np.dot(R2, R1)
        normal_h_1_r = np.dot(R, normal_h_1) / np.linalg.norm(normal_h_1)

        h_1_sign = normal_h_1_r[1]
        assert np.all(np.isclose(h_1_sign * normal_h_1_r, np.array([0, 1, 0])))
        # Find what frac-pair the tip i corresponds to
        j = pp.utils.setmembership.ismember_rows(face_pair[:, np.newaxis],
                                                 g_h.frac_pairs)[1]
        d_u = np.dot(R, np.append(du_faces[:, j],
                                  np.zeros((1, 3 - g_h.dim))))[:g_h.dim]
        # Normal_h_1 points outward from the right cell in g_h.frac_pairs. If
        # it now points downwards, that cell is on the upper side in the
        # rotated coordinate system. Thne, d_u should be u_right - u_left,
        # rather than the opposite (which it is by default).
        d_u *= h_1_sign
        delta_us = np.append(delta_us, d_u[:, np.newaxis], axis=1)
    return delta_us


def estimate_rm(g, **kw):
    """
    Estimate the optimal distance between tip face centers and correlation
    points. Based on the findings in Nejati et al. (see
    displacement_correlation), where a optimum is found related to local cell
    size.

    Parameters:
        g  - fracture grid.

    Returns:
        rm  - distance estimate.
    """
    # Constant, see Nejati et al.
    k = kw.get('rm_factor', .8)
    faces = g.tags['tip_faces'].nonzero()[0]
    if g.dim == 2:
        rm = k * g.face_areas[faces]
    else:
        # Dimension is 1, and face_area is 1. Use cell volume of neighbouring
        # cell instead.
        cells = g.cell_faces[faces].nonzero()[1]
        rm = k * g.cell_volumes[cells]
    return rm
