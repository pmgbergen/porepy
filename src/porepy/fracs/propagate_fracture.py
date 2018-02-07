#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 17:32:35 2018

@author: ivar

Propagation of fractures. Much in common with (and reuse of) split_grid.
For now assumes:
    single fracture
When this assumption is relieved, some (re)structuring will be needed.
The structure for multi-fracture propagation may possibly strongly resemble
that of split_grid.
"""
import warnings
import numpy as np
import scipy.sparse as sps
from porepy.fracs import split_grid
from porepy.utils import setmembership, tags, sort_points, sparse_mat, comp_geom
from porepy.numerics.mixed_dim.coupler import Coupler
from porepy.numerics.mixed_dim.solver import SolverMixedDim
from porepy.numerics.fv import tpfa, mpfa, fvutils
from porepy.params import bc, tensor


def propagate_fracture(gb, g_h, g_l, faces_h):
    """
    Extend the fracture defined by g_l to the higher-dimensional faces_h and
    splits g_h along the same faces.
    Changes to grids done in-place.
    The call changes:
        Geometry and connectivity fields of the two grids involved.
        The face_cells mapping between them
        Their respective face tags.
    Also adds the following to node data dictionaries:
        new_cells and new_faces tags, for use in e.g. local discretization
        updates.
        partial_update, a boolean flag indicating that the grids have been
        updated.
    These may have to be returned by the function to be handled externally when
    multiple fractures are implemented.
    """

    if faces_h.size == 0:
        for g, d in gb:
            d['partial_update'] = True
            if g.dim == gb.dim_max():
                d['new_cells'] = np.empty(0, dtype=int)
                d['new_faces'] = np.empty(0, dtype=int)
            else:
                d['new_cells'] = np.empty(0, dtype=int)
                d['new_faces'] = np.empty(0, dtype=int)
        return

    # Keep track of original information:
    n_old_faces_l = g_l.num_faces
    n_old_cells_l = g_l.num_cells
    n_old_nodes_l = g_l.num_nodes

    # It is convenient to tag the nodes lying on the domain boundary. This
    # helps updating the face tags later:
    tags.add_node_tags_from_face_tags(gb, 'domain_boundary')

    # Get the "involved nodes", i.e., the union between the new nodes in the
    # lower dimension and the boundary nodes where the fracture propagates.
    # The former are added to the global_point_ind of g_l.
    unique_node_ind_l, unique_node_ind_h = update_nodes(g_h, g_l, faces_h)

    # Update the connectivity matrices (cell_faces and face_nodes) and tag
    # the lower-dimensional faces, including re-classification of (former) tips
    # to internal faces, where appropriate.
    n_new_faces, new_face_centers \
        = update_connectivity(g_l, g_h, n_old_nodes_l, unique_node_ind_l,
                              n_old_faces_l, n_old_cells_l, unique_node_ind_h,
                              faces_h)
    # Add new faces to g_l
    append_face_geometry(g_l, n_new_faces, new_face_centers)
    # Same for cells
    new_cells = update_cells(g_h, g_l, faces_h)

    # Split g_h along faces_h
    split_fracture_extension(gb, g_h, g_l, faces_h, unique_node_ind_h,
                             new_cells)

    # Store information on which faces and cells have just been added. Note,
    # we only keep track of the faces and cells from the last propagation call!
    new_faces_h = g_h.frac_pairs[1, np.isin(g_h.frac_pairs[0], faces_h)]
    for g, d in gb:
        d['partial_update'] = True
        if g.dim == gb.dim_max():
            d['new_cells'] = np.empty(0)
            d['new_faces'] = new_faces_h
        else:
            d['new_cells'] = new_cells
            d['new_faces'] = np.arange(g.num_faces - n_new_faces, g.num_faces)
    gb.compute_geometry()


def update_connectivity(g_l, g_h, n_old_nodes_l, unique_node_indices_l,
                        n_old_faces_l, n_old_cells_l, unique_nodes_h, faces_h):
    """
    Update of cell_faces of the lower grid after insertion of new cells at the
    higher-dimensional faces_h. Also tags the faces as domain_boundary or tip
    Should be called after initialization of tags
    and geometry of g_l by append_face_geometry and append_face_tags.
    """
    # Extract immediate information
    n_new_cells_l = faces_h.size
    new_cells_l = np.arange(n_old_cells_l, n_old_cells_l + n_new_cells_l)
    # Initialize
    new_faces_l = np.empty((g_l.dim, 0))
    new_face_centers_l = np.empty((3, 0))
    face_counter_l = n_old_faces_l
    # Copy what is to be updated
    old_cell_faces = g_l.cell_faces.copy()
    old_face_nodes = g_l.face_nodes.copy()
    # Get the face_node indices to form lower-dimensional faces on the form
    # [[nodes of face 1], [nodes of face 2], ...], i.e., array where each face
    # is represented by the nodes it consists of.
    all_faces_l = np.reshape(g_l.face_nodes.indices,
                             (n_old_faces_l, g_l.dim)).T

    # Initialize indices and values for the cell_faces update
    (ind_f, ind_c, cf_val) = (np.empty(0), np.empty(0), np.empty(0))
    # and for the face_nodes update
    (fn_ind_f, fn_ind_n) = (np.empty(0), np.empty(0))

    for i, c in enumerate(new_cells_l):
        # Find the nodes involved
        face_h = faces_h[i]
        local_nodes_h = g_h.face_nodes[:, face_h].nonzero()[0]
        in_unique_nodes = order_preserving_find(local_nodes_h, unique_nodes_h)
        local_nodes_l = np.array(unique_node_indices_l[in_unique_nodes],
                                 dtype=int)
        # Get geometry information
        local_pts = g_l.nodes[:, local_nodes_l]
        local_cell_center = np.mean(local_pts, axis=1)
        # Store face center for the update of g_l.face_centers

        # Faces are defined by one node in 1d and two in 2d. This requires
        # dimension dependent treatment:
        if g_l.dim == 2:
            # Sort nodes clockwise (!)
            map_to_sorted = sort_points.sort_point_plane(local_pts,
                                                         local_cell_center)
            sorted_nodes_l = local_nodes_l[map_to_sorted]
            sorted_nodes_h = local_nodes_h[map_to_sorted]
            # Define the faces of c (size: 2 x faces_per_cell_l). "Duplicate"
            # of the higher dimension used for tag identification.
            faces_l = np.vstack((sorted_nodes_l,
                                 np.append(sorted_nodes_l[1:],
                                           sorted_nodes_l[0])))
            local_faces_h = np.vstack((sorted_nodes_h,
                                       np.append(sorted_nodes_h[1:],
                                                 sorted_nodes_h[0])))
        else:
            # Faces and nodes are 1:1, but ismember_rows requires 2d array
            faces_l = np.atleast_2d(local_nodes_l)
            local_faces_h = np.atleast_2d(local_nodes_h)

        # Now the faces_per_cell_l faces of c are defined by sorted_nodes_l
        # and their arrangement in faces_l.
        n_local_faces_l = faces_l.shape[-1]

        # Check which faces exist in g_l already, either from before propgation
        # or from previous runs through current loop:
        (exist, existing_faces_l) = setmembership.ismember_rows(faces_l,
                                                                all_faces_l)
        # The existing faces are no longer tips (but internal). The new faces
        # are tips (or on the domain boundary).
        g_l.tags['tip_faces'][existing_faces_l] = False
        n_new_local_faces_l = np.sum(~exist)

        new_face_indices_l = np.arange(face_counter_l,
                                       face_counter_l
                                       + n_new_local_faces_l)
        face_counter_l += n_new_local_faces_l
        append_face_tags(g_l, n_new_local_faces_l)
        # The existing faces are tagged according to the information from the
        # node tags of g_h.
        fi = local_faces_h[:, ~exist]
        domain_boundary_faces \
            = np.all(g_h.tags['domain_boundary_nodes'][fi], axis=0)
        g_l.tags['tip_faces'][new_face_indices_l] = ~domain_boundary_faces
        g_l.tags['domain_boundary_faces'][new_face_indices_l] \
            = domain_boundary_faces
        # Add the new faces
        all_faces_l = np.append(all_faces_l, faces_l[:, ~exist], axis=1)
        # Find indices of face_nodes to be updated.
        ind_n_local = faces_l[:, ~exist]
        local_pts = g_l.nodes[:, ind_n_local]
        local_face_centers = np.mean(local_pts, axis=1)
        new_face_centers_l = np.append(new_face_centers_l,
                                       np.atleast_2d(local_face_centers),
                                       axis=1)
        new_faces_l = np.append(new_faces_l, ind_n_local, axis=1)
        all_local_faces = np.empty(faces_l.shape[-1])
        all_local_faces[exist] = existing_faces_l
        all_local_faces[~exist] = new_face_indices_l
        ind_f_local = np.tile(all_local_faces, g_l.dim)
        fn_ind_f = np.append(fn_ind_f, ind_f_local)
        fn_ind_n = np.append(fn_ind_n, faces_l)
        # Same for cell_faces:
        ind_f = np.append(ind_f, all_local_faces)
        ind_c = np.append(ind_c, c*np.ones(n_local_faces_l))
        # and find the sign:
        cf_val_loc = np.zeros(n_local_faces_l)
        cf_val_loc[~exist] = 1
        are_in_original = existing_faces_l < n_old_faces_l
        ind_in_original = existing_faces_l[are_in_original]
        ind_not_in_original = existing_faces_l[~are_in_original]
        ind_local = np.in1d(all_local_faces, ind_in_original)
        ind_not_local = np.in1d(all_local_faces, ind_not_in_original)
        cf_val_loc[ind_local] = - g_l.cell_faces[ind_in_original, :].data
        cf_val_loc[ind_not_local] = -1

        cf_val = np.append(cf_val, cf_val_loc)

    # Resize and update face_nodes ...
    g_l.face_nodes = sps.csc_matrix((g_l.num_nodes, face_counter_l), dtype=bool)
    g_l.face_nodes[:n_old_nodes_l, :n_old_faces_l] = old_face_nodes
    g_l.face_nodes[fn_ind_n, fn_ind_f] = True
    g_l.face_nodes.eliminate_zeros()
    # ... and cell_faces

    g_l.cell_faces = sps.csc_matrix((face_counter_l,
                                    n_old_cells_l + n_new_cells_l))
    g_l.cell_faces[0:n_old_faces_l, 0:n_old_cells_l] = old_cell_faces
    g_l.cell_faces[ind_f, ind_c] = cf_val
    g_l.cell_faces.eliminate_zeros()
    n_new_faces = face_counter_l - n_old_faces_l
    return n_new_faces, new_face_centers_l


def order_preserving_find(a, b):
    """
    Finds the occurences of a in b (both np arrays) and returns stripped of
    None (elements of b that are not found).
    """
    c_with_none = np.asarray(setmembership._find_occ(a, b))
    not_none_ind = np.logical_not(np.isnan(c_with_none.astype(float)))
    return np.array(c_with_none[not_none_ind], dtype=int)


def update_cells(g_h, g_l, faces_h):
    """
    Cell information for g_l is inherited directly from the higher-dimensional
    faces we are splitting. The function updates num_cells, cell_centers and
    cell_volumes.
    """
    n_new_cells = g_l.num_cells + faces_h.size
    new_cells = np.arange(g_l.num_cells, n_new_cells)
    g_l.num_cells = n_new_cells
    g_l.cell_centers = np.append(g_l.cell_centers,
                                g_h.face_centers[:, faces_h], axis=1)
    g_l.cell_volumes = np.append(g_l.cell_volumes, g_h.face_areas[faces_h])
    return new_cells


def update_nodes(g_h, g_l, faces_h):
    """
    Finds the nodes in the lower-dimensional grid corresponding to the higher-
    dimensional faces to be split. Updates node information in g_l:
        global_point_ind
        nodes
        num_nodes
    Returns:
        unique_nodes_l - numpy array (number of involved nodes x 1) Indices of
            the nodes (as arranged in g_l.nodes).
        unique_nodes_h - same, but corresponding to g_h.nodes.
    """
    nodes_h = g_h.face_nodes[:, faces_h].nonzero()[0]
    unique_nodes_h = np.unique(nodes_h)
    unique_global_nodes = g_h.global_point_ind[unique_nodes_h]

    are_old_global_nodes_in_l = np.in1d(unique_global_nodes,
                                        g_l.global_point_ind)
    are_new_global_nodes_in_l = np.logical_not(are_old_global_nodes_in_l)
    new_node_indices_h = unique_nodes_h[are_new_global_nodes_in_l]
    new_global_node_indices_l = unique_global_nodes[are_new_global_nodes_in_l]
    g_l.global_point_ind = np.append(g_l.global_point_ind,
                                    new_global_node_indices_l)
    new_nodes_l = g_h.nodes[:, new_node_indices_h].copy()

    # Order preserving find:
    unique_nodes_l = order_preserving_find(unique_global_nodes,
                                           g_l.global_point_ind)

    g_l.nodes = np.append(g_l.nodes, new_nodes_l, axis=1)
    g_l.num_nodes += new_nodes_l.shape[1]
    return unique_nodes_l, unique_nodes_h


def append_face_geometry(g, n_new_faces, new_centers):
    """
    Appends and updates faces geometry information for new faces. Also updates
    num_faces.
    """
    g.face_normals = np.append(g.face_normals,
                               np.zeros((3, n_new_faces)), axis=1)
    g.face_areas = np.append(g.face_areas, np.ones(n_new_faces))
    g.face_centers = np.append(g.face_centers,
                               new_centers, axis=1)
    g.num_faces += n_new_faces


def append_face_tags(g, n_new_faces):
    """
    Initiates default face tags (False) for new faces.
    """
    keys = tags.standard_face_tags()
    new_tags = [np.zeros(n_new_faces, dtype=bool) for _ in range(len(keys))]
    tags.append_tags(g.tags, keys, new_tags)


def split_fracture_extension(bucket, g_h, g_l, faces_h, nodes_h, cells_l):
    """
    Split the higher-dimensional grid along specified faces. Updates made to
    face_cells of the grid pair and the nodes and faces of the higher-
    dimensional grid.
    Parameters
    ----------
    bucket      - A grid bucket
    g_h          - Higher-dimensional grid to be split along specified faces.
    g_l          - Immersed lower-dimensional grid.
    faces_h     - The higher-dimensional faces to be split.
    cells_l     - The corresponding lower-dimensional cells.
    nodes_h     - The corresponding (hig_her-dimensional) nodes.
    """

    face_cells = bucket.edge_prop((g_h, g_l), 'face_cells')

    # We split all the faces that are connected to faces_h
    # The new faces will share the same nodes and properties (normals,
    # etc.)

    face_cells = split_grid.split_certain_faces(g_h, face_cells, faces_h,
                                                cells_l)
    bucket.add_edge_prop('face_cells', [(g_h, g_l)], face_cells)

    # We now find which lower-dim nodes correspond to which higher-
    # dim nodes. We split these nodes according to the topology of
    # the connected higher-dim cells. At a X-intersection we split
    # the node into four, while at the fracture boundary it is not split.

    split_grid.split_nodes(g_h, [g_l], [nodes_h])

    # Remove zeros from cell_faces

    [g.cell_faces.eliminate_zeros() for g, _ in bucket]
    return bucket


def tag_affected_cells_and_faces(gb):
    """
    Tag the lower-dimensional cells and higher-dimensional faces which have
    been affected by the update. Should be the new cells, and both the original
    (defining the split, see e.g. faces_h in propagate_fracture) and the newly
    created faces.
    Assumes only two dimensions.
    """
    dh = gb.dim_max()
    dl = gb.dim_min()
    g_h = gb.grids_of_dimension(dh)[0]
    g_l = gb.grids_of_dimension(dl)[0]
    dh = gb.node_props(g_h)
    dl = gb.node_props(g_l)
    cells_l = dl['new_cells']
    faces_h = dh['new_faces']
    old_faces_h = g_h.frac_pairs[0, np.isin(g_h.frac_pairs[1], faces_h)]
    all_faces_h = np.concatenate((old_faces_h, faces_h))
    t = np.zeros(g_h.num_faces, dtype=bool)
    t[all_faces_h] = True
    g_h.tags['discretize_faces'] = t

    # TODO: Fix tpfa, so that local 1d update is possible (MPFA calls TPFA for
    # 1d). Once fixed, change to t = np.zeros(...) in this line:
    t = np.ones(g_l.num_cells, dtype=bool)
    t[cells_l] = True
    g_l.tags['discretize_cells'] = t


#----------------------propagation criteria----------------------------------#

def displacement_correlation(gb, u, critical_sifs, **kw):
    """
    Determine where a fracture should propagate based on the displacement
    solution u using the displacement correlation technique.
    Parameters:
        gb  - grid bucket. For now, contains one higher-dimensional (2D or 3D)
            grid and one lower-dimensional fracture, to be referred to as g_h
            and g_l, respectively.
        u   - solution as computed by FracturedMpsa. One displacement vector
            for each cell center in g_h and one for each of the fracture faces.
            The ordering for a 2D g_h with four cells and four fracture faces
            (two on each side of two g_l fracture cells) is
            [u(c0), v(c0), u(c1), v(c1), u(c2), v(c2), u(c3), v(c3),
            u(f0), v(f0), u(f1), v(f1), u(f2), v(f2), u(f3), v(f3)]
            Here, f0 and f1 are the "left" faces, the original faces before
            splitting of the grid, found in g_h.frac_pairs[0]. f2 and f3 are
            the "right" faces, the faces added by the splitting, found in
            g_h.frac_pairs[1].
        critical_sifs - the stress intensity factors at which the fracture
            yields, one per mode (i.e., we assume this rock parameter to be
            homogeneous for now)
        kw  - optional keyword arguments:
            rm  - oprimal distance from tip faces to correlation points. If not
                provided, a educated guess is made by estimate_rm().

    Returns:
        faces_h_to_open - (possibly empty) array of higher-dimensional faces
            to be split
        stress_intensity_factors - the calculated stress intensity factors for
            each of the lower-dimensional tip faces.


    For more on displacement correlation, see e.g.
        Nejati et al.
        On the use of quarter-point tetrahedral finite elements in linear
        elastic fracture mechanics
        Engineering Fracture Mechanics 144 (2015) 194–221
    """
    dh = gb.dim_max()
    dl = gb.dim_min()
    g_h = gb.grids_of_dimension(dh)[0]
    g_l = gb.grids_of_dimension(dl)[0]
    rm = kw.get('rm', estimate_rm(g_l))
    E = kw.get('E', 1)
    poisson = kw.get('poisson', 0.3)
    mu = E/(2*(1 + poisson))
    kappa = 3 - 4 * poisson
    f_c = gb.edge_prop((g_h, g_l), 'face_cells')[0]
    # Obtain the g_h.dim components of the relative displacement vector
    # corresponding to each fracture tip face of the lower dimension.
    delta_u, rm_vectors, actual_rm = relative_displacements(gb, g_h, g_l, rm, u, f_c)
    # Use them to compute the SIFs locally
    stress_intensity_factors = sif_from_delta_u(delta_u, actual_rm, mu, kappa)
    tips_to_propagate = determine_onset(stress_intensity_factors,
                                        np.array(critical_sifs))
    # Find the right direction. For now, only normal extension is allowed.
    faces_h_to_open = identify_faces_to_open(g_h, g_l, tips_to_propagate, f_c,
                                             rm_vectors)
    return faces_h_to_open, stress_intensity_factors

def identify_faces_to_open(g_h, g_l, tips_to_propagate, f_c, rm_vectors):
    """
    Identify the faces to open. For now, just pick out the face lying
    immediately outside the existing fracture tip faces which we wish to
    propagate.
    TODO: Include angle computation.
    """
    faces_l = g_l.tags['tip_faces'].nonzero()[0]
    # Construct points lying just outside the fracture tips
    extended_points = g_l.face_centers[:, faces_l] + rm_vectors / 100
    # Find closest higher-dimensional face.
    faces_h = []
    for i in range(extended_points.shape[1]):
        if tips_to_propagate[i]:
            p = extended_points[:, i]
            distances = comp_geom.dist_point_pointset(p, g_h.face_centers)
            faces_h.append(np.argmin(distances))

    return np.array(faces_h, dtype=int)


def determine_onset(stress_intensity_factors, critical_values):
    """
    For the time being, very crude criterion: K_I > K_I,cricial.
    TODO: Extend to equivalent SIF, taking all three modes into account.
    """
    return np.absolute(stress_intensity_factors[0]) > critical_values[0]


def sif_from_delta_u(d_u, rm, mu, kappa):
    """
    Compute the stress intensity factors from the relative displacements
    Parameters:
        d_u     - relative displacements, g_h.dim x n
        rm      -
    """
#    rm = np.linalg.norm(rm_vectors, axis=0)
    (dim, n_points) = d_u.shape
    sifs = np.zeros(d_u.shape)
    rm = rm.T
    sifs[0] = np.sqrt(2 * np.pi / rm) * np.divide(mu, kappa + 1) * d_u[1, :]
    sifs[1] = np.sqrt(2 * np.pi / rm) * np.divide(mu, kappa + 1) * d_u[0, :]
    if dim == 3:
        sifs[2] = np.sqrt(2 * np.pi / rm) * np.divide(mu, 4) * d_u[2, :]

    return sifs


def relative_displacements(gb, g_h, g_l, rm, u, f_c):
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
        gb
        g_h
        g_l
        rm  - optimal distance from fracture front to correlation point.
        u   - displacement solution.
        f_c - face_cells of the grid pair.
    Returns:
        np.array(ndim, ntips) - the relative displacements in local coordinates
            for each fracture tip.
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
    rm_vectors = normals_l * rm
    optimal_points = g_l.face_centers[:, faces_l] - rm_vectors
    cells_l = []
    actual_rm = []
    for i in range(n_tips):
        p = optimal_points[:, i]
        distances = comp_geom.dist_point_pointset(p, g_l.cell_centers)
        cell_ind = np.argmin(distances)
        dist = comp_geom.dist_point_pointset(g_l.face_centers[:, faces_l[i]],
                                             g_l.cell_centers[:, cell_ind])
        cells_l.append(cell_ind)
        actual_rm.append(dist)
    # Find the two faces used for relative displacement calculations for each
    # of the tip faces

    faces_h = f_c[cells_l].nonzero()[1]

    faces_h = faces_h.reshape((2, n_tips), order='F')

    # Extract the face displacements
    u_faces = u[g_h.dim * g_h.num_cells:]
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
        pts -= g_l.cell_centers[:, cell_l].reshape((3,1))
        normal_h_1 = g_h.face_normals[:, face_pair[1]] \
                    * g_h.cell_faces[face_pair[1]].data
        if g_h.dim == 3:
            # Rotate to xz, i.e., normal alignes with y. Pass normal of the
            # second face to ensure that the first one is on top in the local
            # coordinate system

            R1 = comp_geom.project_plane_matrix(pts, normal=normal_h_1,
                                                reference=[0, 1, 0])
            # Rotate so that tangent aligns with z-coordinate
#            normal = comp_geom.compute_normal(pts)
#            reference=[0, 1, 0]
#            reference = np.asarray(reference, dtype=np.float)
#            angle = np.arccos(np.dot(normal, reference))
#            vect = np.cross(normal, reference)
            nodes_l = g_l.face_nodes[:, faces_l[i]].nonzero()[0]
            translated_pts = g_l.nodes[:, nodes_l] \
                            - g_l.face_centers[:, face_l].reshape((3,1))
            face_coordinates_rot = np.dot(R1, translated_pts)
            normal_l = np.dot(R1, g_l.face_normals[:, face_l])
            tangent_l = np.cross(normal_l, np.array([0, 1, 0]))

            R2 = comp_geom.project_line_matrix(face_coordinates_rot,
                                               tangent = tangent_l,
                                               reference=[0, 0, 1], flip_flag=True)
            R = np.dot(R2, R1)
        else:
            # Rotate so that the face normal, on whose line pts lie, aligns
            # with x
            tangent_l = np.cross(normal_h_1, np.array([0, 0, 1]))
            R = comp_geom.project_line_matrix(pts, tangent=tangent_l,
                                              reference=[1, 0, 0], flip_flag=True)


        # Find what frac-pair the tip i corresponds to
        j = setmembership.ismember_rows(face_pair[:, np.newaxis],
                                        g_h.frac_pairs)[1]
        u_left = u_faces[fvutils.expand_indices_nd(j, g_h.dim)]
        u_right = u_faces[fvutils.expand_indices_nd(j, g_h.dim)
                        + g_l.num_cells * g_h.dim]
        d_u = u_left - u_right
        d_u = np.dot(R, np.append(d_u, np.zeros((1, 3 - g_h.dim))))[:g_h.dim]

        delta_us = np.append(delta_us, d_u[:, np.newaxis], axis=1)

    return delta_us, rm_vectors, np.array(actual_rm)

def identify_correlation_nodes(g_l, rm, face_cells):
    """
    For each of the tip faces, find the corresponding correlation node. These
    are centers of the adjacent higher-dimensional faces.
    Parameters:
        g_l     - fracture grid.
        rm      - optimal distance from tip face to correlation point.
        face_cells  - the connectivity matrix of between g_l and the
            surrounding g_h.
    Returns:
        faces_l     - np.array of size 2 x number of tips with the indices of
            the lower-dimensional faces.
        faces_h     - same for the higher-dimensional faces.

    """

    faces_l = g_l.tags['tip_faces'].nonzero()[0]
    normals_l = np.divide(g_l.face_normals[:, faces_l], g_l.face_areas[faces_l])
    signed_normals_l = normals_l * g_l.cell_faces[faces_l]
    optimal_points = g_l.face_centers[:, faces_l] - signed_normals_l * rm
    cells_l = []
    for p in np.nditer(optimal_points, flags=['external_loop'], order='F'):
        distances = comp_geom.dist_point_pointset(p, g_l.cell_centers)
        cells_l.append(np.argmin(distances))
    faces_h = face_cells[cells_l].nonzero()[1]
    faces_h.resize((2, len(cells_l)))
    return faces_l, faces_h

def estimate_rm(g):
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
    k = 1.4

    faces = g.tags['tip_faces'].nonzero()[0]
    if g.dim == 2:
        rm = k * np.mean(g.face_areas[faces])
    else:
        # Dimension is 1, and face_area is 1. Use cell volume of neighbouring
        # cell instead.
        cells = g.cell_faces[faces].nonzero()[1]
        rm = k * np.mean(g.cell_volumes[cells])
    return rm