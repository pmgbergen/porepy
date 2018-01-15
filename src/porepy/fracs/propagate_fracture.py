#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 17:32:35 2018

@author: ivar

Propagation of fractures. Much in common with (and reuse of) split_grid.
For now assumes:
        single growing fracture, not growing into other fractures
"""
import warnings
import numpy as np
import scipy.sparse as sps
from porepy.fracs import split_grid
from porepy.utils import setmembership, tags, sort_points


def propagate_fracture(gb, gh, gl, faces_h, apertures_l):
    """
    Propagate the fracture defined by gl to the higher-dimensional faces.

    """

    # Keep track of original information:
    n_old_faces_l = gl.num_faces
    n_old_cells_l = gl.num_cells
    n_old_nodes_l = gl.num_nodes

    # It is convenient to tag the nodes lying on the domain boundary. This
    # helps updating the face tags later:
    tags.add_node_tags_from_face_tags(gb, 'domain_boundary')

    # The new nodes in the lower dimension and their union with the
    # boundary nodes where the fracture propagates. The former are added to
    # the global_point_ind of gl:
    unique_node_ind_l, unique_node_ind_h = update_nodes(gh, gl, faces_h)

    n_new_faces, new_face_centers = update_connectivity(gl, gh, n_old_nodes_l,
                                                        unique_node_ind_l,
                                                        n_old_faces_l,
                                                        n_old_cells_l,
                                                        unique_node_ind_h,
                                                        faces_h)
    # Add new faces to gl
    append_face_geometry(gl, n_new_faces, new_face_centers)
    # Same for cells
    update_cells(gh, gl, faces_h)

    update_permeability(gb, gl, apertures_l)

    # Split gh along faces_h
    split_fracture_extension(gb, gh, gl, faces_h, unique_node_ind_h)


def update_connectivity(gl, gh, n_old_nodes_l, unique_node_indices_l,
                        n_old_faces_l, n_old_cells_l, unique_nodes_h, faces_h):
    """
    Update of cell_faces of the lower grid after incertion of new cells at the
    higher-dimensional faces_h. Should be called after initialization of tags
    and geometry of gl by append_face_geometry and append_face_tags.
    """
    # Extract immediate information
    n_new_cells_l = faces_h.size
    new_cells_l = np.arange(n_old_cells_l, n_old_cells_l + n_new_cells_l)
    # Initialize
    new_faces_l = np.empty((gl.dim, 0))
    new_face_centers_l = np.empty((3, 0))
    face_counter_l = n_old_faces_l
    # Copy what is to be updated
    old_cell_faces = gl.cell_faces.copy()
    old_face_nodes = gl.face_nodes.copy()
    # Get the face_node indices to form lower-dimensional faces on the form
    # [[nodes of face 1], [nodes of face 2], ...], i.e., array where each face
    # is represented by the nodes it consists of.
    all_faces_l = np.reshape(gl.face_nodes.indices, (n_old_faces_l, gl.dim)).T

    # Initialize indices and values for the cell_faces update
    (ind_f, ind_c, cf_val) = (np.empty(0), np.empty(0), np.empty(0))
    # and for the face_nodes update
    (fn_ind_f, fn_ind_n) = (np.empty(0), np.empty(0))

    for i, c in enumerate(new_cells_l):
        # Find the nodes involved
        face_h = faces_h[i]
        local_nodes_h = gh.face_nodes[:, face_h].nonzero()[0]
        in_unique_nodes = order_preserving_find(local_nodes_h, unique_nodes_h)
        local_nodes_l = np.array(unique_node_indices_l[in_unique_nodes],
                                 dtype=int)
        # Get geometry information
        local_pts = gl.nodes[:, local_nodes_l]
        local_cell_center = np.mean(local_pts, axis=1)
        # Store face center for the update of gl.face_centers

        # Faces are defined by one node in 1d and two in 2d. This requires
        # dimension dependent treatment:
        if gl.dim == 2:
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

        # Check which faces exist in gl already, either from before propgation
        # or from previous runs through current loop:
        (exist, existing_faces_l) = setmembership.ismember_rows(faces_l,
                                                                all_faces_l)
        # The existing faces are no longer tips (but internal). The new faces
        # are tips (or on the domain boundary).
        gl.tags['tip_faces'][existing_faces_l] = False
        gl.tags['boundary_faces'][existing_faces_l] = False
        n_new_local_faces_l = np.sum(~exist)

        new_face_indices_l = np.arange(face_counter_l,
                                       face_counter_l
                                       + n_new_local_faces_l)
        face_counter_l += n_new_local_faces_l
        append_face_tags(gl, n_new_local_faces_l)
        # The existing faces are tagged according to the information from the
        # node tags of gh.
        fi = local_faces_h[:, ~exist]
        domain_boundary_faces \
            = np.all(gh.tags['domain_boundary_nodes'][fi], axis=0)
        gl.tags['tip_faces'][new_face_indices_l] = ~domain_boundary_faces
        gl.tags['domain_boundary_faces'][new_face_indices_l] \
            = domain_boundary_faces
        gl.tags['boundary_faces'][new_face_indices_l] = True
        # Add the new faces
        all_faces_l = np.append(all_faces_l, faces_l[:, ~exist], axis=1)
        # Find indices of face_nodes to be updated.
        ind_n_local = faces_l[:, ~exist]
        local_pts = gl.nodes[:, ind_n_local]
        local_face_centers = np.mean(local_pts, axis=1)
        new_face_centers_l = np.append(new_face_centers_l,
                                       np.atleast_2d(local_face_centers),
                                       axis=1)
        new_faces_l = np.append(new_faces_l, ind_n_local, axis=1)
        all_local_faces = np.empty(faces_l.shape[-1])
        all_local_faces[exist] = existing_faces_l
        all_local_faces[~exist] = new_face_indices_l
        ind_f_local = np.tile(all_local_faces, gl.dim)
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
        cf_val_loc[ind_local] = - gl.cell_faces[ind_in_original, :].data
        cf_val_loc[ind_not_local] = -1

        cf_val = np.append(cf_val, cf_val_loc)

    # Resize and update face_nodes ...
    gl.face_nodes = sps.csc_matrix((gl.num_nodes, face_counter_l), dtype=bool)
    gl.face_nodes[:n_old_nodes_l, :n_old_faces_l] = old_face_nodes
    gl.face_nodes[fn_ind_n, fn_ind_f] = True
    # ... and cell_faces
    gl.cell_faces = sps.csc_matrix((face_counter_l,
                                    n_old_cells_l + n_new_cells_l))
    gl.cell_faces[0:n_old_faces_l, 0:n_old_cells_l] = old_cell_faces
    gl.cell_faces[ind_f, ind_c] = cf_val

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


def update_cells(gh, gl, faces_h):
    """
    Cell information is inherited directly from the higher-dimensional faces
    we are splitting. The function updates num_cells, cell_centers and
    cell_volumes.
    """
    gl.num_cells += faces_h.size
    gl.cell_centers = np.append(gl.cell_centers,
                                gh.face_centers[:, faces_h], axis=1)
    gl.cell_volumes = np.append(gl.cell_volumes, gh.face_areas[faces_h])


def update_permeability(gb, gl, apertures_l):
    warnings.warn('apertures not updated', )


def update_nodes(gh, gl, faces_h):
    """
    Finds the nodes in the lower-dimensional grid corresponding to the higher-
    dimensional faces to be split. Updates node information in gl:
        global_point_ind
        nodes
        num_nodes
    Returns:
        unique_nodes_l - numpy array (number of involved nodes x 1) Indices of
            the nodes (as arranged in gl.nodes).
        unique_nodes_h - same, but corresponding to gh.nodes.
    """
    nodes_h = gh.face_nodes[:, faces_h].nonzero()[0]
    unique_nodes_h = np.unique(nodes_h)
    unique_global_nodes = gh.global_point_ind[unique_nodes_h]

    are_old_global_nodes_in_l = np.in1d(unique_global_nodes,
                                        gl.global_point_ind)
    are_new_global_nodes_in_l = np.logical_not(are_old_global_nodes_in_l)
    new_node_indices_h = unique_nodes_h[are_new_global_nodes_in_l]
    new_global_node_indices_l = unique_global_nodes[are_new_global_nodes_in_l]
    gl.global_point_ind = np.append(gl.global_point_ind,
                                    new_global_node_indices_l)
    new_nodes_l = gh.nodes[:, new_node_indices_h].copy()

    # Order preserving find:
    unique_nodes_l = order_preserving_find(unique_global_nodes,
                                                  gl.global_point_ind)

    gl.nodes = np.append(gl.nodes, new_nodes_l, axis=1)
    gl.num_nodes += new_nodes_l.shape[1]
    return unique_nodes_l, unique_nodes_h


def append_face_geometry(g, n_new_faces, new_centers):
    """
    Appends and updates faces geometry information for new faces.
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


def split_fracture_extension(bucket, gh, gl, faces_h, nodes_h):
    """
    Split the higher-dimensional grid along specified faces. Updates made to
    face_cells of the grid pair and the nodes and faces of the higher-
    dimensional grid.
    Parameters
    ----------
    bucket      - A grid bucket
    gh          - Higher-dimensional grid to be split along specified faces.
    gl          - Immersed lower-dimensional grid.
    faces_h     - The higher-dimensional faces to be split. Should be direct
                    extension of already split faces.
    nodes_h     - The corresponding (higher-dimensional) nodes.
    """

    face_cells = bucket.edge_prop((gh, gl), 'face_cells')

    # We split all the faces that are connected to faces_h
    # The new faces will share the same nodes and properties (normals,
    # etc.)

    face_cells = split_grid.split_certain_faces(gh, face_cells, faces_h)
    bucket.add_edge_prop('face_cells', [(gh,gl)], face_cells)

    # We now find which lower-dim nodes correspond to which higher-
    # dim nodes. We split these nodes according to the topology of
    # the connected higher-dim cells. At a X-intersection we split
    # the node into four, while at the fracture boundary it is not split.

#    gl_2_gh_nodes = [bucket.target_2_source_nodes(gl, gh)]

    split_grid.split_nodes(gh, [gl], [nodes_h])

    # Remove zeros from cell_faces

    [g.cell_faces.eliminate_zeros() for g, _ in bucket]
    return bucket


