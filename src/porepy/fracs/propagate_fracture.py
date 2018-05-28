#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
import porepy as pp


def propagate_fractures(gb, faces):
    """
    gb - grid bucket with matrix and fracture grids.
    faces_h - list of list of faces to be split in the highest-dimensional
        grid. The length of the outer list equals the number of fractures.
        Each entry in the list is a list containing the higher-dimensional
        indices of the faces to be split for the extension of the corresponding
        fracture.
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
    """
    dim_h = gb.dim_max()
    g_h = gb.grids_of_dimension(dim_h)[0]
    # First initialise certain tags to get rid of any existing tags from
    # previous calls
    for g, d in gb:
        if g.dim == dim_h:
            d['new_cells'] = np.empty(0, dtype=int)
            d['new_faces'] = np.empty(0, dtype=int)
        else:
            d['new_cells'] = np.empty(0, dtype=int)
            d['new_faces'] = np.empty(0, dtype=int)

    for i, g_l in enumerate(gb.grids_of_dimension(dim_h - 1)):
        faces_h = np.array(faces[i])
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
        pp.utils.tags.add_node_tags_from_face_tags(gb, 'domain_boundary')

        # Get the "involved nodes", i.e., the union between the new nodes in
        # the lower dimension and the boundary nodes where the fracture
        # propagates. The former are added to the global_point_ind of g_l.
        unique_node_ind_l, unique_node_ind_h = update_nodes(g_h, g_l, faces_h)

        # Update the connectivity matrices (cell_faces and face_nodes) and tag
        # the lower-dimensional faces, including re-classification of (former)
        # tips to internal faces, where appropriate.
        n_new_faces, new_face_centers \
            = update_connectivity(g_l, g_h, n_old_nodes_l, unique_node_ind_l,
                                  n_old_faces_l, n_old_cells_l,
                                  unique_node_ind_h, faces_h)
        # Add new faces to g_l
        append_face_geometry(g_l, n_new_faces, new_face_centers)
        # Same for cells
        new_cells = update_cells(g_h, g_l, faces_h)

        # Split g_h along faces_h
        split_fracture_extension(gb, g_h, g_l, faces_h, unique_node_ind_h,
                                 new_cells)

        # Store information on which faces and cells have just been added.
        # Note that we only keep track of the faces and cells from the last
        # propagation call!
        new_faces_h = g_h.frac_pairs[1, np.isin(g_h.frac_pairs[0], faces_h)]
        d_h = gb.node_props(g_h)
        d_l = gb.node_props(g_l)
        d_h['partial_update'] = True
        d_l['partial_update'] = True
        d_h['new_faces'] = np.append(d_h['new_faces'], new_faces_h)
        d_l['new_cells'] = np.append(d_l['new_cells'], new_cells)
        new_faces_l = np.arange(g_l.num_faces - n_new_faces, g_l.num_faces)
        d_l['new_faces'] = np.append(d_l['new_faces'], new_faces_l)

        # Update geometry on each iteration to ensure correct tags.
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
        in_unique_nodes = pp.utils.setmembership.ismember_rows(local_nodes_h,
                                                               unique_nodes_h,
                                                               sort=False)[1]
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
            map_to_sorted \
                = pp.utils.sort_points.sort_point_plane(local_pts,
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
        (exist, existing_faces_l) \
            = pp.utils.setmembership.ismember_rows(faces_l, all_faces_l)
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
    g_l.face_nodes = sps.csc_matrix((g_l.num_nodes, face_counter_l),
                                    dtype=bool)
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
    unique_nodes_l = pp.utils.setmembership.ismember_rows(unique_global_nodes,
                                                          g_l.global_point_ind,
                                                          sort=False)[1]

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
    keys = pp.utils.tags.standard_face_tags()
    new_tags = [np.zeros(n_new_faces, dtype=bool) for _ in range(len(keys))]
    pp.utils.tags.append_tags(g.tags, keys, new_tags)


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

    Same level as split_faces
    """
    # We are splitting faces in g_h. This affects all the immersed fractures,
    # as face_cells has to be extended for the new faces_h.
    neigh = np.array(bucket.node_neighbors(g_h))

    # Find the neighbours that are lower dimensional
    is_low_dim_grid = np.where([w.dim < g_h.dim for w in neigh])
    low_dim_neigh = neigh[is_low_dim_grid]
    edges = [(g_h, w) for w in low_dim_neigh]
    g_l_ind = np.nonzero(low_dim_neigh == g_l)[0]
    if len(edges) == 0:
        # No lower dim grid. Nothing to do.
        warnings.warn('Unexpected neighbourless g_h in fracture propagation')
        return

    face_cell_list = [bucket.edge_props(e, 'face_cells') for e in edges]  #[bucket.edge_props((g_h, g_l), 'face_cells')]

    # We split all the faces that are connected to faces_h
    # The new faces will share the same nodes and properties (normals,
    # etc.)
    face_cell_list = pp.fracs.split_grid.split_certain_faces(g_h,
                                                             face_cell_list,
                                                             faces_h, cells_l,
                                                             g_l_ind)

    for e, f in zip(edges, face_cell_list):
        bucket.edge_props(e)['face_cells'] = f

    # We now find which lower-dim nodes correspond to which higher-
    # dim nodes. We split these nodes according to the topology of
    # the connected higher-dim cells. At a X-intersection we split
    # the node into four, while at the fracture boundary it is not split.
    pp.fracs.split_grid.split_nodes(g_h, [g_l], [nodes_h])

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


def propgation_angle(K):
    """
    Compute the angle of propagation from already computed SIFs. The
    computation is done in the local coordinsate system of the fracture tips,
    and positive angles correspond to.
    Intended for a single fracture grid.

    Parameters:
        K: array of stress intensity factors, with mode along first axis and
            face along second.

    Returns:
        phi: array (number of tip faces, ) of propagation angles in radians.
    """
    A = 140 / 180 * np.pi
    B = -70 / 180 * np.pi
    aK = np.absolute(K)
    phi = (A * aK[1]/(K[0] + aK[1] + aK[2])
            + B * np.square(aK[2]/(K[0] + aK[1] + aK[2])))
    neg_ind = K[1] > 0
    phi[neg_ind] = - phi[neg_ind]
    return phi
