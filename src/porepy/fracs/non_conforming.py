#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:17:04 2017

@author: eke001
"""

import numpy as np
import scipy.sparse as sps

import porepy as pp

from porepy.utils import tags
from porepy.utils.matrix_compression import rldecode
from porepy.utils.setmembership import unique_columns_tol, ismember_rows

from porepy.fracs import tools as fractools
import porepy.utils.comp_geom as cg


def merge_grids(grids, intersections, tol=1e-4):
    """ Main method of module, merge all grids
    """
    list_of_grids, global_ind_offset = init_global_ind(grids)
    grids_1d = process_intersections(
        grids, intersections, global_ind_offset, list_of_grids, tol
    )
    grid_list_by_dim = [[], [], []]

    grid_list_by_dim[1] = grids_1d

    for g in grids:
        grid_list_by_dim[0].append(g[0][0])

    return grid_list_by_dim


def init_global_ind(gl):
    """ Initialize a global indexing of nodes to a set of local grids.

    Parameters:
        gl (triple list): Outer: One per fracture, middle: dimension, inner:
            grids within the dimension.

    Returns:
        list (Grid): Single list representation of all grids. Fractures and
            dimensions will be mixed up.
        int: Global number of nodes.

    """

    list_of_grids = []

    # The global point index is for the moment set within each fracture
    # (everybody start at 0). Adjust this.
    global_ind_offset = 0
    # Loop over fractures
    for frac_ind, f in enumerate(gl):
        # The 2d grid is the only item in the middle list
        f[0][0].frac_num = frac_ind

        # Loop over dimensions within the fracture
        for gd in f:
            # Loop over grids within the dimension
            for g in gd:
                g.global_point_ind += global_ind_offset
                list_of_grids.append(g)
        # Increase the offset with the total number of nodes on this fracture
        global_ind_offset += f[0][0].num_nodes

    return list_of_grids, global_ind_offset


def process_intersections(grids, intersections, global_ind_offset, list_of_grids, tol):
    """ Loop over all intersections, combined two and two grids.
    """

    # All connections will be hit upon twice, one from each intersecting fracture.
    # Keep track of which connections have been treated, and can be skipped.
    num_frac = len(grids)
    isect_is_processed = sps.lil_matrix((num_frac, num_frac), dtype=np.bool)

    grid_1d_list = []

    # Loop over all fractures
    for frac_ind, frac in enumerate(grids):

        # Pick out the 2d grid of this fracture
        g = grids[frac_ind][0][0]

        # Loop over all 1d grids in this fracture
        for ind_1d, g_1d in enumerate(frac[1]):

            # Find index and grid of the other fracture of this intersection
            other_frac_ind = intersections[frac_ind][ind_1d]
            h = grids[other_frac_ind][0][0]

            # Check if we have treated this intersection before
            row = g.frac_num
            col = h.frac_num
            if isect_is_processed[row, col]:
                continue
            else:
                # Mark the connection as treated
                isect_is_processed[row, col] = True
                isect_is_processed[col, row] = True

            # Find the 1d grid of this intersection, as created from the other fracture
            g_in_other = np.where(intersections[other_frac_ind] == frac_ind)[0][0]
            h_1d = grids[other_frac_ind][1][g_in_other]
            g_1d.compute_geometry()
            h_1d.compute_geometry()
            g_new_1d, global_ind_offset = combine_grids(
                g, g_1d, h, h_1d, global_ind_offset, list_of_grids, tol
            )
            # Append the new 1d grid to the general list of grids, so that it
            # will have its global point indices updated as we go.
            grid_1d_list.append(g_new_1d)
    return grid_1d_list


def combine_grids(g, g_1d, h, h_1d, global_ind_offset, list_of_grids, tol):

    combined_1d, global_ind_offset, g_in_combined, h_in_combined, g_sort, h_sort = merge_1d_grids(
        g_1d, h_1d, global_ind_offset, tol
    )

    # First update fields for first grid
    fn_orig = np.reshape(g.face_nodes.indices, (2, g.num_faces), order="F")
    node_coord_orig = g.nodes.copy()
    new_nodes, delete_faces, global_ind_offset = update_nodes(
        g, g_1d, combined_1d, g_in_combined, g_sort, global_ind_offset, list_of_grids
    )

    num_new_faces = combined_1d.num_cells
    new_nodes_offset = new_nodes[0]
    new_faces = update_face_nodes(g, delete_faces, num_new_faces, new_nodes_offset)

    update_cell_faces(
        g, delete_faces, new_faces, g_in_combined, fn_orig, node_coord_orig
    )

    # Then updates for the second grid
    fn_orig = np.reshape(h.face_nodes.indices, (2, h.num_faces), order="F")
    node_coord_orig = h.nodes.copy()
    new_nodes, delete_faces, global_ind_offset = update_nodes(
        h, h_1d, combined_1d, h_in_combined, h_sort, global_ind_offset, list_of_grids
    )

    new_nodes_offset = new_nodes[0]
    new_faces = update_face_nodes(h, delete_faces, num_new_faces, new_nodes_offset)

    update_cell_faces(
        h, delete_faces, new_faces, h_in_combined, fn_orig, node_coord_orig
    )

    return combined_1d, global_ind_offset


def merge_1d_grids(g, h, global_ind_offset=0, tol=1e-4):
    """ Merge two 1d grids with non-matching nodes to a single grid.

    The grids should have common start and endpoints. They can be into 3d space
    in a genreal way.

    The function is primarily intended for merging non-conforming DFN grids.

    Parameters:
        g: 1d tensor grid.
        h: 1d tensor grid
        glob_ind_offset (int, defaults to 0): Off set for the global point
            index of the new grid.
        tol (double, defaults to 1e-4): Tolerance for when two nodes are merged
            into one.

    Returns:
        TensorGrid: New tensor grid, containing the combined node definition.
        int: New global ind offset, increased by the number of cells in the
            combined grid.
        np.array (int): Indices of common nodes (after sorting) of g and the
            new grid.
        np.array (int): Indices of common nodes (after sorting) of h and the
            new grid.
        np.array (int): Permutation indices that sort the node coordinates of
            g. The common indices between g and the new grid are found as
            new_grid.nodes[:, g_in_combined] = g.nodes[:, sorted]
        np.array (int): Permutation indices that sort the node coordinates of
            h. The common indices between h and the new grid are found as
            new_grid.nodes[:, h_in_combined] = h.nodes[:, sorted]

    """

    # Nodes of the two 1d grids, combine them
    gp = g.nodes
    hp = h.nodes
    combined = np.hstack((gp, hp))

    num_g = gp.shape[1]
    num_h = hp.shape[1]

    # Keep track of where we put the indices of the original grids
    g_in_full = np.arange(num_g)
    h_in_full = num_g + np.arange(num_h)

    # The tolerance should not be larger than the smallest distance between
    # two points on any of the grids.
    diff_gp = np.min(cg.dist_pointset(gp, True))
    diff_hp = np.min(cg.dist_pointset(hp, True))
    min_diff = np.minimum(tol, 0.5 * np.minimum(diff_gp, diff_hp))

    # Uniquify points
    combined_unique, _, new_2_old = unique_columns_tol(combined, tol=min_diff)
    # Follow locations of the original grid points
    g_in_unique = new_2_old[g_in_full]
    h_in_unique = new_2_old[h_in_full]

    # The combined nodes must be sorted along their natural line.
    # Find the dimension with the largest spatial extension, and sort those
    # coordinates
    max_coord = combined_unique.max(axis=1)
    min_coord = combined_unique.min(axis=1)
    dx = max_coord - min_coord
    sort_dim = np.argmax(dx)

    sort_ind = np.argsort(combined_unique[sort_dim])
    combined_sorted = combined_unique[:, sort_ind]

    # Follow the position of the orginial nodes through sorting
    _, g_sorted = ismember_rows(g_in_unique, sort_ind)
    _, h_sorted = ismember_rows(h_in_unique, sort_ind)

    num_new_grid = combined_sorted.shape[1]

    # Create a new 1d grid.
    # First use a 1d coordinate to initialize topology
    new_grid = pp.structured.TensorGrid(np.arange(num_new_grid))
    # Then set the right, 3d coordinates
    new_grid.nodes = cg.make_collinear(combined_sorted)

    # Set global point indices
    new_grid.global_point_ind = global_ind_offset + np.arange(num_new_grid)
    global_ind_offset += num_new_grid

    return (
        new_grid,
        global_ind_offset,
        g_sorted,
        h_sorted,
        np.arange(num_g),
        np.arange(num_h),
    )


def update_global_point_ind(grid_list, old_ind, new_ind):
    """ Update global point indices in a list of grids.

    The method replaces indices in the attribute global_point_ind in the grid.
    The update is done in place.

    Parameters:
        grid_list (list of grids): Grids to be updated.
        old_ind (np.array): Old global indices, to be replaced.
        new_ind (np.array): New indices.

    """
    for g in grid_list:
        ismem, o2n = ismember_rows(old_ind, g.global_point_ind)
        g.global_point_ind[o2n] = new_ind[ismem]


def update_nodes(
    g, g_1d, new_grid_1d, this_in_combined, sort_ind, global_ind_offset, list_of_grids
):
    """ Update a 2d grid to conform to a new grid along a 1d line.

    Intended use: A 1d mesh that is embedded in a 2d mesh (along a fracture)
    has been updated / refined. This function then updates the node information
    in the 2d grid.

    Parameters:
        g (grid, dim==2): Main grid to update. Has faces along a fracture.
        g_1d (grid, dim==1): Original line grid along the fracture.
        new_grid_1d (grid, dim==1): New line grid, formed by merging two
            coinciding, but non-matching grids along a fracture.
        this_in_combined (np.ndarray): Which nodes in the new grid are also in
            the old one, as returned by merge_1d_grids().
        sort_ind (np.ndarray): Sorting indices of the coordinates of the old
            grid, as returned by merge_1d_grids().
        list_of_grids (list): Grids for all dimensions.

    Returns:


    """
    nodes_per_face = 2
    # Face-node relation for the grid, in terms of local and global indices
    fn = g.face_nodes.indices.reshape((nodes_per_face, g.num_faces), order="F")
    fn_glob = np.sort(g.global_point_ind[fn], axis=0)

    # Mappings between faces in 2d grid and cells in 1d
    # 2d faces along the 1d grid will be deleted.
    delete_faces, cell_1d = fractools.obtain_interdim_mappings(
        g_1d, fn_glob, nodes_per_face
    )

    # All 1d cells should be identified with 2d faces
    assert (
        cell_1d.size == g_1d.num_cells
    ), """ Failed to find mapping between
        1d cells and 2d faces"""

    # The nodes of identified faces on the 2d grid will be deleted
    delete_nodes = np.unique(fn[:, delete_faces])

    # Nodes to be added will have indicies towards the end
    num_nodes_orig = g.num_nodes
    num_delete_nodes = delete_nodes.size
    num_nodes_not_on_fracture = num_nodes_orig - num_delete_nodes

    # Define indices of new nodes.
    new_nodes = num_nodes_orig - delete_nodes.size + np.arange(new_grid_1d.num_nodes)

    # Adjust node indices in the face-node relation
    # First, map nodes between 1d and 2d grids. Use sort_ind here to map
    # indices of g_1d to the same order as the new grid
    _, node_map_1d_2d = ismember_rows(
        g.global_point_ind[delete_nodes], g_1d.global_point_ind
    )
    tmp = np.arange(g.num_nodes)
    adjustment = np.zeros_like(tmp)
    adjustment[delete_nodes] = 1
    node_adjustment = tmp - np.cumsum(adjustment)
    # Nodes along the 1d grid are deleted and inserted again. Let the
    # adjutsment point to the restored nodes.
    # node_map_1d_2d maps from ordering in delete_nodes to ordering of 1d
    # points (old_grid). this_in_combined then maps further to the ordering of
    # the new 1d grid
    node_adjustment[delete_nodes] = (
        g.num_nodes - num_delete_nodes + this_in_combined[node_map_1d_2d]
    )

    g.face_nodes.indices = node_adjustment[g.face_nodes.indices]

    # Update node coordinates and global indices for 2d mesh
    g.nodes = np.hstack((g.nodes, new_grid_1d.nodes))

    new_global_points = new_grid_1d.global_point_ind
    g.global_point_ind = np.append(g.global_point_ind, new_global_points)

    # Global index of deleted points
    old_global_pts = g.global_point_ind[delete_nodes]

    # Update any occurences of the old points in other grids. When sewing
    # together a DFN grid, this may involve more and more updates as common
    # nodes are found along intersections.

    # The new grid should also be added to the list, if it is not there before
    if not new_grid_1d in list_of_grids:
        list_of_grids.append(new_grid_1d)
    update_global_point_ind(
        list_of_grids,
        old_global_pts,
        new_global_points[this_in_combined[node_map_1d_2d]],
    )

    # Delete old nodes
    g.nodes = np.delete(g.nodes, delete_nodes, axis=1)
    g.global_point_ind = np.delete(g.global_point_ind, delete_nodes)

    g.num_nodes = g.nodes.shape[1]
    return new_nodes, delete_faces, global_ind_offset


def update_face_nodes(
    g, delete_faces, num_new_faces, new_node_offset, nodes_per_face=None
):
    """ Update face-node map by deleting and inserting new faces.

    The method deletes specified faces, adds new ones towards the end. It does
    nothing to adjust the face-node relation for remaining faces.

    The code assumes a constant number of nodes per face.

    Parameters:
        g (grid): Grid to have its faces modified. Should have fields
            face_nodes and num_faces.
        delete_faces (np.array): Index of faces to be deleted.
        num_new_faces (int): Number of new faces to create.
        new_node_offset (int): Offset index of the new nodes.
        nodes_per_face (int, optional): Number of nodes per face, assumed equal
            for all faces. Defaults to g.dim, that is, simplex grids

    Returns:
        np.array: Index of the new faces.

    """

    if nodes_per_face is None:
        nodes_per_face = g.dim

    # Indices of new nodes.
    new_face_nodes = np.tile(np.arange(num_new_faces), (nodes_per_face, 1)) + np.arange(
        nodes_per_face
    ).reshape((nodes_per_face, 1))
    # Offset the numbering: The new nodes are inserted after all outside nodes
    new_face_nodes = new_node_offset + new_face_nodes
    # Number of new faces in mesh
    ind_new_face = g.num_faces - delete_faces.size + np.arange(num_new_faces)

    # Modify face-node map
    # First obtain face-node relation as a matrix. Thankfully, we know the
    # number of nodes per face.
    fn = g.face_nodes.indices.reshape((nodes_per_face, g.num_faces), order="F")
    # Delete old faces
    fn = np.delete(fn, delete_faces, axis=1)
    # Add new face-nodes
    fn = np.append(fn, new_face_nodes, axis=1)

    indices = fn.flatten(order="F")

    # Trivial updates of data and indptr. Fortunately, this is 2d
    data = np.ones(fn.size, dtype=np.bool)
    indptr = np.arange(0, fn.size + 1, nodes_per_face)
    g.face_nodes = sps.csc_matrix((data, indices, indptr))
    g.num_faces = int(fn.size / nodes_per_face)
    assert g.face_nodes.indices.max() < g.nodes.shape[1]

    return ind_new_face


def update_cell_faces(
    g, delete_faces, new_faces, in_combined, fn_orig, node_coord_orig, tol=1e-4
):
    """ Replace faces in a cell-face map.

    If faces have been refined (or otherwise modified), it is necessary to
    update the cell-face relation as well. This function does so, while taking
    care that the (implicit) mapping between cells and nodes is ordered so that
    geometry computation still works.

    The changes of g.cell_faces are done in-place.

    It is assumed that the new faces that replace an old are ordered along the
    common line. E.g. if a face with node coordinates (0, 0) and (3, 0) is
    replaced by three new faces of unit length, they should be ordered as
    1. (0, 0) - (1, 0)
    2. (1, 0) - (2, 0)
    3. (2, 0) - (3, 0)
    Switching the order into 3, 2, 1 is okay, but, say, 1, 3, 2 will create
    problems.

    It is also tacitly assumed that each cell has at most one face deleted.
    Changing this may not be difficult, but has not been prioritized so far.

    The function has been tested in 2d only, reliability in 3d is unknown,
    but doubtful.

    Parameters:
        g (grid): To be updated.
        delete_faces (np.ndarray): Faces to be deleted, as found in
            g.cell_faces
        new_faces (np.ndarray): Index of new faces, as found in g.face_nodes
        in_combined (np.ndarray): Map between old and new faces.
            delete_faces[i] is replaced by
            new_faces[in_combined[i]:in_combined[i+1]].
        fn_orig (np.ndarray): Face-node relation of the orginial grid, before
            update of faces.
        node_coord_orig (np.ndarray): Node coordinates of orginal grid,
            before update of nodes.
        tol (double, defaults to 1e-4): Small tolerance, used to compare
            coordinates of points.

    """

    #

    nodes_per_face = g.dim

    cell_faces = g.cell_faces

    # Mapping from new
    deleted_2_new_faces = np.empty(in_combined.size - 1, dtype=object)

    # The nodes in the original 1d grid was sorted either in the same way, or
    # in the oposite order of the new grid. In the latter case, we need to
    # reverse the order of in_combined to reconstruct the old face-node
    # relations
    if in_combined[0] < in_combined[-1]:
        for i in range(deleted_2_new_faces.size):
            if in_combined[i] == in_combined[i + 1]:
                deleted_2_new_faces[i] = new_faces[in_combined[i]]
            else:
                deleted_2_new_faces[i] = new_faces[
                    np.arange(in_combined[i], in_combined[i + 1])
                ]
    #            assert deleted_2_new_faces[i].size > 0, \
    #                str(i)+" "+str(in_combined[i])+" "+str(in_combined[i+1])+\
    #                " "+str(np.arange(in_combined[i], in_combined[i+1]))
    else:
        for i in range(deleted_2_new_faces.size):
            if in_combined[i] == in_combined[i + 1]:
                deleted_2_new_faces[i] = new_faces[in_combined[i]]
            else:
                deleted_2_new_faces[i] = new_faces[
                    np.arange(in_combined[i + 1], in_combined[i])
                ]
    #            assert deleted_2_new_faces[i].size > 0, \
    #                str(i)+" "+str(in_combined[i+1])+" "+str(in_combined[i])+\
    #                " "+str(np.arange(in_combined[i+1], in_combined[i]))

    # Now that we have mapping from old to new faces, also update face tags
    update_face_tags(g, delete_faces, deleted_2_new_faces)

    # The cell-face relations
    cf = cell_faces.indices
    indptr = cell_faces.indptr

    # Find elements in the cell-face relation that are also along the
    # intersection, and should be replaced
    hit = np.where(np.in1d(cf, delete_faces))[0]

    # Mapping from cell_face of 2d grid to cells in 1d grid. Can be combined
    # with deleted_2_new_faces to match new and old faces
    # Safeguarding (or stupidity?): Only faces along 1d grid have non-negative
    # index, but we should never hit any of the other elements
    cf_2_f = -np.ones(delete_faces.max() + 1, dtype=np.int)
    cf_2_f[delete_faces] = np.arange(delete_faces.size)

    # Map from faces, as stored in cell_faces,to the corresponding cells
    face_2_cell = rldecode(np.arange(indptr.size), np.diff(indptr))

    # The cell-face map will go from 3 faces per cell to an arbitrary number.
    # Split mapping into list of arrays to prepare for this
    new_cf = [cf[indptr[i] : indptr[i + 1]] for i in range(g.num_cells)]
    # Similar treatment of direction of normal vectors
    new_sgn = [g.cell_faces.data[indptr[i] : indptr[i + 1]] for i in range(g.num_cells)]

    # Create mapping to adjust face indices for deletions
    tmp = np.arange(cf.max() + 1)
    adjust_deleted = np.zeros_like(tmp)
    adjust_deleted[delete_faces] = 1
    face_adjustment = tmp - np.cumsum(adjust_deleted)

    # Face-node relations as array
    fn = g.face_nodes.indices.reshape((nodes_per_face, g.num_faces), order="F")

    # Collect indices of cells that have one of their faces on the fracture.
    hit_cell = []

    for i in hit:
        # The loop variable refers to indices in the face-cell map. Get cell
        # number.
        cell = face_2_cell[i]
        hit_cell.append(cell)
        # For this cell, find where in the cell-face map the fracture face is
        # placed.
        tr = np.where(new_cf[cell] == cf[i])[0]
        # There should be only one face on the fracture
        assert tr.size == 1
        tr = tr[0]

        # Implementation note: If we ever get negative indices here, something
        # has gone wrong related to cf_2_f, see above.
        # Digestion of loop: i (in hit) refers to elements in cell-face
        # cf[i] is specific face
        # cf_2_f[cf[i]] maps to deleted face along fracture
        # outermost is one-to-many map from deleted to new faces.
        new_faces_loc = deleted_2_new_faces[cf_2_f[cf[i]]]

        # Index of the replaced face
        ci = cf[i]

        # We need to sort the new face-cell relation so that the edges defined
        # by cell-face-> face_nodes form a closed, non-intersecting loop. If
        # this is not the case, geometry computation will go wrong.
        # By assumption, the new faces are defined so that their nodes are
        # contiguous along the line of the old face.

        # Coordinates of the nodes of the replaced face.
        # Note use of original coordinates here.
        ci_coord = node_coord_orig[:, fn_orig[:, ci]]
        # Coordinates of the nodes of the first new face
        fi_coord = g.nodes[:, fn[:, new_faces_loc[0]]]

        # Distance between the new nodes and the first node of the old face.
        dist = cg.dist_point_pointset(ci_coord[:, 0], fi_coord)
        # Length of the old face.
        length_face = cg.dist_point_pointset(ci_coord[:, 0], ci_coord[:, 1])[0]
        # If the minimum distance is larger than a (scaled) tolerance, the new
        # faces were defined from the second to the first node. Switch order.
        # This will create trouble if one of the new faces are very small.
        if dist.min() > length_face * tol:
            new_faces_loc = new_faces_loc[::-1]

        # Replace the cell-face relation for this cell.
        # At the same time (stupid!) also adjust indices of the surviving
        # faces.
        new_cf[cell] = np.hstack(
            (
                face_adjustment[new_cf[cell][:tr].ravel()],
                new_faces_loc,
                face_adjustment[new_cf[cell][tr + 1 :].ravel()],
            )
        )
        # Also replicate directions of normal vectors
        new_sgn[cell] = np.hstack(
            (
                new_sgn[cell][:tr].ravel(),
                np.tile(new_sgn[cell][tr], new_faces_loc.size),
                new_sgn[cell][tr + 1 :].ravel(),
            )
        )

    # Adjust face index of cells that have no contact with the updated faces
    for i in np.setdiff1d(np.arange(len(new_cf)), hit_cell):
        new_cf[i] = face_adjustment[new_cf[i]]

    # New pointer structure for cell-face relations
    num_cell_face = np.array([new_cf[i].size for i in range(len(new_cf))])
    indptr_new = np.hstack((0, np.cumsum(num_cell_face)))

    ind = np.concatenate(new_cf)
    data = np.concatenate(new_sgn)
    # All faces in the cell-face relation should be referred to by 1 or 2 cells
    assert np.bincount(ind).max() <= 2
    assert np.all(np.bincount(ind) > 0)

    g.cell_faces = sps.csc_matrix((data, ind, indptr_new))


def update_face_tags(g, delete_faces, new_faces):
    """ Update the face tags of a cell.

    Delete tags for old faces, and add new tags for their replacements.

    If the grid has no face tags, no change is done

    Parameters:
        g (grid): To be modified
        delete_faces (np.array or list): Faces to be deleted.
        new_faces (list of list): For each item in delete_faces, a list of new
            replacement faces.

    """
    keys = tags.standard_face_tags()
    for key in keys:
        if hasattr(g, "tags"):
            old_tags = g.tags[key].copy()
            old_tags = np.delete(old_tags, delete_faces)
            num_new = np.array([len(new_faces[i]) for i in range(len(new_faces))])
            new_tags = np.zeros(num_new.sum(), dtype=bool)
            divides = np.hstack((0, np.cumsum(num_new)))
            for i, d in enumerate(delete_faces):
                new_tags[divides[i] : divides[i + 1]] = g.tags[key][d]
            g.tags[key] = np.hstack((old_tags, new_tags))
