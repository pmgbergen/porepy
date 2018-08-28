#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intended workflow for mortars:
    1) Create a full grid bucket of all objects to be included.
    2) Somehow create new grids for (some of) the nodes in the bucket.
    3) Find relation between new grid and neighboring grids.
    4) Replace nodes in the bucket
    5) Replace projection operators located on grid_bucket edges.

Implementation needs:
    1) Initialize identity (whatever that means) projections when a grid
       bucket is created.
    2) Create a framework for modifying grids. In the first stage this will
       involve perturbing nodes (not on the boundary). Second stage shoud be
       refinements of simplex grids. Partial remeshing with other parameters
       should also be on the list.
       -> Partly solved by grids.refinement.
    3) Methods to match cells and faces from different grids.
       -> First attempt in relate_1d_and_2d_grids()
    4) Creation of general projection matrices. Closely related to
    5) Numerical methods that can handle general projections.

Created on Sat Nov 11 16:22:36 2017

@author: Eirik Keilegavlen
"""

import numpy as np
import scipy.sparse as sps

from porepy.fracs import non_conforming
from porepy.utils.matrix_compression import rldecode
import porepy.utils.comp_geom as cg
from porepy.utils.setmembership import ismember_rows, unique_columns_tol
from porepy.grids.structured import TensorGrid
from porepy.grids import mortar_grid

# ------------------------------------------------------------------------------#


def update_mortar_grid(mg, new_side_grids, tol):
    """
    Update the maps in the mortar class when the mortar grids are changed.
    The update of the mortar grid is in-place.

    It is asumed that the grids are aligned, with common start and endpoints.

    Parameters:
        mg (MortarGrid): the mortar grid class to be updated
        new_side_grids (dictionary): for each SideTag key a new grid to be
            updated in the mortar grid class.
    """

    split_matrix = {}

    # For each side we compute the mapping between the old and the new mortar
    # grids, we store them in a dictionary with SideTag as key.
    for side, new_g in new_side_grids.items():
        g = mg.side_grids[side]
        assert g.dim == new_g.dim

        if g.dim == 0:
            # Nothing to do
            return
        elif g.dim == 1:
            split_matrix[side] = split_matrix_1d(g, new_g, tol)
        elif g.dim == 2:
            split_matrix[side] = split_matrix_2d(g, new_g, tol)
        else:
            # No 3d mortar grid
            raise ValueError

        mg.side_grids[side] = new_g.copy()

    # Update the mortar grid class
    mg.update_mortar(split_matrix)


# ------------------------------------------------------------------------------#


def update_physical_low_grid(mg, new_g, tol):
    """
    Update the maps in the mortar class when the lower dimensional grid is
    changed. The update of the lower dimensional grid in the grid bucket needs
    to be done outside.

    It is asumed that the grids are aligned (cover the same domain), with
    common start and endpoints. However, 1D grids need not be oriented in the
    same direction (e.g. from 'left' to 'right'), and no restrictions are
    placed on nodes on the 2D grid.

    Parameters:
        mg (MortarGrid): the mortar grid class to be updated
        new_g (Grid): the new lower dimensional grid.

    """
    split_matrix = {}

    # For each side we compute the mapping between the new lower dimensional
    # grid and the mortar grid, we store them in a dictionary with SideTag as key.
    for side, g in mg.side_grids.items():
        assert g.dim == new_g.dim

        if mg.dim == 0:
            # Nothing to do
            return
        elif mg.dim == 1:
            split_matrix[side] = split_matrix_1d(g, new_g, tol).T
        elif mg.dim == 2:
            split_matrix[side] = split_matrix_2d(g, new_g, tol).T
        else:
            # No 3d mortar grid
            raise ValueError

    # Update the mortar grid class
    mg.update_low(split_matrix)


# ------------------------------------------------------------------------------#


def update_physical_high_grid(mg, g_new, g_old, tol):

    split_matrix = {}

    if mg.dim == 0:

        # retrieve the old faces and the corresponding coordinates
        _, old_faces, _ = sps.find(mg.high_to_mortar_int)
        old_nodes = g_old.face_centers[:, old_faces]

        # retrieve the boundary faces and the corresponding coordinates
        new_faces = g_new.get_boundary_faces()
        new_nodes = g_new.face_centers[:, new_faces]

        # we assume only one old node
        mask = cg.dist_point_pointset(old_nodes, new_nodes) < tol
        new_faces = new_faces[mask]

        shape = (g_old.num_faces, g_new.num_faces)
        matrix_DIJ = (np.ones(old_faces.shape), (old_faces, new_faces))
        split_matrix = sps.csc_matrix(matrix_DIJ, shape=shape)

    elif mg.dim == 1:
        # The case is conceptually similar to 0d, but quite a bit more
        # technical. Implementation is moved to separate function
        split_matrix = _match_grids_along_line_from_geometry(mg, g_new, g_old, tol)

    else:  # should be mg.dim == 2
        # It should be possible to use essentially the same approach as in 1d,
        # but this is not yet covered.
        raise NotImplementedError("Have not yet implemented this.")

    mg.update_high(split_matrix)


# ------------------------------------------------------------------------------#


def split_matrix_1d(g_old, g_new, tol):
    """
    By calling matching grid the function compute the cell mapping between two
    different grids.

    It is asumed that the two grids are aligned, with common start and
    endpoints. However, their nodes can be ordered in oposite directions.

    Parameters:
        g_old (Grid): the first (old) grid
        g_new (Grid): the second (new) grid
    Return:
        csr matrix: representing the cell mapping. The entries are the relative
            cell measure between the two grids.

    """
    weights, new_cells, old_cells = match_grids_1d(g_new, g_old, tol)
    shape = (g_new.num_cells, g_old.num_cells)
    return sps.csr_matrix((weights, (new_cells, old_cells)), shape=shape)


# ------------------------------------------------------------------------------#


def split_matrix_2d(g_old, g_new, tol):
    """
    By calling matching grid the function compute the cell mapping between two
    different grids.

    It is asumed that the two grids have common boundary.

    Parameters:
        g_old (Grid): the first (old) grid
        g_new (Grid): the second (new) grid
    Return:
        csr matrix: representing the cell mapping. The entries are the relative
            cell measure between the two grids.

    """
    weights, new_cells, old_cells = match_grids_2d(g_new, g_old, tol)
    shape = (g_new.num_cells, g_old.num_cells)
    # EK: Is it really safe to use csr_matrix here?
    return sps.csr_matrix((weights, (new_cells, old_cells)), shape=shape)


# ------------------------------------------------------------------------------#


def match_grids_1d(new_1d, old_1d, tol):
    """ Obtain mappings between the cells of non-matching 1d grids.

    The function constructs an refined 1d grid that consists of all nodes
    of at least one of the input grids.

    It is asumed that the two grids are aligned, with common start and
    endpoints.

    Implementation note: It should be possible to avoid old_1d, by extracting
    points from a 2D grid that lie along the line defined by g_1d.
    However, if g_2d is split along a fracture, the nodes will be
    duplicated. We should then return two grids, probably based on the
    coordinates of the cell centers. sounds cool.

    Parameters:
         new_1d (grid): First grid to be matched
         old_1d (grid): Second grid to be matched.

    Returns:
         np.array: Ratio of cell volume in the common grid and the original grid.
         np.array: Mapping between cell numbers in common and first input
              grid.
         np.array: Mapping between cell numbers in common and second input
              grid.

    """

    # Create a grid that contains all nodes of both the old and new grids.
    combined, _, new_ind, old_ind, _, _ = non_conforming.merge_1d_grids(
        new_1d, old_1d, tol=tol
    )
    combined.compute_geometry()
    weights = combined.cell_volumes

    switch_new = new_ind[0] > new_ind[-1]
    if switch_new:
        new_ind = new_ind[::-1]
    switch_old = old_ind[0] > old_ind[-1]
    if switch_old:
        old_ind = old_ind[::-1]

    diff_new = np.diff(new_ind)
    diff_old = np.diff(old_ind)
    new_in_full = rldecode(np.arange(diff_new.size), diff_new)
    old_in_full = rldecode(np.arange(diff_old.size), diff_old)

    if switch_new:
        new_in_full = new_in_full.max() - new_in_full
    if switch_old:
        old_in_full = old_in_full.max() - old_in_full

    old_1d.compute_geometry()

    weights /= old_1d.cell_volumes[old_in_full]
    return weights, new_in_full, old_in_full


# ------------------------------------------------------------------------------#


def match_grids_2d(new_g, old_g, tol):
    """ Match two simplex tessalations to identify overlapping cells.

    The overlaps are identified by the cell index of the two overlapping cells,
    and their weighted common area.

    Parameters:
        new_g: simplex grid of dimension 2.
        old_g: simplex grid of dimension 2.

    Returns:
        np.array: Ratio of cell volume in the common grid and the original grid.
        np.array: Index of overlapping cell in the first grid.
        np.array: Index of overlapping cell in the second grid.

    """

    def proj_pts(p, cc):
        rot = cg.project_plane_matrix(p - cc)
        return rot.dot(p - cc)[:2]

    shape = (new_g.dim + 1, new_g.num_cells)
    cn_new_g = new_g.cell_nodes().indices.reshape(shape, order="F")

    shape = (old_g.dim + 1, old_g.num_cells)
    cn_old_g = old_g.cell_nodes().indices.reshape(shape, order="F")
    cc = np.mean(new_g.nodes, axis=1).reshape((3, 1))
    isect = cg.intersect_triangulations(
        proj_pts(new_g.nodes, cc), proj_pts(old_g.nodes, cc), cn_new_g, cn_old_g
    )

    num = len(isect)
    new_g_ind = np.zeros(num, dtype=np.int)
    old_g_ind = np.zeros(num, dtype=np.int)
    weights = np.zeros(num)

    for ind, i in enumerate(isect):
        new_g_ind[ind] = i[0]
        old_g_ind[ind] = i[1]
        weights[ind] = i[2]

    weights /= old_g.cell_volumes[old_g_ind]
    return weights, new_g_ind, old_g_ind


# ------------------------------------------------------------------------------#


def replace_grids_in_bucket(gb, g_map={}, mg_map={}, tol=1e-6):
    """ Replace grids and / or mortar grids in a grid_bucket. Recompute mortar
    mappings as needed.

    NOTE: These are implementation notes for an unfinished implementation.

    Parameters:
        gb (GridBucket): To be updated.
        g_map (dictionary): Grids to replace. Keys are grids in the old bucket,
            values are their replacements.
        mg_map (dictionary): Mortar grids to replace. Keys are EITHER related
            to mortar grids, or to edges. Probably, mg is most relevant, the we
            need to identify the right edge shielded from user.

    Returns:
        GridBucket: New grid bucket, with all relevant replacements. Not sure
            how deep the copy should be - clearly a new graph, nodes and edges
            replaced, but can we keep untouched grids?

    """
    # refine the mortar grids when specified
    for mg_old, mg_new in mg_map.items():
        update_mortar_grid(mg_old, mg_new, tol)

    # refine the grids when specified
    for g_old, g_new in g_map.items():
        gb.update_nodes(g_old, g_new)

        for e, d in gb.edges_of_node(g_new):
            mg = d["mortar_grid"]
            if mg.dim == g_new.dim:
                # update the mortar grid of the same dimension
                update_physical_low_grid(mg, g_new, tol)
            else:  # g_new.dim == mg.dim + 1
                update_physical_high_grid(mg, g_new, g_old, tol)

    return gb


# ----------------- Helper function below


def _match_grids_along_line_from_geometry(mg, g_new, g_old, tol):

    # The purpose of this function is to construct a mapping between faces in
    # the old and new grid. Specifically, we need to match faces that lies on
    # the 1d segment identified by the mortar grid, and get the right area
    # weightings when the two grids do not conform.
    #
    # The algorithm is technical, partly because we also need to differ between
    # the left and right side of the segment, as these will belong to different
    # mortar grids.
    #
    # The main steps are:
    #   1) Identify faces in the old grid along the segment via the existing
    #      mapping between mortar grid and higher dimensional grid. Use this
    #      to define the geometry of the segment.
    #   2) Define positive and negative side of the segment, and split cells
    #      and faces along the segement according to this criterion.
    #   3) For all sides (pos, neg), pick out faces in the old and new grid,
    #      and match them up. Extend the mapping to go from all faces in the
    #      two grids.
    #
    # Known weak points: Identification of geometric objects, in particular
    # points, is based on a geometric tolerance. For very fine, or bad, grids
    # this may give trouble.

    def cells_from_faces(g, fi):
        # Find cells of faces, specified by face indices fi.
        # It is assumed that fi is on the boundary, e.g. there is a single
        # cell for each element in fi.
        f, ci, _ = sps.find(g.cell_faces[fi])
        assert f.size == fi.size, "We assume fi are boundary faces"

        ismem, ind_map = ismember_rows(fi, fi[f], sort=False)
        assert np.all(ismem)
        return ci[ind_map]

    def create_1d_from_nodes(nodes):
        # From a set of nodes, create a 1d grid. duplicate nodes are removed
        # and we verify that the nodes are indeed colinear
        assert cg.is_collinear(nodes, tol=tol)
        sort_ind = cg.argsort_point_on_line(nodes, tol=tol)
        n = nodes[:, sort_ind]
        unique_nodes, _, _ = unique_columns_tol(n, tol=tol)
        g = TensorGrid(np.arange(unique_nodes.shape[1]))
        g.nodes = unique_nodes
        g.compute_geometry()
        return g, sort_ind

    def nodes_of_faces(g, fi):
        # Find nodes of a set of faces.
        f = np.zeros(g.num_faces)
        f[fi] = 1
        nodes = np.where(g.face_nodes * f > 0)[0]
        return nodes

    def face_to_cell_map(g_2d, g_1d, loc_faces, loc_nodes):
        # Match faces in a 2d grid and cells in a 1d grid by identifying
        # face-nodes and cell-node relations.
        # loc_faces are faces in 2d grid that are known to coincide with
        # cells.
        # loc_nodes are indices of 2d nodes along the segment, sorted so that
        # the ordering coincides with nodes in 1d grid

        # face-node relation in higher dimensional grid
        fn = g_2d.face_nodes.indices.reshape((g_2d.dim, g_2d.num_faces), order="F")
        # Reduce to faces along segment
        fn_loc = fn[:, loc_faces]
        # Mapping from global (2d) indices to the local indices used in 1d
        # grid. This also account for a sorting of the nodes, so that the
        # nodes.
        ind_map = np.zeros(g_2d.num_faces)
        ind_map[loc_nodes] = np.arange(loc_nodes.size)
        # Face-node in local indices
        fn_loc = ind_map[fn_loc]
        # Handle special case
        if loc_faces.size == 1:
            fn_loc = fn_loc.reshape((2, 1))

        # Cell-node relation in 1d
        cn = g_1d.cell_nodes().indices.reshape((2, g_1d.num_cells), order="F")

        # Find cell index of each face
        ismem, ind = ismember_rows(fn_loc, cn)
        # Quality check, the grids should be conforming
        assert np.all(ismem)
        return ind

    # First create a virtual 1d grid along the line, using nodes from the old grid
    # Identify faces in the old grid that is on the boundary
    _, faces_on_boundary_old, _ = sps.find(mg.high_to_mortar_int)
    # Find the nodes of those faces
    nodes_on_boundary_old = nodes_of_faces(g_old, faces_on_boundary_old)
    nodes_1d_old = g_old.nodes[:, nodes_on_boundary_old]

    # Normal vector of the line. Somewhat arbitrarily chosen as the first one.
    # This may be prone to rounding errors.
    normal = g_old.face_normals[:, faces_on_boundary_old[0]].reshape((3, 1))

    # Create first version of 1d grid, we really only need start and endpoint
    g_aux, _ = create_1d_from_nodes(nodes_1d_old)

    # Start, end and midpoint
    start = g_aux.nodes[:, 0]
    end = g_aux.nodes[:, -1]
    mp = 0.5 * (start + end).reshape((3, 1))

    # Find cells in 2d close to the segment
    bound_cells_old = cells_from_faces(g_old, faces_on_boundary_old)
    # This may occur if the mortar grid is one sided (T-intersection)
    #    assert bound_cells_old.size > 1, 'Have not implemented this. Not difficult though'
    # Vector from midpoint to cell centers. Check which side the cells are on
    # relative to normal vector.
    # We are here assuming that the segment is not too curved (due to rounding
    # errors). Pain to come.
    cc_old = g_old.cell_centers[:, bound_cells_old]
    side_old = np.sign(np.sum(((cc_old - mp) * normal), axis=0))

    # Find cells on the positive and negative side, relative to the positioning
    # in cells_from_faces
    pos_side_old = np.where(side_old > 0)[0]
    neg_side_old = np.where(side_old < 0)[0]
    assert pos_side_old.size + neg_side_old.size == side_old.size

    both_sides_old = [pos_side_old, neg_side_old]

    # Then virtual 1d grid for the new grid. This is a bit more involved,
    # since we need to identify the nodes by their coordinates.
    # This part will be prone to rounding errors, in particular for
    # bad cell shapes.
    nodes_new = g_new.nodes

    # Represent the 1d line by its start and end point, as pulled
    # from the old 1d grid (known coordinates)
    # Find distance from the
    dist, _ = cg.dist_points_segments(nodes_new, start, end)
    # Look for points in the new grid with a small distance to the
    # line
    hit = np.argwhere(dist.ravel() < tol).reshape((1, -1))[0]

    # Depending on geometric tolerance and grid quality, hit
    # may contain nodes that are close to the 1d line, but not on it
    # To improve the results, also require that the faces are boundary faces

    # We know we are in 2d, thus all faces have two nodes
    # We can do the same trick in 3d, provided we have simplex grids
    # but this will fail on Cartesian or polyhedral grids
    fn = g_new.face_nodes.indices.reshape((2, g_new.num_faces), order="F")
    fn_in_hit = np.isin(fn, hit)
    # Faces where all points are found in hit
    faces_by_hit = np.where(np.all(fn_in_hit, axis=0))[0]
    faces_on_boundary_new = np.where(g_new.tags["fracture_faces"].ravel())[0]
    # Only consider faces both in hit, and that are boundary
    faces_on_boundary_new = np.intersect1d(faces_by_hit, faces_on_boundary_new)

    # Cells along the segment, from the new grid
    bound_cells_new = cells_from_faces(g_new, faces_on_boundary_new)
    #    assert bound_cells_new.size > 1, 'Have not implemented this. Not difficult though'
    cc_new = g_new.cell_centers[:, bound_cells_new]
    side_new = np.sign(np.sum(((cc_new - mp) * normal), axis=0))

    pos_side_new = np.where(side_new > 0)[0]
    neg_side_new = np.where(side_new < 0)[0]
    assert pos_side_new.size + neg_side_new.size == side_new.size

    both_sides_new = [pos_side_new, neg_side_new]

    # Mapping matrix.
    matrix = sps.coo_matrix((g_old.num_faces, g_new.num_faces))

    for so, sn in zip(both_sides_old, both_sides_new):

        if sn.size == 0 or so.size == 0:
            continue
        # Pick out faces along boundary in old grid, uniquify nodes, and
        # define auxiliary grids
        loc_faces_old = faces_on_boundary_old[so]
        loc_nodes_old = np.unique(nodes_of_faces(g_old, loc_faces_old))
        g_aux_old, sort_ind_old = create_1d_from_nodes(g_old.nodes[:, loc_nodes_old])

        # Similar for new grid
        loc_faces_new = faces_on_boundary_new[sn]
        loc_nodes_new = np.unique(fn[:, loc_faces_new])
        g_aux_new, sort_ind_new = create_1d_from_nodes(nodes_new[:, loc_nodes_new])

        # Map from global faces to faces along segment in old grid
        n_loc_old = loc_faces_old.size
        face_map_old = sps.coo_matrix(
            (np.ones(n_loc_old), (np.arange(n_loc_old), loc_faces_old)),
            shape=(n_loc_old, g_old.num_faces),
        )

        # Map from global faces to faces along segment in new grid
        n_loc_new = loc_faces_new.size
        face_map_new = sps.coo_matrix(
            (np.ones(n_loc_new), (np.arange(n_loc_new), loc_faces_new)),
            shape=(n_loc_new, g_new.num_faces),
        )

        # Map from faces along segment in old to new grid. Consists of three
        # stages: faces in old to cells in 1d version of old, between 1d cells
        # in old and new, cells in new to faces in new

        # From faces to cells in old grid
        rows = face_to_cell_map(
            g_old, g_aux_old, loc_faces_old, loc_nodes_old[sort_ind_old]
        )
        cols = np.arange(rows.size)
        face_to_cell_old = sps.coo_matrix((np.ones(rows.size), (rows, cols)))

        # Mapping between cells in 1d grid
        weights, new_cells, old_cells = match_grids_1d(g_aux_new, g_aux_old, tol)
        between_cells = sps.csr_matrix((weights, (old_cells, new_cells)))

        # From faces to cell in new grid
        rows = face_to_cell_map(
            g_new, g_aux_new, loc_faces_new, loc_nodes_new[sort_ind_new]
        )
        cols = np.arange(rows.size)
        cell_to_face_new = sps.coo_matrix((np.ones(rows.size), (rows, cols)))

        face_map_segment = face_to_cell_old * between_cells * cell_to_face_new

        face_map = face_map_old.T * face_map_segment * face_map_new

        matrix += face_map

    return matrix.tocsr()


# ------------------------------------------------------------------------------#
