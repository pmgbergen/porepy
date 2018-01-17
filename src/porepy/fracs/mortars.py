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

#------------------------------------------------------------------------------#

def update_mortar_grid(mg, new_side_grids):
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
            split_matrix[side] = split_matrix_1d(g, new_g)
        elif g.dim == 2:
            split_matrix[side] = split_matrix_2d(g, new_g)
        else:
            # No 3d mortar grid
            raise ValueError

        mg.side_grids[side] = new_g.copy()

    # Update the mortar grid class
    mg.update_mortar(split_matrix)

#------------------------------------------------------------------------------#

def update_physical_low_grid(mg, new_g):
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
            split_matrix[side] = split_matrix_1d(g, new_g).T
        elif mg.dim == 2:
            split_matrix[side] = split_matrix_2d(g, new_g).T
        else:
            # No 3d mortar grid
            raise ValueError

    # Update the mortar grid class
    mg.update_low(split_matrix)

#------------------------------------------------------------------------------#

def update_physical_high_grid(mg, g_new, g_old, tol):

    split_matrix = {}

    if mg.dim == 0:

        # retrieve the old faces and the corresponding coordinates
        _, old_faces, _ = sps.find(mg.high_to_mortar)
        old_nodes = g_old.face_centers[:, old_faces]

        # retrieve the boundary faces and the corresponding coordinates
        new_faces = g_new.get_boundary_faces()
        new_nodes = g_new.face_centers[:, new_faces]

        for side, g in mg.side_grids.items():
            # we assume only one old node
            mask = cg.dist_point_pointset(old_nodes, new_nodes) < tol
            new_faces = new_faces[mask]

            shape = (g_old.num_faces, g_new.num_faces)
            data = np.ones(old_faces.shape)
            split_matrix[side] = sps.csc_matrix((data, (old_faces, new_faces)),
                                                                    shape=shape)

    elif mg.dim == 1:
        # The case is conceptually similar to 0d, but quite a bit more
        # technical. Implementation is moved to separate function
        split_matrix = _match_grids_along_line_from_geometry(mg, g_new, g_old, tol)

    else: # should be mg.dim == 2
        # It should be possible to use essentially the same approach as in 1d,
        # but this is not yet covered.
        raise NotImplementedError('Have not yet implemented this.')

    mg.update_high(split_matrix)

#------------------------------------------------------------------------------#

def split_matrix_1d(g_old, g_new):
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
    weights, new_cells, old_cells = match_grids_1d(g_new, g_old)
    return sps.csr_matrix((weights, (new_cells, old_cells)))

#------------------------------------------------------------------------------#

def split_matrix_2d(g_old, g_new):
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
    weights, new_cells, old_cells = match_grids_2d(g_new, g_old)
    return sps.csr_matrix((weights, (new_cells, old_cells)))

#------------------------------------------------------------------------------#

def match_grids_1d(new_1d, old_1d):
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
    combined, _, new_ind, old_ind, _, _ = \
         non_conforming.merge_1d_grids(new_1d, old_1d)
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

#------------------------------------------------------------------------------#

def match_grids_2d(new_g, old_g):
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
    shape = (new_g.dim+1, new_g.num_cells)
    cn_new_g = new_g.cell_nodes().indices.reshape(shape, order='F')

    shape = (old_g.dim+1, old_g.num_cells)
    cn_old_g = old_g.cell_nodes().indices.reshape(shape, order='F')

    isect = cg.intersect_triangulations(new_g.nodes[:2], old_g.nodes[:2],
                                        cn_new_g, cn_old_g)

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

#------------------------------------------------------------------------------#

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
    # Tolerance used in geometric comparisons. May be critical for performance,
    # in particular for bad grids
    tol = 1e-4

    #gb = gb.copy() nope it's not workign with this

    # refine the mortar grids when specified
    for mg_old, mg_new in mg_map.items():
        update_mortar_grid(mg_old, mg_new)

    # refine the grids when specified
    for g_old, g_new in g_map.items():
        gb.update_nodes(g_old, g_new)

        for e, d in gb.edges_props_of_node(g_new):
            mg = d['mortar_grid']
            if mg.dim == g_new.dim:
                # update the mortar grid of the same dimension
                update_physical_low_grid(mg, g_new)
            else: # g_new.dim == mg.dim + 1
                update_physical_high_grid(mg, g_new, g_old, tol)

    return gb

#----------------- Helper function below

def _match_grids_along_line_from_geometry(mg, g_new, g_old, tol):

    def cells_from_faces(g, fi):
        # Find cells of faces, specified by face indices fi.
        # It is assumed that fi is on the boundary, e.g. there is a single
        # cell for each element in fi.
        f, ci, _ = sps.find(g.cell_faces[fi])
        assert f.size == fi.size, 'We assume fi are boundary faces'

        ismem, ind_map = ismember_rows(fi[f], fi, sort=False)
        assert np.all(ismem)
        return ci[ind_map]

    def create_1d_from_nodes(nodes):
        assert cg.is_collinear(nodes, tol=tol)
        sort_ind = cg.argsort_point_on_line(nodes, tol=tol)
        n = nodes[:, sort_ind]
        unique_nodes, _, _ = unique_columns_tol(n, tol=tol)
        g = TensorGrid(np.arange(unique_nodes.shape[1]))
        g.nodes = unique_nodes
        g.compute_geometry()
        return g

    def nodes_of_faces(g, fi):
        f = np.zeros(g.num_faces)
        f[fi] = 1
        nodes = np.where(g.face_nodes * f > 0)[0]
        return nodes

    # First create a virtual 1d grid along the line, using nodes from the old grid
    # Identify faces in the old grid that is on the boundary
    _, faces_on_boundary, _ = sps.find(mg.high_to_mortar)
    # Find the nodes of those faces
    nodes_on_boundary = nodes_of_faces(g_old, faces_on_boundary)
    nodes_1d_old = g_old.nodes[:, nodes_on_boundary]

    # Normal vector of the line. Somewhat arbitrarily chosen as the first one.
    # This may be prone to rounding errors.
    normal = g_old.face_normals[:, faces_on_boundary[0]].reshape((3, 1))

    # Create first version of 1d grid, we really only need start and endpoint
    g_aux = create_1d_from_nodes(nodes_1d_old)

    start = g_aux.nodes[:, 0]
    end = g_aux.nodes[:, -1]

    mp = 0.5 * (start + end).reshape((3, 1))

    bound_cells_old = cells_from_faces(g_old, faces_on_boundary)
    assert bound_cells_old.size > 1, 'Have not implemented this. Not difficult though'
    cc_old = g_old.cell_centers[:, bound_cells_old]
    side_old = np.sign(np.sum(((cc_old - mp) * normal), axis=0))

    # Find cells on the positive and negative side, relative to the positioning
    # in cells_from_faces
    pos_side_old = np.where(side_old > 0)[0]
    neg_side_old = np.where(side_old < 0)[0]
    assert pos_side_old.size + neg_side_old.size == side_old.size

    if mg.num_sides() == 2:
        # If mg has two sides, each side must map to at least one face in
        # mg.high_to_mortar. By construction (or assumed construction), the
        # LEFT side is always added first, and thus the first column is
        # associated with this mortar grid. Below, we need to know match
        # positive and negative sides with LEFT and RIGHT. This should do the
        # trick
        pos_is_left = side_old[0] > 0

    elif mg.num_sides() == 1:
        raise NotImplementedError('Not sure about this one')

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
    fn = g_new.face_nodes.indices.reshape((2, g_new.num_faces),
                                          order='F')
    fn_in_hit = np.isin(fn, hit)
    # Faces where all points are found in hit
    faces_by_hit = np.where(np.all(fn_in_hit, axis=0))[0]
    faces_on_boundary_new = g_new.get_boundary_faces()
    # Only consider faces both in hit, and that are boundary
    faces_on_line_new = np.intersect1d(faces_by_hit,
                                       faces_on_boundary_new)

    bound_cells_new = cells_from_faces(g_new, faces_on_line_new)
    assert bound_cells_new.size > 1, 'Have not implemented this. Not difficult though'
    cc_new = g_new.cell_centers[:, bound_cells_new]
    side_new = np.sign(np.sum(((cc_new - mp) * normal), axis=0))

    pos_side_new = np.where(side_new > 0)[0]
    neg_side_new = np.where(side_new < 0)[0]
    assert pos_side_new.size + neg_side_new.size == side_new.size

    both_sides_new = [pos_side_new, neg_side_new]

    split_matrix = {}

    for i, (so, sn) in enumerate(zip(both_sides_old, both_sides_new)):
        loc_faces = faces_on_boundary[so]
        loc_nodes = nodes_of_faces(g_old, loc_faces)
        g_aux_old = create_1d_from_nodes(g_old.nodes[:, loc_nodes])

        # We have the new nodes on the line
        nodes_on_line_new = np.unique(fn[:, faces_on_line_new[sn]])

        g_aux_new = create_1d_from_nodes(nodes_new[:, nodes_on_line_new])

        if mg.num_sides() == 2:
            if pos_is_left:
                if i == 0:
                    side = mortar_grid.SideTag.LEFT
                else:  # i == 1
                    side = mortar_grid.SideTag.RIGHT
            else:
                if i == 0:
                    side = mortar_grid.SideTag.RIGHT
                else:  # i == 1
                    side = mortar_grid.SideTag.LEFT
        # Match grids, and create mapping between the cells
        split_matrix[side] = split_matrix_1d(g_aux_old, g_aux_new)

    return split_matrix

#------------------------------------------------------------------------------#
