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

#------------------------------------------------------------------------------#

def refine_mortar(mg, new_side_grids):
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
    for side, g in mg.side_grids.items():
        new_g = new_side_grids[side]
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
    mg.refine_mortar(split_matrix)

#------------------------------------------------------------------------------#

def refine_co_dimensional_grid(mg, new_g):
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
    mg.refine_low(split_matrix)

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

def replace_grids_in_bucket(gb, g_map={}, mg_map={}):
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
    #gb = gb.copy() nope it's not workign with this

    for g_old, g_new in g_map.items():
        gb.update_nodes(g_old, g_new)

        for e, d in gb.edges_props_of_node(g_new):
            mg = d['mortar_grid']
            if mg.dim == g_new.dim:
                refine_co_dimensional_grid(mg, g_new)
            else:
                pass

    for mg_old, mg_new in mg_map.items():
        refine_mortar(mg_old, mg_new)


#    for e, d in gb.edges_props():
#        print(d['mortar_grid'])

#    # Not sure if this is the best solution here.
#    for e, d in gb.edges():
#        mg_e = d['mortar_grid']
#        if mg_e in mg_map.keys():
#            d['mortar_grid'] = mg_map[mg_e]
#            edges_to_process.append(e)
#            nodes_to_process.append()

    return gb
    # Next step: Loop over nodes and edges to process, and update mortar maps as needed.

#------------------------------------------------------------------------------#
