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

#------------------------------------------------------------------------------#

def update_gb_1d(gb, gs_1d, hs_1d):

    gs_1d = np.atleast_1d(gs_1d)
    hs_1d = np.atleast_1d(hs_1d)

    assert gs_1d.size == hs_1d.size

    for g, h in zip(gs_1d, hs_1d):
        weights, new_cells, old_cells = match_grids_1d(h, g)
        split_matrix = sps.csr_matrix((weights, (new_cells, old_cells)))

        for g_2d in gb.node_neighbors(g, lambda _g: _g.dim > g.dim):
            face_cells = split_matrix * gb.edge_prop((g_2d, g), "face_cells")[0]
            gb.add_edge_prop("face_cells", (g_2d, g), face_cells)

    gb.update_nodes(gs_1d, hs_1d)

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

