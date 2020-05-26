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

import porepy as pp


def _update_mortar_grid(mg, new_side_grids, tol):
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
        if g.dim != new_g.dim:
            raise ValueError("Grid dimension has to be the same")

        if g.dim == 0:
            # Nothing to do
            return
        elif g.dim == 1:
            split_matrix[side] = _split_matrix_1d(g, new_g, tol)
        elif g.dim == 2:
            split_matrix[side] = _split_matrix_2d(g, new_g, tol)
        else:
            # No 3d mortar grid
            raise ValueError

    # Update the mortar grid class
    mg.update_mortar(split_matrix, new_side_grids)


# ------------------------------------------------------------------------------#


def _update_physical_low_grid(mg, new_g, tol):
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
        if g.dim != new_g.dim:
            raise ValueError("Grid dimension has to be the same")

        if mg.dim == 0:
            # Nothing to do
            return
        elif mg.dim == 1:
            split_matrix[side] = _split_matrix_1d(g, new_g, tol).T
        elif mg.dim == 2:
            split_matrix[side] = _split_matrix_2d(g, new_g, tol).T
        else:
            # No 3d mortar grid
            raise ValueError

    # Update the mortar grid class
    mg.update_slave(split_matrix)


def _update_physical_high_grid(mg, g_new, g_old, tol):

    mg.update_master(split_matrix)


# ------------------------------------------------------------------------------#


def _replace_grids_in_bucket(gb, g_map=None, mg_map=None, tol=1e-6):
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
    if mg_map is None:
        mg_map = {}

    # refine the mortar grids when specified
    for mg_old, mg_new in mg_map.items():
        update_mortar_grid(mg_old, mg_new, tol)

    # update the grid bucket considering the new grids instead of the old one
    # valid only for physical grids and not for mortar grids
    if g_map is not None:
        gb.update_nodes(g_map)
    else:
        g_map = {}

    # refine the grids when specified
    for g_old, g_new in g_map.items():
        for _, d in gb.edges_of_node(g_new):
            mg = d["mortar_grid"]
            if mg.dim == g_new.dim:
                # update the mortar grid of the same dimension
                update_physical_low_grid(mg, g_new, tol)
            else:  # g_new.dim == mg.dim + 1
                update_physical_high_grid(mg, g_new, g_old, tol)

    return gb
