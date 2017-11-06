#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:17:04 2017

@author: eke001
"""

import numpy as np
import scipy.sparse as sps

from porepy import Fracture, FractureNetwork

from porepy.fracs import importer
from porepy.fracs import simplex, meshing, split_grid


import porepy.utils.comp_geom as cg
from porepy.utils.setmembership import unique_columns_tol, ismember_rows
import porepy

from porepy.fracs import meshing
from porepy import TensorGrid


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
        np.array (int): Indices of common nodes of g and the new grid.
        np.array (int): Indices of common nodes of h and the new grid.

    """

    # Rotate meshes into natural 1d coordinates.
    _, _, _, g_rot, g_rot_dim, g_node = cg.map_grid(g)
    _, _, _, h_rot, h_rot_dim, h_node = cg.map_grid(h)

    # Do a sorting of 1d coordinates
    g_sort = np.argsort(g_node[0])
    h_sort = np.argsort(h_node[0])
    gx = g_node[0][g_sort]
    hx = h_node[0][h_sort]
    # Translate h grid so that their starting point are the same.
    # Might not be necessary, depending on how map_grid is behaving, but we do
    # it anyhow.
    hx += (gx[0]-hx[0])

    # Sanity test: End points of 1d grids should be equal
    assert np.abs(gx[0] - hx[0]) < tol
    assert np.abs(gx[-1] - hx[-1]) < tol

    # Combined coordinates along 1d line
    combined = np.hstack((gx[0], gx[1:-1], hx[1:-1], gx[-1]))
    combined_sort = np.argsort(combined)

    # We know where we put the coordinates of g grid, find out where they ended
    # after sorting
    g_ind = np.hstack((0, 1 + np.arange(gx.size-2), combined.size-1))
    g_in_combined = np.where(np.in1d(combined_sort, g_ind))[0]

    # Similar with the h grid
    h_ind = np.hstack((0, g_ind[-2] + 1 + np.arange(hx.size-2),
                       combined.size-1))
    h_in_combined = np.where(np.in1d(combined_sort, h_ind))[0]

    combined_nodes = combined[combined_sort]
    # Use unique nodes along the line
    combined_nodes, all_2_unique, unique_2_all \
        = unique_columns_tol(combined_nodes, tol)
    combined_nodes = combined_nodes[0]

    # Update maps between grids
    g_in_combined = unique_2_all[g_in_combined]
    h_in_combined = unique_2_all[h_in_combined]

    # Create a new 1d grid. Default coordinates for now, will be changed
    new_grid = TensorGrid(combined_nodes)

    # Distance along combined nodes
    d_cn = combined_nodes[-1] - combined_nodes[0]
    # Normalized length along line
    t = (combined_nodes - combined_nodes[0]) / d_cn

    # Find the start node - can be first or last in g
    # The vector along the line is recovered from g
    if combined_sort[0] == 0:
        d_real = g.nodes[:, -1] - g.nodes[:, 0]
        start_node = g.nodes[:, 0].reshape((-1, 1))
    else:
        d_real = g.nodes[:, 0] - g.nodes[:, -1]
        start_node = g[:, -1].reshape((-1, 1))
    # We can finally set the nodes of the 1d grid
    new_grid.nodes = start_node + t * d_real.reshape((-1, 1))

    # Define global points indices to new nodes
    new_global_pts = global_ind_offset + np.arange(new_grid.num_nodes)
    global_ind_offset += new_global_pts.size

    # Assign global indices to 1d mesh
    new_grid.global_point_ind = new_global_pts

    return new_grid, global_ind_offset, g_in_combined, h_in_combined