#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various methods to refine a grid.

Created on Sat Nov 11 17:06:37 2017

@author: Eirik Keilegavlen
"""
import numpy as np
import scipy.sparse as sps

from porepy.grids.grid import Grid
from porepy.grids.structured import TensorGrid
from porepy.grids.simplex import TriangleGrid, TetrahedralGrid
from porepy.utils import comp_geom as cg


def distort_grid_1d(g, ratio=0.1, fixed_nodes=None):
     """ Randomly distort internal nodes in a 1d grid.

     The boundary nodes are left untouched.

     The perturbations will not perturb the topology of the mesh.

     Parameters:
          g (grid): To be perturbed. Modifications will happen in place.
          ratio (optional, defaults to 0.1): Perturbation ratio. A node can be
               moved at most half the distance in towards any of its
               neighboring nodes. The ratio will multiply the chosen
               distortion. Should be less than 1 to preserve grid topology.
          fixed_nodes (np.array): Index of nodes to keep fixed under
              distortion. Boundary nodes will always be fixed, even if not
              expli)itly included as fixed_node

     Returns:
          grid: With distorted nodes

     """
     if fixed_nodes is None:
         fixed_nodes = np.array([0, g.num_nodes-1], dtype=np.int)
     else:
         # Ensure that boundary nodes are also fixed
         fixed_nodes = np.hstack((fixed_nodes, np.array([0, g.num_nodes - 1])))
         fixed_nodes = np.unique(fixed_nodes).astype(np.int)

     g.compute_geometry()
     r = ratio * (0.5 - np.random.random(g.num_nodes - 2))
     r *= np.minimum(g.cell_volumes[:-1], g.cell_volumes[1:])
     direction = (g.nodes[:, -1] - g.nodes[:, 0]).reshape((-1, 1))
     nrm = np.linalg.norm(direction)
     g.nodes[:, 1:-1] += r * direction / nrm
     g.compute_geometry()
     return g

def refine_grid_1d(g, ratio=2):
    """ Refine cells in a 1d grid.

    Note: The method cannot refine without
    """
    nodes, cells, _ = sps.find(g.cell_nodes())

    # To consider also the case of intersections
    num_new = (ratio-1)*g.num_cells+g.num_nodes
    x = np.zeros((3, num_new))
    theta = np.arange(1, ratio)/float(ratio)
    pos = 0
    shift = 0

    cell_nodes = g.cell_nodes()
    if_add = np.r_[1, np.ediff1d(cell_nodes.indices)].astype(np.bool)

    indices = np.empty(0, dtype=np.int)
    ind = np.vstack((np.arange(ratio), np.arange(ratio)+1)).flatten('F')

    for c in np.arange(g.num_cells):
        loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c+1])
        start, end = cell_nodes.indices[loc]
        if_add_loc = if_add[loc]
        indices = np.r_[indices, shift+ind]

        if if_add_loc[0]:
            x[:, pos:(pos+1)] = g.nodes[:, start, np.newaxis]
            pos += 1

        x[:, pos:(pos+ratio-1)] = g.nodes[:, start, np.newaxis]*theta + \
                                  g.nodes[:, end, np.newaxis]*(1-theta)
        pos += ratio-1
        shift += ratio+2-np.sum(if_add_loc)

        if if_add_loc[1]:
            x[:, pos:(pos+1)] = g.nodes[:, end, np.newaxis]
            pos += 1

    face_nodes = sps.identity(x.shape[1], format='csc')
    cell_faces = sps.csc_matrix((np.ones(indices.size, dtype=np.bool),
                                 indices, np.arange(0, indices.size+1, 2)))
    g = Grid(1, x, face_nodes, cell_faces, "Refined 1d grid")
    g.compute_geometry()

    return g

#------------------------------------------------------------------------------#

def new_grid_1d(g, num_nodes):

    theta = np.linspace(0, 1, num_nodes)
    start, end = g.get_boundary_nodes()

    nodes = g.nodes[:, start, np.newaxis]*theta + \
            g.nodes[:, end, np.newaxis]*(1.-theta)

    g = TensorGrid(nodes[0, :])
    g.nodes = nodes
    g.compute_geometry()

    return g

#------------------------------------------------------------------------------#
