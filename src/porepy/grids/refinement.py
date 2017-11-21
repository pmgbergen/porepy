#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various methods to refine a grid.

Created on Sat Nov 11 17:06:37 2017

@author: Eirik Keilegavlen
"""
import numpy as np
import scipy.sparse as sps

from porepy.grids.structured import TensorGrid
from porepy.utils import comp_geom as cg

def distort_grid_1d(g, ratio=0.1):
     """ Randomly distort internal nodes in a 1d grid.

     The boundary nodes are left untouched.

     The perturbations will not perturb the topology of the mesh.

     Parameters:
          g (grid): To be perturbed. Modifications will happen in place.
          ratio (optional, defaults to 0.1): Perturbation ratio. A node can be
               moved at most half the distance in towards any of its
               neighboring nodes. The ratio will multiply the chosen
               distortion. Should be less than 1 to preserve grid topology.

     Returns:
          grid: With distorted nodes
     """
     g.compute_geometry()
     r = ratio * (0.5 - np.random.random(g.num_nodes - 2))
     r *= np.minimum(g.cell_volumes[:-1], g.cell_volumes[1:])
     direction = (g.nodes[:, -1] - g.nodes[:, 0]).reshape((-1, 1))
     nrm = np.linalg.norm(direction)
     g.nodes[:, 1:-1] += r * direction / nrm
     g.compute_geometry()
     return g

def refine_grid_1d(g, ratio=2, return_map_nodes=False):
    """ Refine cells in a 1d grid.

    Note: The method cannot refine without
    """
    nodes, cells, _ = sps.find(g.cell_nodes())

    # To consider also the case of intersections
    num_new = (ratio-1)*g.num_cells
    x = np.zeros((3, num_new))
    theta = np.arange(1, ratio)/float(ratio)
    pos = 0

    for c in np.arange(g.num_cells):
        mask = cells == c
        start, end = nodes[mask]
        x_loc = g.nodes[:, start, np.newaxis]*theta + \
                g.nodes[:, end, np.newaxis]*(1-theta)
        x[:, pos:(pos+ratio-1)] = x_loc
        pos += ratio-1

    face_tags = np.hstack((g.face_tags, np.zeros(x.shape[1], dtype=np.int)))
    x = np.hstack((g.nodes, x))

    g = TensorGrid(x[0, :])
    map_nodes = cg.argsort_point_on_line(x)
    g.nodes = x[:, map_nodes]
    g.face_tags = face_tags[map_nodes]
    g.compute_geometry()

    if return_map_nodes:
        return g, np.argsort(map_nodes)[:nodes.size]
    else:
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
