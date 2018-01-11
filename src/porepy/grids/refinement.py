#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various methods to refine a grid.

Created on Sat Nov 11 17:06:37 2017

@author: Eirik Keilegavlen
"""
import numpy as np
import scipy.sparse as sps

from porepy.grids.grid import Grid, FaceTag
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
    nd = np.r_[np.diff(cell_nodes.indices)[1::2], 0]

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
        shift += ratio+(2-np.sum(if_add_loc)*(1-nd[c]))-nd[c]

        if if_add_loc[1]:
            x[:, pos:(pos+1)] = g.nodes[:, end, np.newaxis]
            pos += 1

    face_nodes = sps.identity(x.shape[1], format='csc')
    cell_faces = sps.csc_matrix((np.ones(indices.size, dtype=np.bool),
                                 indices, np.arange(0, indices.size+1, 2)))
    g = Grid(1, x, face_nodes, cell_faces, "Refined 1d grid")
    g.compute_geometry()

    return g

def refine_triangle_grid(g):
    """ Uniform refinement of triangle grid, all cells are split into four
    subcells by combining existing nodes and face centrers.

    Implementation note: It should be fairly straighforward to extend the
    function to 3D simplex grids as well. The loop over face combinations
    extends straightforwardly, but obtaining the node in the corner defined
    by faces may be a bit tricky.

    Parameters:
        g TriangleGrid. To be refined.

    Returns:
        TriangleGrid: New grid, with nd+2 times as many cells as g.
        np.array: Mapping from new to old cells.

    """
    # g needs to have face centers
    if not hasattr(g, 'face_centers'):
        g.compute_geometry()
    nd = g.dim

    # Construct dense versions of face-node and cell-face maps.
    # This will crash if a non-simplex grid is provided.
    fn = g.face_nodes.indices.reshape((nd, g.num_faces), order='F')
    cf = g.cell_faces.indices.reshape((nd+1, g.num_cells), order='F')

    new_nodes = np.hstack((g.nodes, g.face_centers))
    offset = g.num_nodes

    # Combinations of faces per cell
    binom = ((0, 1), (1, 2), (2, 0))

    # Holder for new tessalation.
    new_tri = np.empty(shape=(nd+1, g.num_cells, nd+2), dtype=np.int)

    # Loop over combinations
    for ti, b in enumerate(binom):
        # Find face-nodes of these faces. Each column corresponds to a single cell.
        # There should be one duplicate in each column (since this is 2D)
        loc_n = np.vstack((fn[:, cf[b[0]]], fn[:, cf[b[1]]]))
        # Find the duplicate: First sort along column, then take diff along column
        # and look for zero.
        # Implementation note: To extend to 3D, also require that np.gradient
        # is zero (that function should be able to do this).
        loc_n.sort(axis=0)
        equal = np.argwhere(np.diff(loc_n, axis=0) == 0)
        # equal is now 2xnum_cells. To pick out the right elements, consider
        # the raveled index, and construct the corresponding raveled array
        equal_n = loc_n.ravel()[np.ravel_multi_index(equal.T, loc_n.shape)]

        # Define node combination. Both nodes associated with a face have their
        # offset adjusted.
        new_tri[:, :, ti] = np.vstack((equal_n, offset + cf[b[0]],
                                       offset + cf[b[1]]))

    # Create final triangle by combining faces only
    new_tri[:, :, -1] = offset + cf

    # Reshape into 2d array
    new_tri = new_tri.reshape((nd+1, (nd+2) * g.num_cells))

    # The new grid inherits the history of the old one.
    name = g.name.copy()
    name.append('Refinement')

    # Also create mapping from refined to parent cells
    parent = np.tile(np.arange(g.num_cells), g.dim+2)

    return TriangleGrid(new_nodes, tri=new_tri, name=name), parent

#------------------------------------------------------------------------------#

def new_grid_1d(g, num_nodes):
    """ EK: This needs a better name!
    """

    theta = np.linspace(0, 1, num_nodes)
    start, end = g.get_boundary_nodes()

    nodes = g.nodes[:, start, np.newaxis]*theta + \
            g.nodes[:, end, np.newaxis]*(1.-theta)

    g = TensorGrid(nodes[0, :])
    g.nodes = nodes
    g.compute_geometry()

    return g

#------------------------------------------------------------------------------#
