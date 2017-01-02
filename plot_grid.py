# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 19:37:05 2016

@author: keile
"""

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import matplotlib.tri
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


from core.grids import simplex, structured
from compgeom import sort_points

#------------------------------------------------------------------------------#

def plot_grid(g, info = None, show=True):

#    if isinstance(g, simplex.TriangleGrid):
#        figId, ax = plot_tri_grid(g)
#    elif isinstance(g, structured.TensorGrid) and g.dim == 2:
#        figId, ax = plot_cart_grid_2d(g)
#    elif g.dim == 2:
    if g.dim == 2:
        figId, ax = plot_grid_2d(g)
    else:
        raise NotImplementedError('Under construction')

    if info is not None: add_info( g, info, figId, ax )
    if show: plt.show()

    return figId

#------------------------------------------------------------------------------#

def plot_tri_grid(g):
    """
    Plot triangular mesh using matplotlib.

    The function uses matplotlib's built-in methods for plotting of
    triangular meshes

    Examples:
    >>> x = np.arange(3)
    >>> y = np.arange(2)
    >>> g = simplex.StructuredTriangleGrid(x, y)
    >>> plot_tri_grid(g)

    Parameters
    ----------
    g

    """
    tri = g.cell_node_matrix()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    triang = matplotlib.tri.Triangulation(g.nodes[0], g.nodes[1], tri)
    ax.triplot(triang)
    return fig.number, ax

#------------------------------------------------------------------------------#

def plot_cart_grid_2d(g):
    """
    Plot quadrilateral mesh using matplotlib.

    The function uses matplotlib's bulit-in function pcolormesh

    For the moment, the cells have an ugly blue color.

    Examples:

    >>> g = structured.CartGrid(np.array([3, 4]))
    >>> plot_cart_grid_2d(g)

    Parameters
    ----------
    g grid to be plotted

    """

    # In each direction there is one more node than cell
    node_dims = g.cart_dims + 1

    x = g.nodes[0].reshape(node_dims)
    y = g.nodes[1].reshape(node_dims)

    # pcolormesh needs colors for its cells. Let the value be one
    z = np.ones(x.shape)

    fig = plt.figure()
    #ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax = fig.add_subplot(111)
    # It would have been better to have a color map which makes all
    ax.pcolormesh(x, y, z, edgecolors='k', cmap='gray', alpha=0.5)
    return fig.number, ax

#------------------------------------------------------------------------------#

def plot_grid_fractures(g, show=True):
    figId = plot_grid(g, show=False)
    fig = plt.figure( figId )
    face_info = g.face_info
    tags = face_info['tags']
    face_nodes = g.face_nodes.indices.reshape((2, -1), order='F')
    xf = g.nodes
    for ff in np.nditer(np.where(np.isfinite(tags))):
        fn_loc = face_nodes[:, ff]
        plt.plot([xf[0, fn_loc[0]], xf[0, fn_loc[1]]],
                 [xf[1, fn_loc[0]], xf[1, fn_loc[1]]], linewidth=4, color='k')
        # if show:
        #     plt.show()

#------------------------------------------------------------------------------#

def add_info( _g, _info, _figId, _ax ):
    def mask_index( _p ): return _p[0:2]
    def disp( _i, _p, _c ): _ax.plot( *_p, _c ); _ax.annotate( _i, _p )
    def disp_loop( _v, _c ): [ disp( i, c, _c ) for i, c in enumerate( _v.T ) ]

    fig = plt.figure( _figId )
    _info = _info.upper()

    if "C" in _info: disp_loop( mask_index( _g.cell_centers ), 'ro' )
    if "N" in _info: disp_loop( mask_index( _g.nodes ), 'bs' )
    if "F" in _info: disp_loop( mask_index( _g.face_centers ), 'yd' )

    if "O" in _info.upper():
        normals = np.divide( mask_index( _g.face_normals ), \
                             np.linalg.norm( _g.face_normals ) )
        [ _ax.arrow( *mask_index( _g.face_centers[:,f] ), \
                     *normals[:,f], fc = 'k', ec = 'k' ) \
                                            for f in np.arange( _g.num_faces ) ]

#------------------------------------------------------------------------------#

def plot_grid_2d( _g ):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    nodes, cells, _  = sps.find( _g.cell_nodes() )
    polygons = []
    for c in np.arange( _g.num_cells ):
        mask = np.where( cells == c )
        cell_nodes = _g.nodes[:, nodes[mask]]
        index = sort_points.sort_point_plane( cell_nodes, _g.cell_centers[:,c] )
        polygons.append( Polygon( cell_nodes[0:2,index].T, True ) )

    p = PatchCollection( polygons, cmap=matplotlib.cm.jet, alpha=0.4 )
    p.set_array( np.zeros( len( polygons ) ) )
    ax.add_collection(p)

    return fig.number, ax

#------------------------------------------------------------------------------#
