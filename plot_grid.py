# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 19:37:05 2016

@author: keile
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri

from core.grids import simplex, structured


def plot_grid(g, show=True):
    
    if isinstance(g, simplex.TriangleGrid):
        return plot_tri_grid(g, show)
    elif isinstance(g, structured.TensorGrid) and g.dim == 2:
        plot_cart_grid_2d(g, show)
    else:
        raise NotImplementedError('Under construction')


def plot_tri_grid(g, show=True):
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
    h = plt.figure()
    triang = matplotlib.tri.Triangulation(g.nodes[0], g.nodes[1], tri)
    plt.triplot(triang)
    if show:
        plt.show()
    return h

def plot_cart_grid_2d(g, show=True):
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
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # It would have been better to have a color map which makes all
    ax.pcolormesh(x, y, z, edgecolors='k', cmap='gray', alpha=0.5)
    if show:
        plt.show()


def plot_grid_fractures(g, show=True):
    h = plot_grid(g, show=False)
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
