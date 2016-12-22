# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 22:20:50 2016

@author: keile
"""

import matplotlib.pyplot as plt
import matplotlib.tri

from core.grids import structured, simplex


def plot_cell_data(g, data):
    if isinstance(g, simplex.TriangleGrid):
        plot_tri_data(g, data)
    elif isinstance(g, structured.TensorGrid) and g.dim == 2:
        plot_cart_data(g, data)
    else:
        raise NotImplementedError('Under construction')


def plot_tri_data(g, data):
    T = g.cell_node_matrix()
    
    if data.shape[0] == g.Nc:
        data = data.T
    elif data.shape[1] != g.Nc:
        raise ValueError('Something\'s wrong with the dimensions of the data')
        
    triang = matplotlib.tri.Triangulation(g.nodes[0], g.nodes[1], T)
       
    for i in range(0, data.shape[0]):
        plt.figure()
        plt.tripcolor(triang, data[i])


def plot_cart_data(g, data):
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
    cart_dims = g.cart_dims
    x = g.nodes[0].reshape(cart_dims + 1)
    y = g.nodes[1].reshape(cart_dims + 1)
    z = data.reshape(cart_dims)
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # It would have been better to have a color map which makes all
    plt.pcolormesh(x, y, z, edgecolors='k', alpha=0.5)
    plt.colorbar()
    plt.show()


def surf_cart_data(g, data):
    """
    Surface plot of data on a Cartesian 2D grid
    
    input: g: 2D Cartesian grid (equidistanc spacing in each direction is assumed)
    data: array of cell data to be visualized.
    """

    if not isinstance(g, structured.TensorGrid) or not g.dim == 2:
        raise ValueError('Method is only available for 2D Cartesian grids')

    nx = g.cart_dims
    node_min = np.min(g.nodes, axis=1)
    Lx = np.max(g.nodes, axis=1) - node_min
    dx = Lx/nx
    x = node_min[0] + np.linspace(0.5 * dx[0], (Lx[0] - 0.5 * dx[0]) * dx[0], nx[0])
    y = node_min[1] + np.linspace(0.5 * dx[1], (Lx[1] - 0.5 * dx[1]) * dx[1], nx[1])
    x, y = np.meshgrid(x, y)
    z = data.reshape(nx)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    return fig

