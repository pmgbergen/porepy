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

import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from core.grids import simplex, structured
from compgeom import sort_points

#------------------------------------------------------------------------------#

def plot_grid(g, info = None, alpha=0.5):
    """ plot the grid in a 3d framework.

    It is possible to add the cell ids at the cells centers (info option 'c'),
    the face ids at the face centers (info option 'f'), the node ids at the node
    (info option 'n'), and the normal at the face (info option 'o').

    Parameters:
    g: the grid
    info: (optional) add extra information to the plot
    alpha: (optonal) transparency of cells (2d) and faces (3d)

    Return:
    fig.number: the id of the plot

    How to use:
    plot_grid( g, info = "ncfo", alpha = 0 )

    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if g.dim == 2:   plot_grid_2d(g, ax, alpha)
    elif g.dim == 3: plot_grid_3d(g, ax, alpha)
    else:            raise NotImplementedError('Under construction')

    x = [ np.amin(g.nodes[0,:]), np.amax(g.nodes[0,:]) ]
    y = [ np.amin(g.nodes[1,:]), np.amax(g.nodes[1,:]) ]
    z = [ np.amin(g.nodes[2,:]), np.amax(g.nodes[2,:]) ]

    if not np.isclose(x[0], x[1]): ax.set_xlim3d( x ); ax.set_xlabel('x')
    if not np.isclose(y[0], y[1]): ax.set_ylim3d( y ); ax.set_ylabel('y')
    if not np.isclose(z[0], z[1]): ax.set_zlim3d( z ); ax.set_zlabel('z')
    ax.set_title( " ".join( g.name ) )

    if info is not None: add_info( g, info, ax )
    plt.show()

    return fig.number

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

def add_info( g, info, ax ):

    def disp( i, p, c, m ): ax.scatter( *p, c=c, marker=m ); ax.text( *p, i )
    def disp_loop( v, c, m ): [disp( i, ic, c, m ) for i, ic in enumerate(v.T)]

    info = info.upper()

    if "C" in info: disp_loop( g.cell_centers, 'r', 'o' )
    if "N" in info: disp_loop( g.nodes, 'b', 's' )
    if "F" in info: disp_loop( g.face_centers, 'y', 'd' )

    if "O" in info.upper():
        normals = np.array( [ n/np.linalg.norm(n) \
                                  for n in g.face_normals.T ] ).T
        ax.quiver( *g.face_centers, *normals, color = 'k', length=0.25 )

#------------------------------------------------------------------------------#

def plot_grid_1d( g, ax ):
    cell_nodes = g.cell_nodes()
    nodes, cells, _  = sps.find( cell_nodes )
    for c in np.arange( g.num_cells ):
        loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c+1])
        ptsId = nodes[loc]
        pts = g.nodes[:, ptsId]
        poly = Poly3DCollection( [pts.T] )
        poly.set_edgecolor('k')
        ax.add_collection3d(poly)

#------------------------------------------------------------------------------#

def plot_grid_2d( g, ax, alpha ):
    nodes, cells, _  = sps.find( g.cell_nodes() )
    RGB_face = [1,0,0,alpha]
    for c in np.arange( g.num_cells ):
        ptsId = nodes[ cells == c ]
        mask = sort_points.sort_point_plane( g.nodes[:, ptsId], \
                                             g.cell_centers[:, c] )

        pts = g.nodes[:, ptsId[mask]]
        poly = Poly3DCollection( [pts.T] )
        poly.set_edgecolor('k')
        poly.set_facecolors(RGB_face)
        ax.add_collection3d(poly)

    ax.view_init(90, -90)

#------------------------------------------------------------------------------#

def plot_grid_3d( g, ax, alpha ):
    faces_cells, cells, _ = sps.find( g.cell_faces )
    nodes_faces, faces, _ = sps.find( g.face_nodes )
    RGB_face = [1,0,0,alpha]
    for c in np.arange( g.num_cells ):
        fs = faces_cells[ cells == c ]
        for f in fs:
            ptsId = nodes_faces[ faces == f ]
            mask = sort_points.sort_point_plane( g.nodes[:, ptsId], \
                                                 g.face_centers[:, f], \
                                                 g.face_normals[:, f] )
            pts = g.nodes[:, ptsId[mask]]
            poly = Poly3DCollection( [pts.T] )
            poly.set_edgecolor('k')
            poly.set_facecolors(RGB_face)
            ax.add_collection3d(poly)

#------------------------------------------------------------------------------#
