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

def plot_grid(g, info = None, show=True):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if g.dim == 2:   plot_grid_2d(g, ax)
    elif g.dim == 3: plot_grid_3d(g, ax)
    else:            raise NotImplementedError('Under construction')

    x = [ np.amin(g.nodes[0,:]), np.amax(g.nodes[0,:]) ]
    y = [ np.amin(g.nodes[1,:]), np.amax(g.nodes[1,:]) ]
    z = [ np.amin(g.nodes[2,:]), np.amax(g.nodes[2,:]) ]

    if x[0] != x[1]: ax.set_xlim( x ); ax.set_xlabel('x')
    if y[0] != y[1]: ax.set_ylim( y ); ax.set_ylabel('y')
    if z[0] != z[1]: ax.set_zlim( z ); ax.set_zlabel('z')
    ax.set_title( g.name )

    if info is not None: add_info( g, info, ax )
    if show: plt.show()

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
    def disp_loop( v, c, m ): [ disp( i, ic, c, m ) for i, ic in enumerate( v.T ) ]

#    fig = plt.figure( figId )
    info = info.upper()

    if "C" in info: disp_loop( g.cell_centers, 'r', 'o' )
    if "N" in info: disp_loop( g.nodes, 'b', 's' )
    if "F" in info: disp_loop( g.face_centers, 'y', 'd' )

    if "O" in info.upper():
        normals = 0.1*np.array( [ n/np.linalg.norm(n) \
                                  for n in g.face_normals.T ] ).T
        [ ax.quiver( *g.face_centers[:,f], *normals[:,f], color = 'k', \
          length=0.25 ) for f in np.arange( g.num_faces ) ]

#------------------------------------------------------------------------------#

def plot_grid_2d( g, ax ):

    nodes, cells, _  = sps.find( g.cell_nodes() )
    for c in np.arange( g.num_cells ):
        ptsId = nodes[ cells == c ]
        mask = sort_points.sort_point_plane( g.nodes[:, ptsId], \
                                             g.cell_centers[:, c] )

        pts = g.nodes[:, ptsId[mask]]
        poly = Poly3DCollection( [pts.T] )
        poly.set_edgecolor('k')
        poly.set_facecolors('r')
        poly.set_alpha(0.5)
        ax.add_collection3d(poly)

    ax.view_init(90, 0)

#------------------------------------------------------------------------------#

def plot_grid_3d( g, ax ):

    faces_cells, cells, _ = sps.find( g.cell_faces )
    nodes_faces, faces, _ = sps.find( g.face_nodes )

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
            poly.set_facecolors('r')
            poly.set_alpha(0.5)
            ax.add_collection3d(poly)

#------------------------------------------------------------------------------#
