# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 19:37:05 2016

@author: keile
"""

import string
import numpy as np
import scipy.sparse as sps

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.collections import PatchCollection

import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mpl_toolkits.mplot3d import Axes3D, proj3d

from core.grids import grid
from compgeom import sort_points
from gridding import grid_bucket

#------------------------------------------------------------------------------#

def plot_grid(g, cell_value=None, vector_value=None, info=None, **kwargs):
    """ plot the grid in a 3d framework.

    It is possible to add the cell ids at the cells centers (info option 'c'),
    the face ids at the face centers (info option 'f'), the node ids at the node
    (info option 'n'), and the normal at the face (info option 'o'). If info is
    set to 'all' all the informations are displayed.

    Parameters:
    g: the grid
    cell_value: (optional) cell scalar field to be represented (only 1d and 2d)
    info: (optional) add extra information to the plot
    alpha: (optonal) transparency of cells (2d) and faces (3d)

    Return:
    fig.number: the id of the plot

    How to use:
    cell_id = np.arange(g.num_cells)
    plot_grid(g, cell_value=cell_id, info="ncfo", alpha=0.75)

    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title( " ".join( g.name ) )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if isinstance(g, grid.Grid):
        if cell_value is not None and g.dim !=3:
            extr_value = np.array([np.amin(cell_value), np.amax(cell_value)])
            kwargs['color_map'] = color_map(extr_value)

        plot_grid_single(g, ax, cell_value, vector_value, info, **kwargs)

    if isinstance(g, grid_bucket.Grid_Bucket):
        if cell_value is not None and g.dim_max() !=3:
            extr_value = np.array([np.inf, -np.inf])
            for _, v in g:
                extr_value[0] = min(np.amin(cell_value[v]), extr_value[0])
                extr_value[1] = max(np.amax(cell_value[v]), extr_value[1])
            kwargs['color_map'] = color_map(extr_value)

        plot_grid_bucket(g, ax, cell_value, vector_value, info, **kwargs)

    if kwargs.get('color_map'): fig.colorbar(kwargs['color_map'])

    plt.draw()
    plt.show()

#------------------------------------------------------------------------------#

def save_img(name, g, cell_value=None, vector_value=None, info=None, **kwargs):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if kwargs.get('title', True): ax.set_title( " ".join( g.name ) )

    if kwargs.get('axis', True):
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    else:
        ax.set_axis_off()

    if isinstance(g, grid.Grid):
        if cell_value is not None and g.dim !=3:
            extr_value = np.array([np.amin(cell_value), np.amax(cell_value)])
            kwargs['color_map'] = color_map(extr_value)

        plot_grid_single(g, ax, cell_value, vector_value, info, **kwargs)

    if isinstance(g, grid_bucket.Grid_Bucket):
        if cell_value is not None and g.dim_max() !=3:
            extr_value = np.array([np.inf, -np.inf])
            for _, v in g:
                extr_value[0] = min(np.amin(cell_value[v]), extr_value[0])
                extr_value[1] = max(np.amax(cell_value[v]), extr_value[1])
            kwargs['color_map'] = color_map(extr_value)

        plot_grid_bucket(g, ax, cell_value, vector_value, info, **kwargs)

    if kwargs.get('colorbar', True) and kwargs.get('color_map'):
        fig.colorbar(kwargs['color_map'])

    plt.draw()
    fig.savefig(name, bbox_inches='tight', pad_inches = 0)

#------------------------------------------------------------------------------#

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

#------------------------------------------------------------------------------#

def quiver(vector_value, ax, g, **kwargs):

    if vector_value.shape[1] == g.num_faces:
        where = g.face_centers
    elif vector_value.shape[1] == g.num_cells:
        where = g.cell_centers
    else:
        raise ValueError

    scale = kwargs.get('vector_scale', 0.02)
    for v in np.arange(vector_value.shape[1]):
        x = [where[0, v], where[0, v] + scale * vector_value[0, v]]
        y = [where[1, v], where[1, v] + scale * vector_value[1, v]]
        z = [where[2, v], where[2, v] + scale * vector_value[2, v]]
        linewidth = kwargs.get('linewidth', 1)
        a = Arrow3D(x, y, z, mutation_scale=5, linewidth=linewidth,
                    arrowstyle="-|>", color="k", zorder=np.inf)
        ax.add_artist(a)

#------------------------------------------------------------------------------#

def plot_grid_single(g, ax, cell_value, vector_value, info, **kwargs):

    plot_grid_xd(g, cell_value, vector_value, ax, **kwargs)
    x, y, z = lim(ax, g.nodes)
    if not np.isclose(x[0], x[1]): ax.set_xlim3d( x )
    if not np.isclose(y[0], y[1]): ax.set_ylim3d( y )
    if not np.isclose(z[0], z[1]): ax.set_zlim3d( z )

    if info is not None: add_info(g, info, ax)

#------------------------------------------------------------------------------#

def plot_grid_bucket(gb, ax, cell_value, vector_value, info, **kwargs):

    if cell_value is None:
        cell_value = gb.g_prop(np.empty(gb.size,dtype=object))

    if vector_value is None:
        vector_value = gb.g_prop(np.empty(gb.size,dtype=object))

    for g, v in gb:
        kwargs['rgb'] = np.divide(kwargs.get('rgb', [1,0,0]), int(v)+1)
        plot_grid_xd(g, cell_value[v], vector_value[v], ax, **kwargs)

    val = np.array([lim(ax, g.nodes) for g, _ in gb])

    x = [np.amin(val[:,0,:]), np.amax(val[:,0,:])]
    y = [np.amin(val[:,1,:]), np.amax(val[:,1,:])]
    z = [np.amin(val[:,2,:]), np.amax(val[:,2,:])]

    if not np.isclose(x[0], x[1]): ax.set_xlim3d( x )
    if not np.isclose(y[0], y[1]): ax.set_ylim3d( y )
    if not np.isclose(z[0], z[1]): ax.set_zlim3d( z )

    if info is not None: [add_info( g, info, ax ) for g, _ in gb]

#    if cell_value is not None and gb.dim_max() !=3:
#        fig.colorbar(color_map(cell_value))

#------------------------------------------------------------------------------#

def plot_grid_xd(g, cell_value, vector_value, ax, **kwargs):
    if g.dim == 0:   plot_grid_0d(g, ax, **kwargs)
    elif g.dim == 1: plot_grid_1d(g, cell_value, ax, **kwargs)
    elif g.dim == 2: plot_grid_2d(g, cell_value, ax, **kwargs)
    else:            plot_grid_3d(g, ax, **kwargs)

    if vector_value is not None: quiver(vector_value, ax, g, **kwargs)

#------------------------------------------------------------------------------#

def lim(ax, nodes):
    x = [np.amin(nodes[0,:]), np.amax(nodes[0,:])]
    y = [np.amin(nodes[1,:]), np.amax(nodes[1,:])]
    z = [np.amin(nodes[2,:]), np.amax(nodes[2,:])]
    return x, y, z

#------------------------------------------------------------------------------#

def color_map(extr_value, cmap_type='jet'):
    cmap = plt.get_cmap(cmap_type)
    scalar_map = mpl.cm.ScalarMappable(cmap=cmap)
    scalar_map.set_array(extr_value)
    scalar_map.set_clim(vmin=extr_value[0], vmax=extr_value[1])
    return scalar_map

#------------------------------------------------------------------------------#

def add_info( g, info, ax ):

    def disp( i, p, c, m ): ax.scatter( *p, c=c, marker=m ); ax.text( *p, i )
    def disp_loop( v, c, m ): [disp( i, ic, c, m ) for i, ic in enumerate(v.T)]

    info = info.upper()
    info = string.ascii_uppercase if info == "ALL" else info

    if "C" in info: disp_loop( g.cell_centers, 'r', 'o' )
    if "N" in info: disp_loop( g.nodes, 'b', 's' )
    if "F" in info: disp_loop( g.face_centers, 'y', 'd' )

    if "O" in info.upper():
        normals = np.array( [ n/np.linalg.norm(n) \
                                      for n in g.face_normals.T ] ).T
        quiver(normals, ax, g, **kwargs)

#------------------------------------------------------------------------------#

def plot_grid_0d(g, ax, **kwargs):
    ax.scatter(*g.nodes, color='k', marker='o', s=kwargs.get('pointsize', 1))

#------------------------------------------------------------------------------#

def plot_grid_1d(g, cell_value, ax, **kwargs):
    cell_nodes = g.cell_nodes()
    nodes, cells, _  = sps.find( cell_nodes )

    if kwargs.get('color_map'):
        scalar_map = kwargs['color_map']
        alpha = kwargs.get('alpha', 1)
        def color_edge(value): return scalar_map.to_rgba(value, alpha)
    else:
        cell_value = np.zeros(g.num_cells)
        def color_edge(value): return 'k'

    for c in np.arange( g.num_cells ):
        loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c+1])
        ptsId = nodes[loc]
        pts = g.nodes[:, ptsId]
        poly = Poly3DCollection( [pts.T] )
        poly.set_edgecolor(color_edge(cell_value[c]))
        ax.add_collection3d(poly)

#------------------------------------------------------------------------------#

def plot_grid_2d(g, cell_value, ax, **kwargs):
    cell_nodes = g.cell_nodes()
    nodes, cells, _  = sps.find( cell_nodes )

    alpha = kwargs.get('alpha', 1)
    if kwargs.get('color_map'):
        scalar_map = kwargs['color_map']
        def color_face(value): return scalar_map.to_rgba(value, alpha)
    else:
        cell_value = np.zeros(g.num_cells)
        rgb = kwargs.get('rgb', [1,0,0])
        def color_face(value): return np.r_[rgb, alpha]

    for c in np.arange( g.num_cells ):
        loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c+1])
        ptsId = nodes[loc]
        mask = sort_points.sort_point_plane( g.nodes[:, ptsId], \
                                             g.cell_centers[:, c] )

        pts = g.nodes[:, ptsId[mask]]
        linewidth = kwargs.get('linewidth', 1)
        poly = Poly3DCollection([pts.T], linewidth = linewidth)
        poly.set_edgecolor('k')
        poly.set_facecolors(color_face(cell_value[c]))
        ax.add_collection3d(poly)

    ax.view_init(90, -90)

#------------------------------------------------------------------------------#

def plot_grid_3d(g, ax, alpha, **kwargs):
    faces_cells, cells, _ = sps.find( g.cell_faces )
    nodes_faces, faces, _ = sps.find( g.face_nodes )
    for c in np.arange(g.num_cells):
        loc_c = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
        fs = faces_cells[loc_c]
        for f in fs:
            loc_f = slice(g.face_nodes.indptr[f], g.face_nodes.indptr[f+1])
            ptsId = nodes_faces[loc_f]
            mask = sort_points.sort_point_plane( g.nodes[:, ptsId], \
                                                 g.face_centers[:, f], \
                                                 g.face_normals[:, f] )
            pts = g.nodes[:, ptsId[mask]]
            linewidth = kwargs.get('linewidth', 1)
            poly = Poly3DCollection([pts.T], linewidth=linewidth)
            poly.set_edgecolor('k')
            poly.set_facecolors(np.r_[rgb, alpha])
            ax.add_collection3d(poly)

#------------------------------------------------------------------------------#
