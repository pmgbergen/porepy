"""

Main module for grid generation in fractured domains in 2d and 3d.

The module serves as the only neccessary entry point to create the grid. It
will therefore wrap interface to different mesh generators, pass options to the
generators etc.

"""
import numpy as np
import scipy.sparse as sps

from gridding.fractured import grid_2d, grid_3d, split_grid, meshing
from gridding.grid_bucket import GridBucket
from gridding.gmsh import mesh_2_grid
from utils import setmembership
from core.grids import structured, point_grid


def cart_grid(fracs, nx, physdims=None, **kwargs):
    """
    Creates a cartesian fractured GridBucket.

    Parameters:
        fracs (list of np.ndarray): One list item for each fracture. Each item
            consist of a (nd x nd) array describing fracture vertices. The
            fractures has to be rectangles(3D) or straight lines(2D) that
            alignes with one of the axis. The fractures may be intersecting.
            The fractures will snap to closest grid face.
        nx (np.ndarray): An array of size 2 (2D) or 3(3D) giving the number of
            cells in each dimension.
        physdims (np.ndarray): Physical dimensions in each direction.
            Defaults to same as nx, that is, cells of unit size.
    Returns:
        GridBucket: A complete bucket where all fractures are represented as
            lower dim grids. The higher dim faces are split in two, and on the
            edges of the GridBucket graph the mapping from lower dim cells to
            higher dim faces are stored as 'face_cells'
    """
    ndim = np.asarray(nx).size
    if physdims is None:
        physdims = nx
    elif np.asarray(physdims).size != ndim:
        raise ValueError('Physical dimension must equal grid dimension')
    # Call relevant method, depending on grid dimensions
    # Note: If we ever develop interfaces to grid generators other than gmsh,
    # this should not be visible here, but rather in the respective
    # nd.create_grid methods.
    if ndim == 2:
        grids = _cart_grid_2d(fracs, nx, physdims)
    elif ndim == 3:
        raise NotImplementedError
        grids = _cart_grid_3d(fracs, nx, physdims)
    else:
        raise ValueError('Only support for 2 and 3 dimensions')
    gb = meshing.assemble_in_bucket(grids)
    gb.compute_geometry()
    split_grid.split_fractures(gb, offset=0.2)
    return gb


def _cart_grid_2d(fracs, nx, physdims):
    g_2d = structured.CartGrid(nx, physdims=physdims)
    g_2d.global_point_ind = np.arange(g_2d.num_nodes)
    g_2d.compute_geometry()
    g_1d = []
    g_0d = []
    # Create grids of fracture:
    tol = .1 * np.asarray(physdims) / np.asarray(nx)
    shared_nodes = np.zeros(g_2d.num_nodes)
    for f in fracs:
        is_x_frac = f[1, 0] == f[1, 1]
        is_y_frac = f[0, 0] == f[0, 1]
        assert is_x_frac != is_y_frac, 'Fracture must align to x- or y-axis'

        if is_x_frac:
            f_y = np.round(f[1] * nx[1] / physdims[1]) * physdims[1] / nx[1]
            f_tag = np.logical_and(
                np.logical_and(f[0, 0] <= g_2d.face_centers[0],
                               g_2d.face_centers[0] <= f[0, 1]),
                np.logical_and(f_y[1] - tol[1] <= g_2d.face_centers[1],
                               g_2d.face_centers[1] < f_y[1] + tol[1]))
        else:
            f_x = np.round(f[0] * nx[0] / physdims[0]) * physdims[0] / nx[0]
            f_tag = np.logical_and(
                np.logical_and(f_x[0] - tol[0] <= g_2d.face_centers[0],
                               g_2d.face_centers[0] < f_x[1] + tol[0]),
                np.logical_and(f[1, 0] <= g_2d.face_centers[1],
                               g_2d.face_centers[1] <= f[1, 1]))
        nodes = sps.find(g_2d.face_nodes[:, f_tag])[0]
        nodes = np.unique(nodes)
        loc_coord = g_2d.nodes[:, nodes]
        g = mesh_2_grid.create_embedded_line_grid(loc_coord, nodes)
        g_1d.append(g)
        shared_nodes[nodes] += 1

    # Create 0-D grids
    for global_node in np.where(shared_nodes > 1):
        g = point_grid.PointGrid(g_2d.nodes[:, global_node])
        g.global_point_ind = np.asarray(global_node)
        g_0d.append(g)

    grids = [[g_2d], g_1d, g_0d]
    return grids
