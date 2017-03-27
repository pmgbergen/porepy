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
from utils import setmembership, half_space
from core.grids import structured, point_grid
from compgeom import basics as cg
from viz import plot_grid


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
        grids = _cart_grid_3d(fracs, nx, physdims)
    else:
        raise ValueError('Only support for 2 and 3 dimensions')
    gb = meshing.assemble_in_bucket(grids)
    gb.compute_geometry()
    split_grid.split_fractures(gb, offset=0.2)
    return gb


def _cart_grid_3d(fracs, nx, physdims):
    g_3d = structured.CartGrid(nx, physdims=physdims)
    g_3d.global_point_ind = np.arange(g_3d.num_nodes)
    g_3d.compute_geometry()
    g_2d = []
    g_1d = []
    g_0d = []

    tol = .1 * np.asarray(physdims) / np.asarray(nx)
    shared_nodes = np.zeros(g_3d.num_nodes)
    # Create 2D grids
    for f in fracs:
        print(f)
        is_xy_frac = np.allclose(f[2, 0], f[2])
        is_xz_frac = np.allclose(f[1, 0], f[1])
        is_yz_frac = np.allclose(f[0, 0], f[0])
        assert is_xy_frac + is_xz_frac + is_yz_frac == 1, \
            'Fracture must align to x- or y-axis'
        # snap to grid
        f_s = np.round(f * nx[:, np.newaxis] / physdims[:, np.newaxis]
                       ) * physdims[:, np.newaxis] / nx[:, np.newaxis]
        if is_xy_frac:
            flat_dim = [2]
            active_dim = [0, 1]
        elif is_xz_frac:
            flat_dim = [1]
            active_dim = [0, 2]
        else:
            flat_dim = [0]
            active_dim = [1, 2]
        # construct normal vectors
        sign = 2 * cg.is_ccw_polygon(f_s[active_dim]) - 1
        print(sign)
        tangent = f_s.take(
            np.arange(f_s.shape[1]) + 1, axis=1, mode='wrap') - f_s
        normal = tangent
        normal[active_dim] = tangent[active_dim[1::-1]]
        normal[active_dim[1]] = -normal[active_dim[1]]
        normal = sign * normal
        in_hull = half_space.half_space_int(
            normal, f_s, g_3d.face_centers)
        f_tag = np.logical_and(
            in_hull,
            np.logical_and(f_s[flat_dim, 0] - tol[flat_dim] <=
                           g_3d.face_centers[flat_dim],
                           g_3d.face_centers[flat_dim] <
                           f_s[flat_dim, 0] + tol[flat_dim]))
        f_tag = f_tag.ravel()
        nodes = sps.find(g_3d.face_nodes[:, f_tag])[0]
        nodes = np.unique(nodes)
        loc_coord = g_3d.nodes[:, nodes]
        g = _create_embedded_2d_grid(loc_coord, nodes)
        print(g_3d.nodes[:, g.global_point_ind] - g.nodes, 'should be 0')
        g_2d.append(g)
        shared_nodes[nodes] += 1

    grids = [[g_3d], g_2d, g_1d, g_0d]
    return grids


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


def _create_embedded_2d_grid(loc_coord, glob_id):
    loc_center = np.mean(loc_coord, axis=1).reshape((-1, 1))
    loc_coord -= loc_center
    # Check that the points indeed form a line
    assert cg.is_planar(loc_coord)
    # Find the tangent of the line
    # Projection matrix
    rot = cg.project_plane_matrix(loc_coord)
    loc_coord_2d = rot.dot(loc_coord)
    # The points are now 2d along two of the coordinate axis, but we
    # don't know which yet. Find this.
    sum_coord = np.sum(np.abs(loc_coord_2d), axis=1)
    active_dimension = np.logical_not(np.isclose(sum_coord, 0))
    # Check that we are indeed in 2d
    assert np.sum(active_dimension) == 2
    # Sort nodes, and create grid
    coord_2d = loc_coord_2d[active_dimension]
    sort_ind = np.lexsort((coord_2d[0], coord_2d[1]))
    sorted_coord = coord_2d[:, sort_ind]
    sorted_coord = np.round(sorted_coord * 1e10) / 1e10
    print(sorted_coord, 'sorted_coord')
    unique_x = np.unique(sorted_coord[0])
    unique_y = np.unique(sorted_coord[1])
    print(unique_x, 'uniqeu_coord')
    print(unique_y, 'uniqeu_coord-y')
    #assert unique_x.size == unique_y.size
    g = structured.TensorGrid(unique_x, unique_y)
    assert np.all(g.nodes[0:2] - sorted_coord == 0)

    # Project back to active dimension
    nodes = np.zeros(g.nodes.shape)
    nodes[active_dimension] = g.nodes[0:2]
    g.nodes = nodes
    # Project back again to 3d coordinates

    irot = rot.transpose()
    g.nodes = irot.dot(g.nodes)
    g.nodes += loc_center

    # Add mapping to global point numbers
    g.global_point_ind = glob_id[sort_ind]
    return g
