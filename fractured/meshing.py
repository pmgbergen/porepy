"""
Main module for grid generation in fractured domains in 2d and 3d.

The module serves as the only neccessary entry point to create the grid. It
will therefore wrap interface to different mesh generators, pass options to the
generators etc.

"""
import numpy as np
import scipy.sparse as sps

from gridding.fractured import structured, simplex, split_grid
from gridding.grid_bucket import GridBucket
from utils import setmembership
from core.grids.grid import FaceTag


def simplex_grid(fracs, domain, **kwargs):
    """
    Main function for grid generation.

    Parameters:
        fracs (list of np.ndarray): One list item for each fracture. Each item
            consist of a (nd x n) array describing fracture vertices. The
            fractures may be intersecting. 
        domain (dict): Domain specification, determined by xmin, xmax, ...
        **kwargs: May contain fracture tags, options for gridding, etc.
    Returns:
        GridBucket: A complete bucket where all fractures are represented as 
            lower dim grids. The higher dim faces are split in two, and on the
            edges of the GridBucket graph the mapping from lower dim cells to 
            higher dim faces are stored as 'face_cells'
    """

    ndim = fracs[0].shape[0]

    # Call relevant method, depending on grid dimensions.
    if ndim == 2:
        # Convert the fracture to a fracture dictionary.
        f_lines = np.reshape(np.arange(2 * len(fracs)), (2, -1), order='F')
        f_pts = np.hstack(fracs)
        frac_dic = {'points': f_pts, 'edges': f_lines}
        grids = simplex.triangle_grid(frac_dic, domain, **kwargs)
    elif ndim == 3:
        grids = simplex.tetrahedral_grid(fracs, domain, **kwargs)
    else:
        raise ValueError('Only support for 2 and 3 dimensions')
    # Tag tip faces
    tag_tip_faces(grids, ndim)
    # Assemble grids in a bucket
    gb = assemble_in_bucket(grids)
    gb.compute_geometry()
    # Split the grids.
    split_grid.split_fractures(gb)
    return gb


def cart_grid(fracs, nx, **kwargs):
    """
    Creates a tensor fractured GridBucket.

    Parameters:
        fracs (list of np.ndarray): One list item for each fracture. Each item
            consist of a (nd x 3) array describing fracture vertices. The
            fractures has to be rectangles(3D) or straight lines(2D) that
            alignes with the axis. The fractures may be intersecting.
            The fractures will snap to closest grid faces.
        nx (np.ndarray): Number of cells in each direction. Should be 2D or 3D
        kwargs:
            physdims (np.ndarray): Physical dimensions in each direction.
                Defaults to same as nx, that is, cells of unit size.
            offset (float):  defaults to 0. Will perturb the nodes around the
                faces that are split. NOTE: this is only for visualization.
                E.g., the face centers are not perturbed.
    Returns:
        GridBucket: A complete bucket where all fractures are represented as
            lower dim grids. The higher dim faces are split in two, and on the
            edges of the GridBucket graph the mapping from lower dim cells to
            higher dim faces are stored as 'face_cells'
    """
    ndim = np.asarray(nx).size
    offset = kwargs.get('offset', 0)
    physdims = kwargs.get('physdims', None)

    if physdims is None:
        physdims = nx
    elif np.asarray(physdims).size != ndim:
        raise ValueError('Physical dimension must equal grid dimension')

    # Call relevant method, depending on grid dimensions
    if ndim == 2:
        grids = structured.cart_grid_2d(fracs, nx, physdims=physdims)
    elif ndim == 3:
        grids = structured.cart_grid_3d(fracs, nx, physdims=physdims)
    else:
        raise ValueError('Only support for 2 and 3 dimensions')
    # Tag tip faces.
    tag_tip_faces(grids, ndim)

    # Asemble in bucket
    gb = assemble_in_bucket(grids)
    gb.compute_geometry()

    # Split grid.
    split_grid.split_fractures(gb, **kwargs)
    return gb


def tag_tip_faces(grids, ndim):
    print('Fracture TIP TAGGING NOT IMPLEMENTED')
    return
    for g_dim in grids:
        for g in g_dim:
            if g.dim != ndim:
                g.add_face_tag(g.get_boundary_faces(), FaceTag.TIP)


def assemble_in_bucket(grids):
    """
    Create a GridBucket from a list of grids.
    Parameters:
        grids: A list of lists of grids. Each element in the list is a list
            of all grids of a the same dimension. It is assumed that the
            grids are sorted from high dimensional grids to low dimensional grids.
            All grids must also have the mapping g.global_point_ind which maps
            the local nodes of the grid to the nodes of the highest dimensional
            grid.
    Returns:
        GridBucket: A GridBucket class where the mapping face_cells are given to
            each edge. face_cells maps from lower-dim cells to higher-dim faces.
    """

    # Create bucket
    bucket = GridBucket()
    [bucket.add_nodes(g_d) for g_d in grids]

    # We now find the face_cell mapings.
    for dim in range(len(grids) - 1):
        for hg in grids[dim]:
            # We have to specify the number of nodes per face to generate a
            # matrix of the nodes of each face.
            if 'TensorGrid'in hg.name and hg.dim == 3:
                nodes_per_face = 4
            elif 'TetrahedralGrid' in hg.name:
                nodes_per_face = 3
            elif 'TensorGrid'in hg.name and hg.dim == 2:
                nodes_per_face = 2
            elif 'TriangleGrid'in hg.name:
                nodes_per_face = 2
            elif 'TensorGrid' in hg.name and hg.dim == 1:
                nodes_per_face = 1
            else:
                raise ValueError(
                    "assemble_in_bucket not implemented for grid: " + str(hg.name))

            fn_loc = hg.face_nodes.indices.reshape((nodes_per_face, hg.num_faces),
                                                   order='F')
            # Convert to global numbering
            fn = hg.global_point_ind[fn_loc]
            fn = np.sort(fn, axis=0)

            for lg in grids[dim + 1]:
                cell_2_face, cell = obtain_interdim_mappings(
                    lg, fn, nodes_per_face)
                face_cells = sps.csc_matrix(
                    (np.array([True] * cell.size), (cell, cell_2_face)),
                    (lg.num_cells, hg.num_faces))

                # This if may be unnecessary, but better safe than sorry.
                if face_cells.size > 0:
                    bucket.add_edge([hg, lg], face_cells)

    return bucket


def obtain_interdim_mappings(lg, fn, nodes_per_face):
    # Next, find mappings between faces in one dimension and cells in the lower
    # dimension
    if lg.dim > 0:
        cn_loc = lg.cell_nodes().indices.reshape((nodes_per_face,
                                                  lg.num_cells),
                                                 order='F')
        cn = lg.global_point_ind[cn_loc]
        cn = np.sort(cn, axis=0)
    else:
        cn = np.array([lg.global_point_ind])
        # We also know that the higher-dimensional grid has faces
        # of a single node. This sometimes fails, so enforce it.
        if cn.ndim == 1:
            fn = fn.ravel()
    is_mem, cell_2_face = setmembership.ismember_rows(
        cn.astype(np.int32), fn.astype(np.int32), sort=False)
    # An element in cell_2_face gives, for all cells in the
    # lower-dimensional grid, the index of the corresponding face
    # in the higher-dimensional structure.

    low_dim_cell = np.where(is_mem)[0]
    return cell_2_face, low_dim_cell
