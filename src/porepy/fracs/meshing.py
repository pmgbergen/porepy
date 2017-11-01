"""
Main module for grid generation in fractured domains in 2d and 3d.

The module serves as the only neccessary entry point to create the grid. It
will therefore wrap interface to different mesh generators, pass options to the
generators etc.

"""
import numpy as np
import scipy.sparse as sps

from porepy.fracs import structured, simplex, split_grid
from porepy.fracs.fractures import Intersection
from porepy import FractureNetwork
from porepy.fracs.fractures import FractureNetwork as FractureNetwork_full
from porepy.grids.grid_bucket import GridBucket
from porepy.grids.grid import FaceTag
from porepy.utils import setmembership, mcolon


def simplex_grid(fracs, domain, **kwargs):
    """
    Main function for grid generation. Creates a fractured simiplex grid in 2
    or 3 dimensions.

    Parameters
    ----------
    fracs (list of np.ndarray): One list item for each fracture. Each item
        consist of a (nd x n) array describing fracture vertices. The
        fractures may be intersecting.
    domain (dict): Domain specification, determined by xmin, xmax, ...
    **kwargs: May contain fracture tags, options for gridding, etc.

    Returns
    -------
    GridBucket: A complete bucket where all fractures are represented as
        lower dim grids. The higher dim fracture faces are split in two,
        and on the edges of the GridBucket graph the mapping from lower dim
        cells to higher dim faces are stored as 'face_cells'. Each face is
        given a FaceTag depending on the type:
           NONE: None of the below (i.e. an internal face)
           DOMAIN_BOUNDARY: All faces that lie on the domain boundary
               (i.e. should be given a boundary condition).
           FRACTURE: All faces that are split (i.e. has a connection to a
               lower dim grid).
           TIP: A boundary face that is not on the domain boundary, nor
               coupled to a lower domentional domain.

    Examples
    --------
    frac1 = np.array([[1,4],[1,4]])
    frac2 = np.array([[1,4],[4,1]])
    fracs = [frac1, frac2]
    domain = {'xmin': 0, 'ymin': 0, 'xmax':5, 'ymax':5}
    gb = simplex_grid(fracs, domain)

    """
    if 'zmax' in domain:
        ndim = 3
    elif 'ymax' in domain:
        ndim = 2
    else:
        raise ValueError('simplex_grid only supported for 2 or 3 dimensions')

    # Call relevant method, depending on grid dimensions.
    if ndim == 2:
        # Convert the fracture to a fracture dictionary.
        if len(fracs) == 0:
            f_lines = np.zeros((2, 0))
            f_pts = np.zeros((2, 0))
        else:
            f_lines = np.reshape(np.arange(2 * len(fracs)), (2, -1), order='F')
            f_pts = np.hstack(fracs)
        frac_dic = {'points': f_pts, 'edges': f_lines}
        grids = simplex.triangle_grid(frac_dic, domain, **kwargs)
    elif ndim == 3:
        grids = simplex.tetrahedral_grid(fracs, domain, **kwargs)
    else:
        raise ValueError('Only support for 2 and 3 dimensions')
    # Tag tip faces
    tag_faces(grids)

    # Assemble grids in a bucket
    gb = assemble_in_bucket(grids)
    gb.compute_geometry()
    # Split the grids.
    split_grid.split_fractures(gb)
    return gb

#------------------------------------------------------------------------------#

def dfn(fracs, conforming, intersections=None, **kwargs):
    """

    Parameters:
        fracs (either Fractures, or a FractureNetwork).
        conforming (boolean): If True, the mesh will be conforming along 1d
            intersections.
        intersections (list of lists, optional): Each item corresponds to an
            intersection between two fractures. In each sublist, the first two
            indices gives fracture ids (refering to order in fracs). The third
            item is a numpy array representing intersection coordinates. If no
            intersections provided, intersections will be detected using
            function in FractureNetwork.
        **kwargs: Parameters passed to gmsh.

    """

    if isinstance(fracs, FractureNetwork) \
       or isinstance(frac, FractureNetwork_full):
        network = fracs
    else:
        network = FractureNetwork(fracs)

    if intersections is not None:
        network.intersections = [Intersection(*i) for i in intersections]
    else:
        network.find_intersections()

    if conforming:
        grids = simplex.triangle_grid_embedded(network, find_isect=False,
                                               **kwargs)
        tag_faces(grids, check_highest_dim=False)
    else:

        grid_list = []
        neigh_list = []

        for fi in range(len(network._fractures)):
            # Rotate fracture vertexes and intersection points
            fp, ip, other_frac, rot, cp = network.fracture_to_plane(fi)

            f_lines = np.reshape(np.arange(ip.shape[1]), (2, -1), order='F')
            frac_dict = {'points': ip, 'edges': f_lines}
            grids = simplex.triangle_grid(frac_dict, fp, verbose=False,
                                          **kwargs)

            irot = rot.T

            # Loop over grids, rotate back again to 3d coordinates
            for gl in grids:
                for g in gl:
                    g.nodes = irot.dot(g.nodes) + cp

            assert len(grids[0]) == 1, 'Fracture should be covered by single'\
                'mesh'

            grid_list.append(grids)
            neigh_list.append(other_frac)
        return grid_list, neigh_list

    gb = assemble_in_bucket(grids)
    gb.compute_geometry()
    split_grid.split_fractures(gb)
    return gb

#------------------------------------------------------------------------------#

def from_gmsh(file_name, dim, **kwargs):
    """
    Import an already generated grid from gmsh.
    NOTE: Only 2d grid is implemented so far.

    Parameters
    ----------
    file_name (string): Gmsh file name.
    dim (int): Spatial dimension of the grid.
    **kwargs: May contain fracture tags, options for gridding, etc.

    Returns
    -------
    Grid or GridBucket: If no fractures are present in the gmsh file a simple
        grid is returned. Otherwise, a complete bucket where all fractures are
        represented as lower dim grids. See the documentation of simplex_grid
        for further details.

    Examples
    --------
    gb = from_gmsh('grid.geo', 2)

    """
    # Call relevant method, depending on grid dimensions.
    if dim == 2:
        if file_name.endswith('.geo'):
            simplex.triangle_grid_run_gmsh(file_name, **kwargs)
            grids = simplex.triangle_grid_from_gmsh(file_name, **kwargs)
        elif file_name.endswith('.msh'):
            grids = simplex.triangle_grid_from_gmsh(file_name, **kwargs)

#    elif dim == 3:
#        grids = simplex.tetrahedral_grid_from_gmsh(file_name, **kwargs)
#   NOTE: function simplex.tetrahedral_grid needs to be split as did for
#   simplex.triangle_grid
    else:
        raise ValueError('Only support for 2 dimensions')

    # No fractures are specified, return a simple grid
    if len(grids[1]) == 0:
        grids[0][0].compute_geometry()
        return grids[0][0]

    # Tag tip faces
    tag_faces(grids)

    # Assemble grids in a bucket
    gb = assemble_in_bucket(grids)
    gb.compute_geometry()
    # Split the grids.
    split_grid.split_fractures(gb)
    return gb

#------------------------------------------------------------------------------#

def cart_grid(fracs, nx, **kwargs):
    """
    Creates a cartesian fractured GridBucket in 2- or 3-dimensions.

    Parameters
    ----------
    fracs (list of np.ndarray): One list item for each fracture. Each item
        consist of a (nd x 3) array describing fracture vertices. The
        fractures has to be rectangles(3D) or straight lines(2D) that
        alignes with the axis. The fractures may be intersecting.
        The fractures will snap to closest grid faces.
    nx (np.ndarray): Number of cells in each direction. Should be 2D or 3D
    **kwargs:
        physdims (np.ndarray): Physical dimensions in each direction.
            Defaults to same as nx, that is, cells of unit size.
        May also contain fracture tags, options for gridding, etc.

    Returns:
    -------
    GridBucket: A complete bucket where all fractures are represented as
        lower dim grids. The higher dim fracture faces are split in two,
        and on the edges of the GridBucket graph the mapping from lower dim
        cells to higher dim faces are stored as 'face_cells'. Each face is
        given a FaceTag depending on the type:
           NONE: None of the below (i.e. an internal face)
           DOMAIN_BOUNDARY: All faces that lie on the domain boundary
               (i.e. should be given a boundary condition).
           FRACTURE: All faces that are split (i.e. has a connection to a
               lower dim grid).
           TIP: A boundary face that is not on the domain boundary, nor
               coupled to a lower domentional domain.

    Examples
    --------
    frac1 = np.array([[1,4],[2,2]])
    frac2 = np.array([[2,2],[4,1]])
    fracs = [frac1, frac2]
    gb = cart_grid(fracs, [5,5])
    """
    ndim = np.asarray(nx).size
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
    tag_faces(grids)

    # Asemble in bucket
    gb = assemble_in_bucket(grids)
    gb.compute_geometry()

    # Split grid.
    split_grid.split_fractures(gb, **kwargs)
    return gb


def tag_faces(grids, check_highest_dim=True):
    """
    Tag faces of grids. Three different tags are given to different types of
    faces:
        NONE: None of the below (i.e. an internal face)
        DOMAIN_BOUNDARY: All faces that lie on the domain boundary
            (i.e. should be given a boundary condition).
        FRACTURE: All faces that are split (i.e. has a connection to a
            lower dim grid).
        TIP: A boundary face that is not on the domain boundary, nor
            coupled to a lower domentional domain.

    Parameters:
        grids (list): List of grids to be tagged. Sorted per dimension.
        check_highest_dim (boolean, default=True): If true, we require there is
            a single mesh in the highest dimension. The test is useful, but
            should be waived for dfn meshes.

    """

    for gs in grids:
        for g in gs:
            g.remove_face_tag([True] * g.num_faces, FaceTag.DOMAIN_BOUNDARY)

    # Assume only one grid of highest dimension
    if check_highest_dim:
        assert len(grids[0]) == 1, 'Must be exactly'\
            '1 grid of dim: ' + str(len(grids))

    for g_h in np.atleast_1d(grids[0]):
        bnd_faces = g_h.get_boundary_faces()
        g_h.add_face_tag(bnd_faces, FaceTag.DOMAIN_BOUNDARY)
        bnd_nodes, _, _ = sps.find(g_h.face_nodes[:, bnd_faces])
        bnd_nodes = np.unique(bnd_nodes)
        for g_dim in grids[1:-1]:
            for g in g_dim:
                # We find the global nodes of all boundary faces
                bnd_faces_l = g.get_boundary_faces()
                indptr = g.face_nodes.indptr
                fn_loc = mcolon.mcolon(
                    indptr[bnd_faces_l], indptr[bnd_faces_l + 1])
                nodes_loc = g.face_nodes.indices[fn_loc]
                # Convert to global numbering
                nodes_glb = g.global_point_ind[nodes_loc]
                # We then tag each node as a tip node if it is not a global
                # boundary node
                is_tip = np.in1d(nodes_glb, bnd_nodes, invert=True)
                # We reshape the nodes such that each column equals the nodes of
                # one face. If a face only contains global boundary nodes, the
                # local face is also a boundary face. Otherwise, we add a TIP tag.
                n_per_face = nodes_per_face(g)
                is_tip = np.any(is_tip.reshape(
                    (n_per_face, bnd_faces_l.size), order='F'), axis=0)
                g.add_face_tag(bnd_faces_l[is_tip], FaceTag.TIP)
                g.add_face_tag(bnd_faces_l[is_tip == False],
                               FaceTag.DOMAIN_BOUNDARY)


def nodes_per_face(g):
    """
    Returns the number of nodes per face for a given grid
    """
    if ('TensorGrid' in g.name or 'CartGrid' in g.name) and g.dim == 3:
        n_per_face = 4
    elif 'TetrahedralGrid' in g.name:
        n_per_face = 3
    elif ('TensorGrid' in g.name or 'CartGrid' in g.name) and g.dim == 2:
        n_per_face = 2
    elif 'TriangleGrid'in g.name:
        n_per_face = 2
    elif ('TensorGrid' in g.name or 'CartGrid' in g.name) and g.dim == 1:
        n_per_face = 1
    else:
        raise ValueError(
            "Can not find number of nodes per face for grid: " + str(g.name))
    return n_per_face


def assemble_in_bucket(grids):
    """
    Create a GridBucket from a list of grids.
    Parameters
    ----------
    grids: A list of lists of grids. Each element in the list is a list
        of all grids of a the same dimension. It is assumed that the
        grids are sorted from high dimensional grids to low dimensional grids.
        All grids must also have the mapping g.global_point_ind which maps
        the local nodes of the grid to the nodes of the highest dimensional
        grid.

    Returns
    -------
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
            n_per_face = nodes_per_face(hg)
            fn_loc = hg.face_nodes.indices.reshape((n_per_face, hg.num_faces),
                                                   order='F')
            # Convert to global numbering
            fn = hg.global_point_ind[fn_loc]
            fn = np.sort(fn, axis=0)

            for lg in grids[dim + 1]:
                cell_2_face, cell = obtain_interdim_mappings(
                    lg, fn, n_per_face)
                face_cells = sps.csc_matrix(
                    (np.array([True] * cell.size), (cell, cell_2_face)),
                    (lg.num_cells, hg.num_faces))

                # This if may be unnecessary, but better safe than sorry.
                if face_cells.size > 0:
                    bucket.add_edge([hg, lg], face_cells)

    return bucket


def obtain_interdim_mappings(lg, fn, n_per_face):
    """
    Find mappings between faces in higher dimension and cells in the lower
    dimension
    """
    if lg.dim > 0:
        cn_loc = lg.cell_nodes().indices.reshape((n_per_face,
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
