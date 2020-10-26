"""
Main module for grid generation in fractured domains in 2d and 3d.

The module serves as the only neccessary entry point to create the grid. It
will therefore wrap interface to different mesh generators, pass options to the
generators etc.

"""
import logging
import time
from typing import List

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.fracs import split_grid, structured, tools
from porepy.grids import mortar_grid
from porepy.grids.grid_bucket import GridBucket
from porepy.utils import mcolon

logger = logging.getLogger(__name__)


def grid_list_to_grid_bucket(
    grids: List[List[pp.Grid]], time_tot: float = None, **kwargs
) -> pp.GridBucket:
    """ Convert a list of grids to a full GridBucket.

    The list can come from several mesh constructors, both simplex and
    structured approaches uses this in 2D and 3D.

    The function can not be used on an arbitrary set of grids; they should
    contain information to glue grids together. This will be included for grids
    created by the standard mixed-dimensional grid constructors. In other
    words: Do *not* use this function directly unless you know what you are
    doing.

    Parameters:
        grids (list of lists of grids): Grids to enter into the bucket.
            Sorted per dimension.
        time_tot (double, optional): Start time for full mesh construction.
            Used for logging. Defaults to None, in which case no information
            on total time consumption is logged.
        **kwargs: Passed on to subfunctions.

    Returns:
        GridBucket: Final mixed-dimensional grid.

    """
    # Tag tip faces
    check_highest_dim = kwargs.get("check_highest_dim", False)
    _tag_faces(grids, check_highest_dim)

    logger.info("Assemble in bucket")
    tm_bucket = time.time()
    gb = _assemble_in_bucket(grids)
    logger.info("Done. Elapsed time " + str(time.time() - tm_bucket))

    logger.info("Compute geometry")
    tm_geom = time.time()
    gb.compute_geometry()
    # Split the grids.
    logger.info("Done. Elapsed time " + str(time.time() - tm_geom))
    logger.info("Split fractures")
    tm_split = time.time()
    split_grid.split_fractures(gb, **kwargs)
    logger.info("Done. Elapsed time " + str(time.time() - tm_split))

    create_mortar_grids(gb, **kwargs)

    gb.assign_node_ordering()

    if time_tot is not None:
        logger.info(
            "Mesh construction completed. Total time " + str(time.time() - time_tot)
        )

    return gb


def cart_grid(fracs: List[np.ndarray], nx: np.ndarray, **kwargs) -> pp.GridBucket:
    """
    Creates a cartesian fractured GridBucket in 2- or 3-dimensions.

    Parameters
    ----------
    fracs (list of np.ndarray): One list item for each fracture. Each item
        consist of a (nd x npt) array describing fracture vertices, where npt is 2
        for 2d domains, 4 for 3d domains. The fractures has to be rectangles(3D) or
        straight lines(2D) that alignes with the axis. The fractures may be intersecting.
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
        given boolean tags depending on the type:
           domain_boundary_faces: All faces that lie on the domain boundary
               (i.e. should be given a boundary condition).
           fracture_faces: All faces that are split (i.e. has a connection to a
               lower dim grid).
           tip_faces: A boundary face that is not on the domain boundary, nor
               coupled to a lower domentional domain.
        The union of the above three is the tag boundary_faces.

    Examples
    --------
    frac1 = np.array([[1, 4], [2, 2]])
    frac2 = np.array([[2, 2], [1, 4]])
    fracs = [frac1, frac2]
    gb = cart_grid(fracs, [5, 5])

    """
    ndim = np.asarray(nx).size
    physdims = kwargs.get("physdims", None)

    if physdims is None:
        physdims = nx
    elif np.asarray(physdims).size != ndim:
        raise ValueError("Physical dimension must equal grid dimension")

    # Call relevant method, depending on grid dimensions
    if ndim == 2:
        grids = structured._cart_grid_2d(fracs, nx, physdims=physdims)
    elif ndim == 3:
        grids = structured._cart_grid_3d(fracs, nx, physdims=physdims)
    else:
        raise ValueError("Only support for 2 and 3 dimensions")

    return grid_list_to_grid_bucket(grids, **kwargs)


def _tag_faces(grids, check_highest_dim=True):
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

    # Assume only one grid of highest dimension
    if check_highest_dim:
        assert len(grids[0]) == 1, "Must be exactly" "1 grid of dim: " + str(len(grids))

    for g_h in np.atleast_1d(grids[0]):
        bnd_faces = g_h.get_all_boundary_faces()
        domain_boundary_tags = np.zeros(g_h.num_faces, dtype=bool)
        domain_boundary_tags[bnd_faces] = True
        g_h.tags["domain_boundary_faces"] = domain_boundary_tags
        bnd_nodes, _, _ = sps.find(g_h.face_nodes[:, bnd_faces])

        # Boundary nodes of g_h in terms of global indices
        bnd_nodes_glb = g_h.global_point_ind[np.unique(bnd_nodes)]

        for g_dim in grids[1:-1]:
            for g in g_dim:
                # We find the global nodes of all boundary faces
                bnd_faces_l = g.get_all_boundary_faces()
                indptr = g.face_nodes.indptr
                fn_loc = mcolon.mcolon(indptr[bnd_faces_l], indptr[bnd_faces_l + 1])
                nodes_loc = g.face_nodes.indices[fn_loc]
                # Convert to global numbering
                nodes_glb = g.global_point_ind[nodes_loc]
                # We then tag each node as a tip node if it is not a global
                # boundary node
                is_tip = np.in1d(nodes_glb, bnd_nodes_glb, invert=True)
                # We reshape the nodes such that each column equals the nodes of
                # one face. If a face only contains global boundary nodes, the
                # local face is also a boundary face. Otherwise, we add a TIP tag.
                n_per_face = _nodes_per_face(g)
                is_tip = np.any(
                    is_tip.reshape((n_per_face, bnd_faces_l.size), order="F"), axis=0
                )

                g.tags["tip_faces"][bnd_faces_l[is_tip]] = True
                domain_boundary_tags = np.zeros(g.num_faces, dtype=bool)
                domain_boundary_tags[bnd_faces_l[np.logical_not(is_tip)]] = True
                g.tags["domain_boundary_faces"] = domain_boundary_tags


def _nodes_per_face(g):
    """
    Returns the number of nodes per face for a given grid
    """
    if ("TensorGrid" in g.name or "CartGrid" in g.name) and g.dim == 3:
        n_per_face = 4
    elif "TetrahedralGrid" in g.name:
        n_per_face = 3
    elif ("TensorGrid" in g.name or "CartGrid" in g.name) and g.dim == 2:
        n_per_face = 2
    elif "TriangleGrid" in g.name:
        n_per_face = 2
    elif ("TensorGrid" in g.name or "CartGrid" in g.name) and g.dim == 1:
        n_per_face = 1
    else:
        raise ValueError(
            "Can not find number of nodes per face for grid: " + str(g.name)
        )
    return n_per_face


def _assemble_in_bucket(grids, **kwargs):
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
            n_per_face = _nodes_per_face(hg)
            fn_loc = hg.face_nodes.indices.reshape(
                (n_per_face, hg.num_faces), order="F"
            )
            # Convert to global numbering
            fn = hg.global_point_ind[fn_loc]
            fn = np.sort(fn, axis=0)

            for lg in grids[dim + 1]:
                cell_2_face, cell = tools.obtain_interdim_mappings(lg, fn, n_per_face)
                if cell_2_face.size > 0:
                    face_cells = sps.csc_matrix(
                        (np.ones(cell.size, dtype=bool), (cell, cell_2_face)),
                        (lg.num_cells, hg.num_faces),
                    )

                    bucket.add_edge([hg, lg], face_cells)

    return bucket


def create_mortar_grids(gb, ensure_matching_face_cell=True, **kwargs):

    gb.add_edge_props("mortar_grid")
    # loop on all the nodes and create the mortar grids
    for e, d in gb.edges():
        lg = gb.nodes_of_edge(e)[0]
        # d['face_cells'].indices gives mappings into the lower dimensional
        # cells. Count the number of occurences for each cell.
        num_sides = np.bincount(d["face_cells"].indices)
        # Each cell should be found either twice (think a regular fracture
        # that splits a higher dimensional mesh), or once (the lower end of
        # a T-intersection, or both ends of an L-intersection).
        if ensure_matching_face_cell:
            assert np.all(num_sides == 1) or np.all(num_sides == 2)
        else:
            assert np.max(num_sides) < 3

        # If all cells are found twice, create two mortar grids
        if np.all(num_sides > 1):
            # we are in a two sides situation
            side_g = {
                mortar_grid.LEFT_SIDE: lg.copy(),
                mortar_grid.RIGHT_SIDE: lg.copy(),
            }
        else:
            # the tag name is just a place-holder we assume left side
            side_g = {mortar_grid.LEFT_SIDE: lg.copy()}
        d["mortar_grid"] = mortar_grid.MortarGrid(lg.dim, side_g, d["face_cells"])
