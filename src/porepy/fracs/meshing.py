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
from porepy.fracs import split_grid, structured
from porepy.grids import mortar_grid
from porepy.grids.grid_bucket import GridBucket
from porepy.utils import mcolon

logger = logging.getLogger(__name__)
module_sections = ["gridding"]

mortar_sides = mortar_grid.MortarSides


@pp.time_logger(sections=module_sections)
def grid_list_to_grid_bucket(
    grids: List[List[pp.Grid]], time_tot: float = None, **kwargs
) -> pp.GridBucket:
    """Convert a list of grids to a full GridBucket.

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


@pp.time_logger(sections=module_sections)
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


@pp.time_logger(sections=module_sections)
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
            tag_tip_node (bool, default=True): If True, nodes in the highest-dimensional grid
    are tagged
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

        # Find the nodes in g_h that are on the tip of a fracture (not counting
        # fracture endings on the global boundary). Do this by identifying tip nodes
        # among the grids of dimension dim_max - 1. Exclude tips that also occur on
        # other fractures (this will correspond to T or L-intersection, which look
        # like tips from individual fractures).

        # IMPLEMENTATION NOTE: To account for cases where not all global nodes are
        # present in g_h (e.g. for DFN-type grids), and avoid issues relating to nodes
        # in lower-dimensional grids that are not present in g_h, we store node numbers
        # instead of using booleans arrays on the nodes in g_h.

        # Keep track of nodes in g_h that correspond to tip nodes of a fracture.
        global_node_as_fracture_tip = np.array([], dtype=int)
        # Also count the number of occurences of nodes on fractures
        num_occ_nodes = np.array([], dtype=int)

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
                node_on_tip_not_global_bnd = np.in1d(
                    nodes_glb, bnd_nodes_glb, invert=True
                )

                # We reshape the nodes such that each column equals the nodes of
                # one face. If a face only contains global boundary nodes, the
                # local face is also a boundary face. Otherwise, we add a TIP tag.
                # Note that we only consider boundary faces, hence any is okay.
                n_per_face = _nodes_per_face(g)
                is_tip_face = np.any(
                    node_on_tip_not_global_bnd.reshape(
                        (n_per_face, bnd_faces_l.size), order="F"
                    ),
                    axis=0,
                )

                # Tag faces on tips and boundaries
                g.tags["tip_faces"][bnd_faces_l[is_tip_face]] = True
                domain_boundary_tags = np.zeros(g.num_faces, dtype=bool)
                domain_boundary_tags[bnd_faces_l[np.logical_not(is_tip_face)]] = True
                g.tags["domain_boundary_faces"] = domain_boundary_tags

                # Also tag the nodes of the lower-dimensional grid that are on a tip
                is_tip_node = np.zeros(g.num_nodes, dtype=bool)
                is_tip_node[nodes_loc[node_on_tip_not_global_bnd]] = True
                g.tags["tip_nodes"] = is_tip_node

                if g.dim == g_h.dim - 1:
                    # For co-dimension 1, we also register those nodes in the host grid which
                    # are correspond to the tip of a fracture. We use a slightly wider
                    # definition of a fracture tip in this context: Nodes that are on the
                    # domain boundary, but also part of a tip face (on the fracture) which
                    # extends into the domain are also considered to be tip nodes. Filtering
                    # away these will be simple, using the domain_boundary_nodes tag, if
                    # necessary.
                    nodes_on_fracture_tip = np.unique(
                        nodes_glb.reshape((n_per_face, bnd_faces_l.size), order="F")[
                            :, is_tip_face
                        ]
                    )

                    global_node_as_fracture_tip = np.hstack(
                        (global_node_as_fracture_tip, nodes_on_fracture_tip)
                    )
                    # Count all global nodes used in this
                    num_occ_nodes = np.hstack((num_occ_nodes, g.global_point_ind))

        # The tip nodes should both be on the tip of a fracture, and not be present
        # on other fractures.
        may_be_tip = np.where(np.bincount(global_node_as_fracture_tip) == 1)[0]
        occurs_once = np.where(np.bincount(num_occ_nodes) == 1)[0]
        true_tip = np.intersect1d(may_be_tip, occurs_once)

        # Take the intersection between tip nodes and the nodes in this fracture.
        _, local_true_tip = pp.utils.setmembership.ismember_rows(
            true_tip, g_h.global_point_ind
        )
        tip_tag = np.zeros(g_h.num_nodes, dtype=bool)
        tip_tag[local_true_tip] = True
        # Tag nodes that are on the tip of a fracture, and not involved in other fractures
        g_h.tags["node_is_fracture_tip"] = tip_tag

        on_any_tip = np.where(np.bincount(global_node_as_fracture_tip) > 0)[0]
        _, local_any_tip = pp.utils.setmembership.ismember_rows(
            on_any_tip, g_h.global_point_ind
        )
        tip_of_a_fracture = np.zeros_like(tip_tag)
        tip_of_a_fracture[local_any_tip] = True
        # Tag nodes that are on the tip of a fracture, independent of whether it is
        g_h.tags["node_is_tip_of_some_fracture"] = tip_of_a_fracture


@pp.time_logger(sections=module_sections)
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


@pp.time_logger(sections=module_sections)
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
        # If there are no grids of dimension one less, continue.
        if len(grids[dim + 1]) == 0:
            continue

        # Loop over all grids of the higher dimension, look for lower-dimensional
        # grids where the cell of the lower-dimensional grid shares nodes with
        # the faces of the higher-dimensional grid. If this face-cell intersection
        # is non-empty, there is a coupling will be made between the higher and
        # lower-dimensional grid, and the face-to-cell relation will be saved.
        for hg in grids[dim]:

            # We have to specify the number of nodes per face to generate a
            # matrix of the nodes of each face.
            n_per_face = _nodes_per_face(hg)

            # Get the face-node relation for the higher-dimensional grid,
            # stored with one column per face
            fn_loc = hg.face_nodes.indices.reshape(
                (n_per_face, hg.num_faces), order="F"
            )
            # Convert to global numbering
            fn = hg.global_point_ind[fn_loc]
            fn = np.sort(fn, axis=0)

            # Get a cell-node relation for the lower-dimensional grids.
            # It turns out that to do the intersection between the node groups
            # is costly (mainly because the call to ismember_rows below does
            # a unique over all faces-nodes in the higher-dimensional grid).
            # To save a lot of time, we first group cell-nodes for all lower-
            # dimensional grids, do the intersection once, and then process
            # the results.

            # The treatmnet of the lower-dimensional grids is a bit special
            # for point grids (else below)
            if hg.dim > 1:
                # Data structure for cell-nodes
                cn = []
                # Number of cells per grid. Will be used to define offsets
                # for cell-node relations for each grid, hence initialize with
                # zero.
                num_cn = [0]
                for lg in grids[dim + 1]:
                    # Local cell-node relation
                    cn_loc = lg.cell_nodes().indices.reshape(
                        (n_per_face, lg.num_cells), order="F"
                    )
                    cn.append(np.sort(lg.global_point_ind[cn_loc], axis=0))
                    num_cn.append(lg.num_cells)

                # Stack all cell-nodes, and define offset array
                cn_all = np.hstack([c for c in cn])
                cell_node_offsets = np.cumsum(num_cn)
            else:
                # 0d grid is much easier, although getting hold of the single
                # point index is a bit technical
                cn_all = np.array(
                    [np.atleast_1d(lg.global_point_ind)[0] for lg in grids[dim + 1]]
                )
                cell_node_offsets = np.arange(cn_all.size + 1)
                # Ensure that face-node relation is 1d in this case
                fn = fn.ravel()

            # Find intersection between cell-node and face-nodes.
            # Node nede to sort along 0-axis, we know we've done that above.
            is_mem, cell_2_face = pp.utils.setmembership.ismember_rows(
                cn_all, fn, sort=False
            )
            # Now, for each lower-dimensional grid, either all of none of the cells
            # have been identified as faces in the higher-dimensional grid.
            # (If hg is the highest dimension, there should be a match for all grids
            # in lg, however, if hg is itself a fracture, lg is an intersection which
            # need not involve hg).

            # Special treatment if not all cells were found: cell_2_face then only
            # contains those cells found; to make them conincide with the indices
            # of is_mem (that is, as the faces are stored in cn_all), we expand the
            # cell_2_face array
            if is_mem.size != cell_2_face.size:
                # If something goes wrong here, we will likely get an index of -1
                # when initializing the sparse matrix below - that should be a
                # clear indicator.
                tmp = -np.ones(is_mem.size, dtype=np.int)
                tmp[is_mem] = cell_2_face
                cell_2_face = tmp

            # Loop over all lower-dimensional grids; find the cells that had matching
            # faces in hg (should be either none or all the cells).
            for counter, lg in enumerate(grids[dim + 1]):
                # Indices of this grid in is_mem and cell_2_face (thanks to the above
                # expansion, involving tmp)
                ind = slice(cell_node_offsets[counter], cell_node_offsets[counter + 1])
                loc_mem = is_mem[ind]
                # If no match, continue
                if np.sum(loc_mem) == 0:
                    continue
                # If some match, all should be matches. If this goes wrong, there is
                # likely something wrong with the mesh.
                assert np.all(loc_mem)
                # Create mapping between faces and cells.
                face_cell_map = sps.csc_matrix(
                    (
                        np.ones(loc_mem.size, dtype=bool),
                        (np.arange(loc_mem.size), cell_2_face[ind]),
                    ),
                    shape=(lg.num_cells, hg.num_faces),
                )
                # Define the new edge.
                bucket.add_edge([hg, lg], face_cell_map)

    return bucket


@pp.time_logger(sections=module_sections)
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
                mortar_sides.LEFT_SIDE: lg.copy(),
                mortar_sides.RIGHT_SIDE: lg.copy(),
            }
        else:
            # the tag name is just a place-holder we assume left side
            side_g = {mortar_sides.LEFT_SIDE: lg.copy()}
        d["mortar_grid"] = mortar_grid.MortarGrid(lg.dim, side_g, d["face_cells"])
