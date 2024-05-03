"""This is the main module for grid generation in fractured domains in 2D and 3D.

The module serves as the only necessary entry point to create a grid. It will thus
wrap interface to different mesh generators, pass options to the generators etc.

"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.fracs import split_grid, structured
from porepy.grids import mortar_grid
from porepy.grids.md_grid import MixedDimensionalGrid
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data
from porepy.utils import mcolon

logger = logging.getLogger(__name__)

mortar_sides = mortar_grid.MortarSides


def subdomains_to_mdg(
    subdomains: list[list[pp.Grid]],
    time_tot: Optional[float] = None,
    **kwargs,
) -> pp.MixedDimensionalGrid:
    """Convert a list of grids to a full mixed-dimensional grid.

    The list can come from several mesh constructors, both simplex and structured
    approaches uses this in 2D and 3D.

    The function can not be used on an arbitrary set of grids. They should contain
    information to glue grids together. This will be included for grids created by
    the standard mixed-dimensional grid constructors. Essentially, do not directly
    use this function unless you are knowledgeable about how it works.

    Parameters:
        subdomains: Nested lists of subdomains to enter into the mixed-dimensional grid.
            Sorted per dimension.
        time_tot: ``default=None``

            Start time for full mesh construction. Used for logging.
            If ``None``, no information on total time consumption is logged.
        **kwargs: Passed on to subfunctions.

    Returns:
        The resulting mixed-dimensional grid.

    """

    # Tag tip faces
    check_highest_dim = kwargs.get("check_highest_dim", False)
    _tag_faces(subdomains, check_highest_dim)
    logger.debug("Assemble mdg")
    tm_mdg = time.time()

    # Assemble the list of subdomain grids into a mixed-dimensional grid. This will
    # also identify pairs of neighboring grids (one dimension apart).
    mdg, sd_pair_to_face_cell_map = _assemble_mdg(subdomains)
    logger.debug("Done. Elapsed time " + str(time.time() - tm_mdg))

    logger.debug("Compute geometry")
    tm_geom = time.time()
    mdg.compute_geometry()

    logger.debug("Done. Elapsed time " + str(time.time() - tm_geom))
    logger.debug("Split fractures")
    tm_split = time.time()

    # Split faces and nodes in the grids of various dimensions
    mdg, node_pairs = split_grid.split_fractures(
        mdg, sd_pair_to_face_cell_map, **kwargs
    )
    logger.debug("Done. Elapsed time " + str(time.time() - tm_split))

    # Now that neighboring subdomains are identified, faces and nodes are split,
    # we are ready to create mortar grids on the interface between subdomains. These
    # will be added to the mixed-dimensional grid.
    create_interfaces(mdg, node_pairs)

    # Set projections to the boundary grids (this must be done after having
    # split the fracture faces, or else the projections will have the wrong dimension).
    mdg.set_boundary_grid_projections()

    if time_tot is not None:
        logger.info(
            "Mesh construction completed. Total time " + str(time.time() - time_tot)
        )

    return mdg


def cart_grid(
    fracs: list[np.ndarray], nx: np.ndarray, **kwargs
) -> pp.MixedDimensionalGrid:
    """Create a cartesian, fractured mixed-dimensional grid in 2 or 3 dimensions.

    Parameters:
        fracs: One list item for each fracture.

            Each item consists of an array of shape ``(nd, num_points)`` describing
            fracture vertices, where ``num_points`` is 2 for 2D domains, 4 for 3D
            domains. The fractures have to be rectangles (3D) or straight lines (2D)
            that align with the axis. The fractures may be intersecting. The
            fractures will snap to the closest grid faces.
        nx: Number of cells in each direction. Should be 2D or 3D.
        **kwargs: Available keyword arguments are

            - ``'physdims'``: A :obj:`~numpy.ndarray` containing the physical dimensions
              in each direction.
              Defaults to same as ``nx``, that is, cells of unit size.
              May also contain fracture tags, options for the grid, etc.

    Returns:
        A complete mixed-dimensional grid where all fractures are represented as
        lower dim grids. The higher dim fracture faces are split in two, and on the
        edges of the MixedDimensionalGrid graph the mapping from lower dim cells to
        higher dim faces are stored as ``face_cells``.

        Each face is given boolean tags depending on the type.

        - ``domain_boundary_faces``:
          All faces that lie on the domain boundary
          (i.e. should be given a boundary condition).
        - ``fracture_faces``:
          All faces that are split (i.e. has a connection to a lower dim grid).
        - ``tip_faces``:
          A boundary face that is neither on the domain boundary,
          nor coupled to a lower dimensional domain.

        The union of the above three is the tag ``boundary_faces``.

    """
    ndim = np.asarray(nx).size
    physdims = kwargs.get("physdims", None)

    if physdims is None:
        physdims = nx
    elif np.asarray(physdims).size != ndim:
        raise ValueError("Physical dimension must equal grid dimension")

    # Call relevant method, depending on grid dimensions
    if ndim == 2:
        subdomains = structured._cart_grid_2d(fracs, nx, physdims=physdims)
    elif ndim == 3:
        subdomains = structured._cart_grid_3d(fracs, nx, physdims=physdims)
    else:
        raise ValueError("Only support for 2 and 3 dimensions")

    return subdomains_to_mdg(subdomains, **kwargs)


def tensor_grid(
    fracs: list[np.ndarray],
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
    **kwargs,
) -> pp.MixedDimensionalGrid:
    """Create a cartesian, fractured, mixed-dimensional grid in 2 or 3 dimensions.

    Note:
        If only the ``x`` coordinate is passed, an error will be raised, as fractured
        tensor grids are not implemented in 1D.

    Parameters:
        fracs: One list item for each fracture.

            Each item consists of an array with ``shape=(nd, num_points)`` describing
            fracture vertices, where ``num_points`` is 2 for
            2D domains, 4 for 3D domains.

            The fractures have to be rectangles (3D) or straight lines (2D) that align
            with the axis.

            The fractures may be intersecting.
            The fractures will snap to closest grid faces
        x: Node coordinates in x-direction.
        y: Node coordinates in y-direction.
        z: Node coordinates in z-direction.
        **kwargs: May contain fracture tags, options for gridding, etc.

    Returns:
        A mixed-dimensional grid where all fractures are represented as
        lower-dimensional grids. The higher-dimensional fracture faces are split in
        two, and the mapping from lower-dimensional cells to higher-dimensional faces
        are stored as ``face_cells``.

        Each face is given boolean tags depending on the type.

        - ``domain_boundary_faces``:
          All faces that lie on the domain boundary
          (i.e., should be given a boundary condition).
        - ``fracture_faces``:
          All faces that are split (i.e., has a connection to a lower-dimensional grid).
        - ``tip_faces``:
          A boundary face that is not on the domain boundary,
          nor coupled to a lower-dimensional domain.

        The union of the above three is the tag ``boundary_faces``.

    """
    # Call relevant method, depending on grid dimensions
    if y is None:
        raise NotImplementedError("fractured tensor grids not implemented in 1D")
    elif z is None:
        subdomains = structured._tensor_grid_2d(fracs, x, y)
    else:
        subdomains = structured._tensor_grid_3d(fracs, x, y, z)

    return subdomains_to_mdg(subdomains, **kwargs)


def _tag_faces(grids, check_highest_dim=True):
    """Tag faces of grids.

    Three different tags are given to different types of faces:
        - ``NONE``: None of the below (i.e., an internal face).
        - ``DOMAIN_BOUNDARY``: All faces that lie on the domain boundary (i.e., should
            be given a boundary condition).
        - ``FRACTURE``: All faces that are split (i.e., has a connection to a lower dim
            grid).
        - ``TIP``: A boundary face that is neither on the domain boundary, nor coupled
            to a lower domentional domain.

    Parameters:
        grids: List of grids to be tagged. Sorted per dimension.
        check_highest_dim: If ``True``, we require there is a single mesh in the highest
            dimension. The test is useful, but should be waived for dfn meshes.
        tag_tip_node: If ``True``, nodes in the highest-dimensional grid are tagged.

    """

    # Assume only one grid of highest dimension
    if check_highest_dim:
        assert len(grids[0]) == 1, "Must be exactly" "1 grid of dim: " + str(len(grids))

    for g_h in np.atleast_1d(grids[0]):
        bnd_faces = g_h.get_all_boundary_faces()
        domain_boundary_tags = np.zeros(g_h.num_faces, dtype=bool)
        domain_boundary_tags[bnd_faces] = True
        g_h.tags["domain_boundary_faces"] = domain_boundary_tags

        # Pick out the face-node relation for the highest dimensional grid,
        # restricted to the faces on the domain boundary. This will be of use for
        # identifying tip faces for 2d grids below.
        fn_h = g_h.face_nodes[:, bnd_faces].tocsr()

        # Nodes on the boundary
        bnd_nodes, _, _ = sparse_array_to_row_col_data(fn_h)

        # Boundary nodes of g_h in terms of global indices
        bnd_nodes_glb = g_h.global_point_ind[np.unique(bnd_nodes)]

        # Find the nodes in g_h that are on the tip of a fracture (not counting
        # fracture endings on the global boundary). Do this by identifying tip nodes
        # among the grids of dimension dim_max - 1. Exclude tips that also occur on
        # other fractures (this will correspond to T or L-intersection, which look
        # like tips from individual fractures).

        # IMPLEMENTATION NOTE: To account for cases where not all global nodes are
        # present in g_h (e.g. for DFN-type grids), and avoid issues relating to
        # nodes in lower-dimensional grids that are not present in g_h, we store node
        # numbers instead of using booleans arrays on the nodes in g_h.

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

                # We reshape the nodes such that each column equals the nodes of one
                # face. If a face only contains global boundary nodes, the local face
                # is also a boundary face. Otherwise, we add a TIP tag. Note that we
                # only consider boundary faces, hence any is okay.
                n_per_face = _nodes_per_face(g)
                is_tip_face = np.any(
                    node_on_tip_not_global_bnd.reshape(
                        (n_per_face, bnd_faces_l.size), order="F"
                    ),
                    axis=0,
                )

                # Special case: In 2d, there may be fractures that are so close to a
                # corner of the domain that it has faces with nodes on different
                # surfaces of the global boundary. These are identified by the two
                # nodes (there will be 2 in 2d) not having any faces in the coarse
                # grid in common.
                if g.dim == 2:
                    assert n_per_face == 2
                    not_tip = np.where(np.logical_not(is_tip_face))[0]
                    for fi in not_tip:
                        g1 = fn_h[nodes_glb[2 * fi]].indices
                        g2 = fn_h[nodes_glb[2 * fi + 1]].indices
                        if np.intersect1d(g1, g2).size == 0:
                            is_tip_face[fi] = True

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
                    # For co-dimension 1, we also register those nodes in the host
                    # grid which correspond to the tip of a fracture. We use a
                    # slightly wider definition of a fracture tip in this context:
                    # Nodes that are on the domain boundary, but also part of a tip
                    # face (on the fracture) which extends into the domain are also
                    # considered to be tip nodes. Filtering away these will be
                    # simple, using the domain_boundary_nodes tag, if necessary.
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
        # Tag nodes that are on the tip of a fracture, and not involved in other
        # fractures.
        g_h.tags["node_is_fracture_tip"] = tip_tag

        on_any_tip = np.where(np.bincount(global_node_as_fracture_tip) > 0)[0]
        _, local_any_tip = pp.utils.setmembership.ismember_rows(
            on_any_tip, g_h.global_point_ind
        )
        tip_of_a_fracture = np.zeros_like(tip_tag)
        tip_of_a_fracture[local_any_tip] = True
        # Tag nodes that are on the tip of a fracture, independent of whether they
        # are actually tips of a fracture.
        g_h.tags["node_is_tip_of_some_fracture"] = tip_of_a_fracture


def _nodes_per_face(g):
    """Return the number of nodes per face for a given grid."""
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


def _assemble_mdg(
    subdomains, **kwargs
) -> tuple[MixedDimensionalGrid, dict[tuple[pp.Grid, pp.Grid], sps.spmatrix]]:
    """Create a :class:~`porepy.MixedDimensionalGrid` from a list of grids.

    Parameters:
        subdomains:

            A list of lists of grids. Each element in the list is a list of all grids
            of the same dimension. It is assumed that the grids are sorted from high
            dimensional grids to low dimensional grids. All grids must also have the
            mapping g.global_point_ind which maps the local nodes of the grid to the
            nodes of the highest dimensional grid.

    Returns:
        mdg: A mixed-dimensional grid, where the mapping ``face_cells`` are given to
            each edge. ``face_cells`` maps from lower-dim cells to higher-dim faces.
        sd_pair_to_face_cell_map: A dictionary of subdomains mapped to a face-cell map.
            The first item represents two neighboring subdomains. The second item is a
            mapping between faces in the high dimension subdomain and cells in the low
            dimension subdomain.

    """

    # Create a mixed-dimensional grid
    mdg = MixedDimensionalGrid()
    for sd_d in subdomains:
        mdg.add_subdomains(sd_d)

    sd_pair_to_face_cell_map: dict[tuple[pp.Grid, pp.Grid], sps.spmatrix] = {}

    # We now find the face_cell mappings.
    for dim in range(len(subdomains) - 1):
        # If there are no grids of dimension one less, continue.
        if len(subdomains[dim + 1]) == 0:
            continue

        # Loop over all grids of the higher dimension, look for lower-dimensional
        # grids where the cell of the lower-dimensional grid shares nodes with the
        # faces of the higher-dimensional grid. If this face-cell intersection is
        # non-empty, there is a coupling will be made between the higher and
        # lower-dimensional grid, and the face-to-cell relation will be saved.
        for hsd in subdomains[dim]:
            # We have to specify the number of nodes per face to generate a matrix of
            # the nodes of each face.
            n_per_face = _nodes_per_face(hsd)

            # Get the face-node relation for the higher-dimensional grid, stored with
            # one column per face
            fn_loc = hsd.face_nodes.indices.reshape(
                (n_per_face, hsd.num_faces), order="F"
            )
            # Convert to global numbering
            fn = hsd.global_point_ind[fn_loc]
            fn = np.sort(fn, axis=0)

            # Get a cell-node relation for the lower-dimensional grids. It turns out
            # that to do the intersection between the node groups is costly (mainly
            # because the call to ismember_rows below does a unique over all
            # faces-nodes in the higher-dimensional grid). To save a lot of time,
            # we first group cell-nodes for all lower- dimensional grids,
            # do the intersection once, and then process the results.

            # The treatment of the lower-dimensional grids is a bit special for point
            # grids (else below)
            if hsd.dim > 1:
                # Data structure for cell-nodes
                cn = []
                # Number of cells per grid. Will be used to define offsets for
                # cell-node relations for each grid, hence initialize with zero.
                num_cn = [0]
                for lg in subdomains[dim + 1]:
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
                # 0d grid is much easier, although getting hold of the single point
                # index is a bit technical
                cn_all = np.array(
                    [
                        np.atleast_1d(lg.global_point_ind)[0]
                        for lg in subdomains[dim + 1]
                    ]
                )
                cell_node_offsets = np.arange(cn_all.size + 1)
                # Ensure that face-node relation is 1d in this case
                fn = fn.ravel()

            # Find intersection between cell-node and face-nodes. Node nede to sort
            # along 0-axis, we know we've done that above.
            is_mem, cell_2_face = pp.utils.setmembership.ismember_rows(
                cn_all, fn, sort=False
            )
            # Now, for each lower-dimensional grid, either all of none of the cells
            # have been identified as faces in the higher-dimensional grid.
            # (If hg is the highest dimension, there should be a match for all grids
            # in lsd, however, if hsd is itself a fracture, lsd is an intersection which
            # need not involve hg).

            # Special treatment if not all cells were found: cell_2_face then only
            # contains those cells found; to make them conincide with the indices
            # of is_mem (that is, as the faces are stored in cn_all), we expand the
            # cell_2_face array
            if is_mem.size != cell_2_face.size:
                # If something goes wrong here, we will likely get an index of -1
                # when initializing the sparse matrix below - that should be a
                # clear indicator.
                tmp = -np.ones(is_mem.size, dtype=int)
                tmp[is_mem] = cell_2_face
                cell_2_face = tmp

            # Loop over all lower-dimensional grids; find the cells that had matching
            # faces in hg (should be either none or all the cells).
            for counter, lsd in enumerate(subdomains[dim + 1]):
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
                    shape=(lsd.num_cells, hsd.num_faces),
                )
                # Add the pairing of subdomains and the cell-face map to the list
                sd_pair_to_face_cell_map[(hsd, lsd)] = face_cell_map

    return mdg, sd_pair_to_face_cell_map


def create_interfaces(
    mdg: pp.MixedDimensionalGrid,
    sd_pair_to_face_cell_map: dict[tuple[pp.Grid, pp.Grid], sps.spmatrix],
) -> None:
    """Create interfaces for a given mixed-dimensional grid.

    Parameters:
        mdg: The mixed-dimensional grid where the interfaces are built.
        sd_pair_to_face_cell_map: A dictionary of subdomain-pairs mapped to a face-cell
            map.

            The keys represent two neighboring subdomains.

            The values are mappings between faces in the higher-dimensional subdomain
            and cells in the lower-dimension subdomain.

    """

    # loop on all the subdomain pairs and create the mortar grids
    for sd_pair, face_cells in sd_pair_to_face_cell_map.items():
        sd_h, sd_l = sd_pair

        # face_cells.indices gives mappings into the lower dimensional cells. Count
        # the number of occurrences for each cell.
        num_sides = np.bincount(face_cells.indices)

        if np.max(num_sides) > 2:
            # Each cell should be found either twice (think a regular fracture that
            # splits a higher dimensional mesh), or once (the lower end of a
            # T-intersection, or both ends of an L-intersection).
            raise ValueError(
                """Found low-dimensional cell which corresponds to
                    too many high-dimensional faces."""
            )

        # If all cells are found twice, create two mortar grids
        if np.all(num_sides > 1):
            # we are in a two sides situation
            side_g = {
                mortar_sides.LEFT_SIDE: sd_l.copy(),
                mortar_sides.RIGHT_SIDE: sd_l.copy(),
            }
        else:
            # the tag name is just a place-holder we assume left side
            side_g = {mortar_sides.LEFT_SIDE: sd_l.copy()}
        mg = mortar_grid.MortarGrid(sd_l.dim, side_g, face_cells)

        mdg.add_interface(mg, sd_pair, face_cells)
