"""This module contains various functions to find overlaps between grid cells.

The module is primarily intended for replacing individual grids in the
mixed-dimensional grid. It is called mostly inside
:class:`~porepy.grids.mortar_grid.MortarGrid`.

That is, the methods herein should as a rule not be invoked directly.

"""
import logging
from typing import Optional

import numpy as np
import scipy.sparse as sps
from typing_extensions import Literal

import porepy as pp
from porepy.grids.structured import TensorGrid
from porepy.utils.setmembership import ismember_rows, unique_columns_tol

import pdb

logger = logging.getLogger(__name__)


def match_1d(
    new_g: pp.Grid,
    old_g: pp.Grid,
    tol: float,
    scaling: Optional[Literal["averaged", "integrated"]] = None,
) -> sps.csr_matrix:
    """Obtain mappings between the cells of non-matching 1D grids.

    The overlapping is identified as a sparse matrix which maps from cells in the old to
    the new grid.

    It is assumed that the two grids are aligned, with common start and endpoints.

    Parameters:
        new_g: Target grid for the mapping. Should have dimension 1.
        old_g: Original grid. Should have dimension 1.
        tol: Tolerance used to filter away false overlaps caused by
             numerical errors. Should be scaled relative to the cell size.
        scaling: ``default=None``.

            Control weights of the returned matrix, see return values for specification.

    Returns:
        Mapping from the cells in the old to the new grid.

        The values in the matrix depend on the parameter ``scaling``.

        If set to 'averaged', a mapping fit for intensive quantities (e.g., pressure) is
        returned (all rows sum to unity).

        If set to 'integrated', the matrix is a mapping for extensive quantities (column
        sum is 1).

        If None, the matrix elements are 1 for cell-pairs (new and old grid) that
        overlap. Overlaps with areas less than the parameter ``tol`` will be ignored.

    """
    # Cell-node relation between grids - we know there are two nodes per cell
    cell_nodes_new = new_g.cell_nodes()
    cell_nodes_old = old_g.cell_nodes()
    nodes_new = pp.utils.mcolon.mcolon(
        cell_nodes_new.indptr[0:-1], cell_nodes_new.indptr[1:]
    )
    nodes_old = pp.utils.mcolon.mcolon(
        cell_nodes_old.indptr[0:-1], cell_nodes_old.indptr[1:]
    )

    # Reshape so that the nodes of cells are stored column-wise
    lines_new = cell_nodes_new.indices[nodes_new].reshape((2, -1), order="F")
    lines_old = cell_nodes_old.indices[nodes_old].reshape((2, -1), order="F")

    p_new = new_g.nodes
    p_old = old_g.nodes

    # Compute the intersection between the two tessellations. intersect is a list, every
    # list member is a tuple with overlapping cells in grid 1 and 2, and their common
    # area.
    intersect = pp.intersections.line_tessellation(p_new, p_old, lines_new, lines_old)

    num = len(intersect)
    new_g_ind = np.zeros(num, dtype=int)
    old_g_ind = np.zeros(num, dtype=int)
    weights = np.zeros(num)

    for ind, i in enumerate(intersect):
        new_g_ind[ind] = i[0]
        old_g_ind[ind] = i[1]
        weights[ind] = i[2]

    # The weights as computed from the intersection algorithm gives the volumes of the
    # intersected cells. Depending on the specified scaling, the weights should be
    # modified.
    if scaling == "averaged":
        weights /= new_g.cell_volumes[new_g_ind]
    elif scaling == "integrated":
        weights /= old_g.cell_volumes[old_g_ind]
    elif scaling is None:
        mask = weights > tol
        new_g_ind = new_g_ind[mask]
        old_g_ind = old_g_ind[mask]
        weights = np.ones_like(new_g_ind)

    return sps.coo_matrix(
        (weights, (new_g_ind, old_g_ind)), shape=(new_g.num_cells, old_g.num_cells)
    ).tocsr()

    # return sps.csr_matrix(
    #     (weights, (old_g_ind, new_g_ind))
    # )  ### OLD version was this


def match_2d(
    new_g: pp.Grid,
    old_g: pp.Grid,
    tol: float,
    scaling: Optional[Literal["averaged", "integrated"]] = None,
) -> sps.csr_matrix:
    """Match two simplex tessellations to identify overlapping cells.

    The overlaps are identified as a sparse matrix which maps from cells in the old to
    the new grid.

    It is assumed that the two grids are aligned, with common start and endpoints.

    Parameters:
        new_g: Target grid for the mapping. Should have dimension 2.
        old_g: Original grid. Should have dimension 2.
        tol: Tolerance used to filter away false overlaps caused by
             numerical errors. Should be scaled relative to the cell size.
        scaling: ``default=None``

            Control weights of the returned matrix, see return values for specification.

    Returns:
        Mapping from the cells in the old to the new grid.

        The values in the matrix depends on the parameter scaling.

        If set to 'averaged', a mapping fit for intensive quantities (e.g., pressure)
        is returned (all rows sum to unity).

        If set to 'integrated', the matrix is a mapping for extensive quantities (column
        sum is 1).

        If None, the matrix elements are 1 for cell-pairs (new and old grid) that
        overlap; overlaps with areas less than the parameter tol will be ignored.

    """

    def proj_pts(p, center, normal):
        """Project points to the 2d plane defined by normal and center them around
        center.

        """
        rot = pp.map_geometry.project_plane_matrix(p - center, normal)
        return rot.dot(p - center)[:2]

    # Represent the cells in terms of their vertexes. This representation will be passed
    # to the external library shapely, which has efficient methods for matching
    # tessallations.
    #
    # IMPLEMENTATION NOTE: For now, the interface to shapely can only deal with simplex
    # cells. This is not a limitation of shapely itself, which can handle general
    # polygons. Thus the below restrictions (look for ValueError) can be relaxed by
    # modifying the interface pp.intersections.triangulations.
    cn_new = new_g.cell_nodes().tocsc()
    if not np.all(np.diff(cn_new.indptr) == (new_g.dim + 1)):
        # See above implementation note for how to relax this restriction.
        raise ValueError(
            "Matching of 2d grids has only been implemented for simplex grids."
        )

    cn_new_array = cn_new.indices.reshape((new_g.dim + 1, new_g.num_cells), order="F")

    cn_old = old_g.cell_nodes().tocsc()
    if not np.all(np.diff(cn_old.indptr) == (new_g.dim + 1)):
        # See above implementation note for how to relax this restriction.
        raise ValueError(
            "Matching of 2d grids has only been implemented for simplex grids."
        )

    cn_old_array = cn_old.indices.reshape((old_g.dim + 1, old_g.num_cells), order="F")

    # Center points around mean
    cc = np.mean(new_g.nodes, axis=1).reshape((3, 1))
    # Calculate common normal for both grids
    n = pp.map_geometry.compute_normal(new_g.nodes - cc)
    n_old = pp.map_geometry.compute_normal(old_g.nodes - cc)
    if not (np.allclose(n, n_old) or np.allclose(n, -n_old)):
        raise ValueError("The new and old grid must lie in the same plane")

    # Calculate intersection
    isect = pp.intersections.triangulations(
        proj_pts(new_g.nodes, cc, n),
        proj_pts(old_g.nodes, cc, n),
        cn_new_array,
        cn_old_array,
    )

    num = len(isect)
    new_g_ind = np.zeros(num, dtype=int)
    old_g_ind = np.zeros(num, dtype=int)
    weights = np.zeros(num)

    for ind, i in enumerate(isect):
        new_g_ind[ind] = i[0]
        old_g_ind[ind] = i[1]
        weights[ind] = i[2]

    # The weights as computed from the intersection algorithm gives the volumes of the
    # intersected cells. Depending on the specified scaling, the weights should be
    # modified.
    if scaling == "averaged":
        weights /= new_g.cell_volumes[new_g_ind]
    elif scaling == "integrated":
        weights /= old_g.cell_volumes[old_g_ind]
    elif scaling is None:
        mask = weights > tol
        new_g_ind = new_g_ind[mask]
        old_g_ind = old_g_ind[mask]
        weights = np.ones_like(new_g_ind)

    return sps.coo_matrix(
        (weights, (new_g_ind, old_g_ind)), shape=(new_g.num_cells, old_g.num_cells)
    ).tocsr()


def match_grids_along_1d_mortar(
    mg: pp.MortarGrid,
    g_new: pp.Grid,
    g_old: pp.Grid,
    tol: float,
    scaling: Literal["averaged", "integrated"],
) -> sps.csr_matrix:
    """Match the faces of two 2D grids along a 1D mortar grid.

    The function identifies faces on the 1D segment specified by the MortarGrid, and
    finds the area weights of the matched faces. Both sides of the mortar grid are taken
    care of.

    Note:
        The returned matrix can be used to update
        :attr:`porepy.grids.mortar_grid.MortarGrid._primary_to_mortar_int`
        by right multiplication.

    Parameters:
        mg: MortarGrid that specifies the target 1D line. Must be of dimension 1.
        g_new: New 2D grid. Should have faces split along the 1D line.
        g_old: Old 2D grid. The mappings in mg from mortar to
            primary should be set for this grid.
        tol: Tolerance used in comparison of geometric quantities.
        scaling: Control weights of the returned matrix, see return
            values of :func:`match_2d` for specification.

    Raises:
        ValueError: If the matching procedure goes wrong.

    Returns:
        Mapping from the new to the old grid.

    """

    # print("\n\n inside match_grids_along_1d_mortar")

    # IMPLEMENTATION NOTE: Contrary to the related methods match_1d/match_2d, the
    # scaling argument is not permitted to be None in this function. This is by design,
    # it is less clear how to realize such a scaling in this case.

    # IMPLEMENTATION NOTE:
    # The algorithm is technical, partly because we also need to differ between
    # the left and right side of the segment, as these will belong to different
    # mortar grids.
    #
    # The main steps are:
    #   1) Identify faces in the old grid along the segment via the existing
    #      mapping between mortar grid and higher dimensional grid. Use this
    #      to define the geometry of the segment.
    #   2) Define positive and negative side of the segment, and split cells
    #      and faces along the segment according to this criterion.
    #   3) For all sides (pos, neg), pick out faces in the old and new grid,
    #      and match them up. Extend the mapping to go from all faces in the
    #      two grids.
    #
    # Known weak points: Identification of geometric objects, in particular
    # points, is based on a geometric tolerance. For very fine, or bad, grids
    # this may give trouble.

    def cells_from_faces(g, fi):
        # Find cells of faces, specified by face indices fi.
        # It is assumed that fi is on the boundary, e.g. there is a single
        # cell for each element in fi.
        f, ci, _ = sps.find(g.cell_faces[fi])
        if f.size != fi.size:
            raise ValueError("We assume fi are boundary faces")

        ismem, ind_map = ismember_rows(fi, fi[f], sort=False)
        if not np.all(ismem):
            raise ValueError

        return ci[ind_map]

    def create_1d_from_nodes(nodes):
        # From a set of nodes, create a 1d grid. duplicate nodes are removed
        # and we verify that the nodes are indeed colinear
        if not pp.geometry_property_checks.points_are_collinear(nodes, tol=tol):
            raise ValueError("Nodes are not colinear")
        sort_ind = pp.map_geometry.sort_points_on_line(nodes, tol=tol)
        n = nodes[:, sort_ind]
        unique_nodes, _, _ = unique_columns_tol(n, tol=tol)
        g = TensorGrid(np.arange(unique_nodes.shape[1]))
        g.nodes = unique_nodes
        g.compute_geometry()
        return g, sort_ind

    def nodes_of_faces(g, fi):
        # Find nodes of a set of faces.
        f = np.zeros(g.num_faces)
        f[fi] = 1
        nodes = np.where(g.face_nodes * f > 0)[0]
        return nodes

    def face_to_cell_map(g_2d, g_1d, loc_faces, loc_nodes):
        # Match faces in a 2d grid and cells in a 1d grid by identifying face-nodes and
        # cell-node relations.
        # loc_faces are faces in 2d grid that are known to coincide with cells.
        # loc_nodes are indices of 2d nodes along the segment, sorted so that the
        # ordering coincides with nodes in 1d grid

        # face-node relation in higher dimensional grid
        fn = g_2d.face_nodes.indices.reshape((g_2d.dim, g_2d.num_faces), order="F")
        # Reduce to faces along segment
        fn_loc = fn[:, loc_faces]

        # Mapping from global (2d) indices to the local indices used in 1d grid. This
        # also account for a sorting of the nodes, so that the nodes.
        ind_map = np.zeros(g_2d.num_faces, dtype=int)
        ind_map[loc_nodes] = np.arange(loc_nodes.size)

        # Face-node in local indices
        fn_loc = ind_map[fn_loc]

        # Handle special case
        if loc_faces.size == 1:
            fn_loc = fn_loc.reshape((2, 1))

        # Cell-node relation in 1d
        cn = g_1d.cell_nodes().indices.reshape((2, g_1d.num_cells), order="F")

        # Find cell index of each face
        ismem, ind = ismember_rows(fn_loc, cn)

        # Quality check, the grids should be conforming
        if not np.all(ismem):
            raise ValueError

        return ind

    # First create a virtual 1d grid along the line, using nodes from the old grid
    # Identify faces in the old grid that is on the boundary
    _, faces_on_boundary_old, _ = sps.find(mg._primary_to_mortar_int)
    # Find the nodes of those faces
    nodes_on_boundary_old = nodes_of_faces(g_old, faces_on_boundary_old)
    nodes_1d_old = g_old.nodes[:, nodes_on_boundary_old]

    # Normal vector of the line. Somewhat arbitrarily chosen as the first one.
    # This may be prone to rounding errors.
    normal = g_old.face_normals[:, faces_on_boundary_old[0]].reshape((3, 1))

    # Create first version of 1d grid, we really only need start and endpoint
    g_aux, _ = create_1d_from_nodes(nodes_1d_old)

    # Start, end and midpoint
    start = g_aux.nodes[:, 0]
    end = g_aux.nodes[:, -1]
    mp = 0.5 * (start + end).reshape((3, 1))

    # Find cells in 2d close to the segment
    bound_cells_old = cells_from_faces(g_old, faces_on_boundary_old)
    # This may occur if the mortar grid is one sided (T-intersection)
    #    assert bound_cells_old.size > 1,
    #    'Have not implemented this. Not difficult though'
    # Vector from midpoint to cell centers. Check which side the cells are on
    # relative to normal vector.
    # We are here assuming that the segment is not too curved (due to rounding
    # errors). Pain to come.
    cc_old = g_old.cell_centers[:, bound_cells_old]
    side_old = np.sign(np.sum(((cc_old - mp) * normal), axis=0))

    # Find cells on the positive and negative side, relative to the positioning
    # in cells_from_faces
    pos_side_old = np.where(side_old > 0)[0]
    neg_side_old = np.where(side_old < 0)[0]
    if pos_side_old.size + neg_side_old.size != side_old.size:
        raise ValueError

    both_sides_old = [pos_side_old, neg_side_old]

    # Then virtual 1d grid for the new grid. This is a bit more involved, since we need
    # to identify the nodes by their coordinates. This part will be prone to rounding
    # errors, in particular for badly shaped cells.
    nodes_new = g_new.nodes

    nodes_1d_new = g_new.nodes[:, nodes_on_boundary_old]  ###
    g_aux_new, _ = create_1d_from_nodes(nodes_1d_new)  ###
    start_new = g_aux_new.nodes[:, 0]  ###
    end_new = g_aux_new.nodes[:, -1]  ###

    # Represent the 1d line by its start and end point, as pulled from the old 1d grid
    # (known coordinates).
    # Find distance from the nodes to the line defined by the mortar grid.
    # dist, _ = pp.distances.points_segments(nodes_new, start, end)
    dist, _ = pp.distances.points_segments(nodes_new, start_new, end_new)  ###
    # Look for points in the new grid with a small distance to the line.
    hit = np.argwhere(dist.ravel() < tol).reshape((1, -1))[0]

    # Depending on geometric tolerance and grid quality, hit may contain nodes that are
    # close to the 1d line, but not on it To improve the results, also require that the
    # faces are boundary faces

    # We know we are in 2d, thus all faces have two nodes We can do the same trick in
    # 3d, provided we have simplex grids but this will fail on Cartesian or polyhedral
    # grids
    fn = g_new.face_nodes.indices.reshape((2, g_new.num_faces), order="F")
    fn_in_hit = np.isin(fn, hit)
    # Faces where all points are found in hit
    faces_by_hit = np.where(np.all(fn_in_hit, axis=0))[0]
    faces_on_boundary_new = np.where(g_new.tags["fracture_faces"].ravel())[0]
    # Only consider faces both in hit, and that are boundary
    faces_on_boundary_new = np.intersect1d(faces_by_hit, faces_on_boundary_new)

    # Cells along the segment, from the new grid
    bound_cells_new = cells_from_faces(g_new, faces_on_boundary_new)
    #    assert bound_cells_new.size > 1,
    #    'Have not implemented this. Not difficult though'
    cc_new = g_new.cell_centers[:, bound_cells_new]
    side_new = np.sign(np.sum(((cc_new - mp) * normal), axis=0))

    pos_side_new = np.where(side_new > 0)[0]
    neg_side_new = np.where(side_new < 0)[0]
    if pos_side_new.size + neg_side_new.size != side_new.size:
        raise ValueError

    both_sides_new = [pos_side_new, neg_side_new]

    # Mapping matrix.
    matrix = sps.coo_matrix((g_old.num_faces, g_new.num_faces))

    for so, sn in zip(both_sides_old, both_sides_new):
        if len(sn) == 0 or len(so) == 0:
            # EK: Not sure how this would happen
            continue

        # Pick out faces along boundary in old grid, uniquify nodes, and
        # define auxiliary grids
        loc_faces_old = faces_on_boundary_old[so]
        loc_nodes_old = np.unique(nodes_of_faces(g_old, loc_faces_old))
        g_aux_old, sort_ind_old = create_1d_from_nodes(g_old.nodes[:, loc_nodes_old])

        # Similar for new grid
        loc_faces_new = faces_on_boundary_new[sn]
        loc_nodes_new = np.unique(fn[:, loc_faces_new])
        g_aux_new, sort_ind_new = create_1d_from_nodes(nodes_new[:, loc_nodes_new])

        # Map from global faces to faces along segment in old grid
        n_loc_old = loc_faces_old.size
        face_map_old = sps.coo_matrix(
            (np.ones(n_loc_old), (np.arange(n_loc_old), loc_faces_old)),
            shape=(n_loc_old, g_old.num_faces),
        )

        # Map from global faces to faces along segment in new grid
        n_loc_new = loc_faces_new.size
        face_map_new = sps.coo_matrix(
            (np.ones(n_loc_new), (np.arange(n_loc_new), loc_faces_new)),
            shape=(n_loc_new, g_new.num_faces),
        )

        # print("\nface_map_old = ", face_map_old)
        # print("\nface_map_new = ", face_map_new)

        # Map from faces along segment in old to new grid. Consists of three stages:
        # faces in old to cells in 1d version of old, between 1d cells in old and new,
        # cells in new to faces in new

        # From faces to cells in old grid
        rows = face_to_cell_map(
            g_old, g_aux_old, loc_faces_old, loc_nodes_old[sort_ind_old]
        )
        cols = np.arange(rows.size)
        face_to_cell_old = sps.coo_matrix((np.ones(rows.size), (rows, cols)))

        # Mapping between cells in 1d grid.
        # Note the order here: The old grid is the target (we want a mapping from the
        # new to the old grid).
        between_cells = match_1d(g_aux_old, g_aux_new, tol, scaling)

        # print("\nbetween_cells = ", between_cells)

        # From faces to cell in new grid
        rows = face_to_cell_map(
            g_new, g_aux_new, loc_faces_new, loc_nodes_new[sort_ind_new]
        )
        cols = np.arange(rows.size)
        face_to_cell_new = sps.coo_matrix((np.ones(rows.size), (rows, cols)))

        # Composite mapping from faces in new 2d grid to faces in old 2d grid.
        # Only faces on the boundary of the 1d grid.
        face_map_segment = face_to_cell_old.T * between_cells * face_to_cell_new

        # Extend face-map to go from all faces in the new grid to all faces in the
        # old one.
        face_map = face_map_old.T * face_map_segment * face_map_new

        matrix += face_map

        # print("\nPorepy New ============================")
        # print("so = ", so)
        # print("sn = ", sn)
        # print("face_map_old.T = ", face_map_old.T)
        # print("face_map_segment = ", face_map_segment)
        # print("face_map_new = ", face_map_new)

        # print("matrix = ", matrix)

        # pdb.set_trace()

    return matrix.tocsr()


def mdg_refinement(
    mdg: pp.MixedDimensionalGrid,
    mdg_ref: pp.MixedDimensionalGrid,
    tol: float = 1e-8,
    mode: Literal["nested"] = "nested",
) -> None:
    """Wrapper for ``'coarse_fine_cell_mapping'`` in the grid data dictionaries
    to construct mapping for grids in mixed-dimensional grid.

    Adds a data dictionary entry to each grid in mdg.
    The key is ``'coarse_fine_cell_mapping'``.

    Currently, only nested refinement is supported; more general cases are also
    possible.

    ..rubric:: Acknowledgement

        The code was contributed by Haakon Ervik.

    Note:
        No node prop is added to the reference grids in mdg_ref.

    Parameters:
        mdg: Coarse mixed-dimensional grid.
        mdg_ref: Refined mixed-dimensional grid.
        tol: ``default=1e-8``

            Tolerance for point_in_poly* -methods
        mode: ``default='nested'``

            Refinement mode. ``'nested'`` corresponds to refinement by splitting.

    """

    # Check that both mdg's have the same dimension
    dim_mdg = mdg.dim_max()
    dim_mdg_ref = mdg_ref.dim_max()
    assert dim_mdg_ref == dim_mdg

    # Check that both mdg's have the same number of subdomains
    subdomains = mdg.subdomains()
    subdomains_ref = mdg_ref.subdomains()
    assert len(subdomains) == len(subdomains_ref)

    # Check that both mdg's have the same bounding box
    bmin, bmax = pp.domain.mdg_minmax_coordinates(mdg)
    bmin_ref, bmax_ref = pp.domain.mdg_minmax_coordinates(mdg_ref)
    np.testing.assert_almost_equal(bmin, bmin_ref, decimal=8)
    np.testing.assert_almost_equal(bmax, bmax_ref, decimal=8)

    # Strongly check that grids refer to same domain
    for i in np.arange(len(subdomains)):
        sd, sd_ref = subdomains[i], subdomains_ref[i]
        assert sd.id == sd_ref

        # Compute the mapping for this subdomain-pair, and assign the result to the node
        # of the coarse mdg
        if mode == "nested":
            mapping = structured_refinement(sd, sd_ref, point_in_poly_tol=tol)
        else:
            raise NotImplementedError("Unknown refinement mode")

        mdg.subdomain_data(sd)["coarse_fine_cell_mapping"] = mapping


def structured_refinement(
    g: pp.Grid, g_ref: pp.Grid, point_in_poly_tol: float = 1e-8
) -> sps.csc_matrix:
    """Construct a mapping between cells of a grid and its refined version.

    Assuming a regular and a refined mesh, where the refinement is executed by
    splitting.
    I.e. a cell in the refined grid is completely contained within a cell in the
    coarse grid.

    .. rubric:: Acknowledgement

    The code was contributed by Haakon Ervik.

    Parameters:
        g: Coarse grid.
        g_ref: Refined grid.
        point_in_poly_tol: ``default=1e-8``

            Tolerance for :func:`~porepy.geometry_property_checks.point_in_polyhedron`.

    Raises:
        AssertionError: If ``g_ref`` has more cells than ``g`` or if they are of unequal
            dimension.
        AssertionError: Depending on the dimension of ``g``, a certain number of nodes
            must be available.
        AssertionError: If any cell in the finer grid is not contained in exactly one
            cell of the coarser grid.

    Returns:
        Column major sparse matrix mapping from coarse to fine cells.

    """
    # This method creates a mapping from fine to coarse cells by creating a matrix 'M'
    # where the rows represent the fine cells while the columns represent the coarse
    # cells. In practice this means that for an array 'p', of known values on a coarse
    # grid, by applying the mapping
    #     q = M * p
    # each value in a coarse cell will now be transferred to all the fine cells
    # contained within the coarse cell.

    # The procedure for creating this mapping relies on two main assumptions.
    #     1. Each fine cell is fully contained inside exactly one coarse cell.
    #     2. Each cell can be characterised as a simplex.
    #         I.e. Every node except one defines every face of the object.

    # The first assumption implies that the problem of assessing if a fine cell is
    # contained within a coarse cell is reduced to assessing if the center of a fine
    # cell is contained within the coarse cell.

    # The second assumption implies that a cell in any dimension (1D, 2D, 3D) will be
    # a convex object. This way, we can use existing algorithms in PorePy to find if
    # a point is inside a polygon (2D) or polyhedron (3D). (The 1D case is trivial)

    # The general algorithm is as follows (refer to start of for-loop in the code):
    # 1. Make a list of (pointers to) untested cell centers called 'test_cells_ptr'.
    # 2. Iterate through all coarse cells. Now, consider one of these:
    # 3. For all untested cell centers (defined by 'test_cells_ptr'), check if they
    #     are inside the coarse cell.
    # 4. Those that pass (is inside the coarse cell) add to the mapping, then remove
    #     those point from the list of untested cell centers.
    # 5. Assemble the mapping.

    if g.dim == 0:
        mapping = sps.csc_matrix((np.ones(1), ([0], [0])))
        return mapping
    assert g.num_cells < g_ref.num_cells, "Wrong order of input grids"
    assert g.dim == g_ref.dim, "Grids must be of same dimension"

    # 1. Step: Create a list of tuples pointing to the (start, end) index of the nodes
    # of each cell on the coarse grid.
    cell_nodes = g.cell_nodes()
    # start/end row pointers for each column
    nodes_of_cell_ptr = zip(cell_nodes.indptr[:-1], cell_nodes.indptr[1:])

    # 2. Step: Initialize a sps.csc_matrix mapping fine cells to coarse cells.
    indptr = np.array([0])
    indices = np.empty(0)

    cells_ref = g_ref.cell_centers.copy()  # Cell centers in fine grid
    test_cells_ptr = np.arange(g_ref.num_cells)  # Pointer to cell centers
    nodes = g.nodes.copy()

    # 3. Step: If the grids are in 1D or 2D, we can simplify the calculation by rotating
    # the coordinate system to local coordinates. For example, a 2D grid embedded in 3D
    # would be "rotated" so that each coordinate is of the form (x, y, 0).
    if g.dim == 1:
        # Rotate coarse nodes and fine cell centers to align with the x-axis
        tangent = pp.map_geometry.compute_tangent(nodes)
        reference = np.array([1.0, 0.0, 0.0])
        R = pp.map_geometry.project_line_matrix(nodes, tangent, reference=reference)
        nodes = R.dot(nodes)[0, :]
        cells_ref = R.dot(cells_ref)[0, :]

    elif g.dim == 2:  # Pre-processing for efficiency
        # Rotate coarse nodes and fine cell centers to the xy-plane.
        R = pp.map_geometry.project_plane_matrix(nodes, check_planar=False)
        nodes = np.dot(R, nodes)[:2, :]
        cells_ref = np.dot(R, cells_ref)[:2, :]

    # 4. Step: Loop through every coarse cell
    for st, nd in nodes_of_cell_ptr:
        nodes_idx = cell_nodes.indices[st:nd]
        num_nodes = nodes_idx.size

        # 5. Step: for the appropriate grid dimension, test all cell centers not already
        # found to be inside some other coarse cell.
        # 'in_poly' is a boolean array that is True for the points inside and False for
        # the points not inside.
        if g.dim == 1:
            # In 1D, we use a numpy method to check which coarse cell the fine points
            # are inside.
            assert num_nodes == 2, "We assume a 'cell' in 1D is defined by two points."
            line = np.sort(nodes[nodes_idx])
            test_points = cells_ref[test_cells_ptr]
            in_poly = np.searchsorted(line, test_points, side="left") == 1

        elif g.dim == 2:
            assert num_nodes == 3, "We assume simplexes in 2D (i.e. 3 nodes)"
            polygon = nodes[:, nodes_idx]
            test_points = cells_ref[:, test_cells_ptr]
            in_poly = pp.geometry_property_checks.point_in_polygon(
                poly=polygon, p=test_points
            )

        elif g.dim == 3:
            # Make polyhedron from node coordinates.
            # Polyhedron defined as a list of nodes defining its (convex) faces.
            # Assumes simplexes: Every node except one defines every face.
            assert num_nodes == 4, "We assume simplexes in 3D (i.e. 4 nodes)"
            node_coords = nodes[:, nodes_idx]

            ids = np.arange(num_nodes)
            polyhedron: np.ndarray = np.array(
                [node_coords[:, ids != i] for i in np.arange(num_nodes)]
            )
            # Test only points not inside another polyhedron.
            test_points = cells_ref[:, test_cells_ptr]
            in_poly = pp.geometry_property_checks.point_in_polyhedron(
                polyhedron=polyhedron, test_points=test_points, tol=point_in_poly_tol
            )

        else:
            logger.warning("A grid of dimension %d encountered. Skip!", g.dim)
            continue

        # 6. Step: Update pointer to which cell centers to use as test points
        in_poly_ids = test_cells_ptr[in_poly]  # id of cells inside this polyhedron
        # Keep only cells not inside this polyhedron
        test_cells_ptr = test_cells_ptr[~in_poly]

        # Update mapping
        indices = np.append(indices, in_poly_ids)
        indptr = np.append(indptr, indptr[-1] + in_poly_ids.size)

    # 7. Step: assemble the sparse matrix with the mapping.
    data = np.ones(indices.size)
    coarse_fine = sps.csc_matrix((data, indices, indptr))

    assert (
        indices.size == g_ref.num_cells
    ), "Every fine cell should be inside exactly one coarse cell"
    return coarse_fine
