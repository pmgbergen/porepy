#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module contains various functions needed to 
"""
import porepy as pp
import numpy as np
import scipy.sparse as sps

from porepy.utils.matrix_compression import rldecode
from porepy.utils.setmembership import ismember_rows, unique_columns_tol
from porepy.grids.structured import TensorGrid


def match_1d(new_1d, old_1d, tol):
    """ Obtain mappings between the cells of non-matching 1d grids.

    The function constructs an refined 1d grid that consists of all nodes
    of at least one of the input grids.

    It is asumed that the two grids are aligned, with common start and
    endpoints.

    Implementation note: It should be possible to avoid old_1d, by extracting
    points from a 2D grid that lie along the line defined by g_1d.
    However, if g_2d is split along a fracture, the nodes will be
    duplicated. We should then return two grids, probably based on the
    coordinates of the cell centers. sounds cool.

    Parameters:
         new_1d (grid): First grid to be matched
         old_1d (grid): Second grid to be matched.
         tol (double): Tolerance used to filter away false overlaps caused by
             numerical errors. Should be scaled relative to the cell size.

    Returns:
         np.array: Ratio of cell volume in the common grid and the original grid.
         np.array: Mapping between cell numbers in common and first input
              grid.
         np.array: Mapping between cell numbers in common and second input
              grid.

    """
    # Cell-node relation between grids - we know there are two nodes per cell
    cell_nodes1 = new_1d.cell_nodes()
    cell_nodes2 = old_1d.cell_nodes()
    nodes1 = pp.utils.mcolon.mcolon(cell_nodes1.indptr[0:-1], cell_nodes1.indptr[1:])
    nodes2 = pp.utils.mcolon.mcolon(cell_nodes2.indptr[0:-1], cell_nodes2.indptr[1:])

    # Reshape so that the nodes of cells are stored columnwise
    lines1 = cell_nodes1.indices[nodes1].reshape((2, -1), order="F")
    lines2 = cell_nodes2.indices[nodes2].reshape((2, -1), order="F")

    p1 = new_1d.nodes
    p2 = old_1d.nodes

    # Compute the intersection between the two tessalations.
    # intersect is a list, every list member is a tuple with overlapping
    # cells in grid 1 and 2, and their common area.
    intersect = pp.intersections.line_tesselation(p1, p2, lines1, lines2)

    num = len(intersect)
    new_g_ind = np.zeros(num, dtype=np.int)
    old_g_ind = np.zeros(num, dtype=np.int)
    weights = np.zeros(num)

    for ind, i in enumerate(intersect):
        new_g_ind[ind] = i[0]
        old_g_ind[ind] = i[1]
        weights[ind] = i[2]
    weights /= old_1d.cell_volumes[old_g_ind]

    # Remove zero weight intersections
    mask = weights > tol
    new_g_ind = new_g_ind[mask]
    old_g_ind = old_g_ind[mask]
    weights = weights[mask]

    return weights, new_g_ind, old_g_ind


def match_2d(new_g, old_g, tol):
    """ Match two simplex tessalations to identify overlapping cells.

    The overlaps are identified by the cell index of the two overlapping cells,
    and their weighted common area.

    Parameters:
        new_g: simplex grid of dimension 2.
        old_g: simplex grid of dimension 2.

    Returns:
        np.array: Ratio of cell volume in the common grid and the original grid.
        np.array: Index of overlapping cell in the first grid.
        np.array: Index of overlapping cell in the second grid.

    """

    def proj_pts(p, cc, normal):
        """ Project points to the 2d plane defined by normal and center them around cc"""
        rot = pp.map_geometry.project_plane_matrix(p - cc, normal)
        return rot.dot(p - cc)[:2]

    shape = (new_g.dim + 1, new_g.num_cells)
    cn_new_g = new_g.cell_nodes().indices.reshape(shape, order="F")

    shape = (old_g.dim + 1, old_g.num_cells)
    cn_old_g = old_g.cell_nodes().indices.reshape(shape, order="F")

    # Center points around mean
    cc = np.mean(new_g.nodes, axis=1).reshape((3, 1))
    # Calculate common normal for both grids
    n = pp.map_geometry.compute_normal(new_g.nodes - cc)
    n_old = pp.map_geometry.compute_normal(old_g.nodes - cc)
    if not (np.allclose(n, n_old) or np.allclose(n, -n_old)):
        raise ValueError("The new and old grid must lie in the same plane")

    # Calculate intersection
    isect = pp.intersections.triangulations(
        proj_pts(new_g.nodes, cc, n), proj_pts(old_g.nodes, cc, n), cn_new_g, cn_old_g
    )

    num = len(isect)
    new_g_ind = np.zeros(num, dtype=np.int)
    old_g_ind = np.zeros(num, dtype=np.int)
    weights = np.zeros(num)

    for ind, i in enumerate(isect):
        new_g_ind[ind] = i[0]
        old_g_ind[ind] = i[1]
        weights[ind] = i[2]

    weights /= old_g.cell_volumes[old_g_ind]
    return weights, new_g_ind, old_g_ind


def _match_grids_along_line_from_geometry(mg, g_new, g_old, tol):

    # The purpose of this function is to construct a mapping between faces in
    # the old and new grid. Specifically, we need to match faces that lies on
    # the 1d segment identified by the mortar grid, and get the right area
    # weightings when the two grids do not conform.
    #
    # The algorithm is technical, partly because we also need to differ between
    # the left and right side of the segment, as these will belong to different
    # mortar grids.
    #
    # The main steps are:
    #   1) Identify faces in the old grid along the segment via the existing
    #      mapping between mortar grid and higher dimensional grid. Use this
    #      to define the geometry of the segment.
    #   2) Define positive and negative side of the segment, and split cells
    #      and faces along the segement according to this criterion.
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
        # Match faces in a 2d grid and cells in a 1d grid by identifying
        # face-nodes and cell-node relations.
        # loc_faces are faces in 2d grid that are known to coincide with
        # cells.
        # loc_nodes are indices of 2d nodes along the segment, sorted so that
        # the ordering coincides with nodes in 1d grid

        # face-node relation in higher dimensional grid
        fn = g_2d.face_nodes.indices.reshape((g_2d.dim, g_2d.num_faces), order="F")
        # Reduce to faces along segment
        fn_loc = fn[:, loc_faces]
        # Mapping from global (2d) indices to the local indices used in 1d
        # grid. This also account for a sorting of the nodes, so that the
        # nodes.
        ind_map = np.zeros(g_2d.num_faces)
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
    _, faces_on_boundary_old, _ = sps.find(mg._master_to_mortar_int)
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
    #    assert bound_cells_old.size > 1, 'Have not implemented this. Not difficult though'
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

    # Then virtual 1d grid for the new grid. This is a bit more involved,
    # since we need to identify the nodes by their coordinates.
    # This part will be prone to rounding errors, in particular for
    # bad cell shapes.
    nodes_new = g_new.nodes

    # Represent the 1d line by its start and end point, as pulled
    # from the old 1d grid (known coordinates)
    # Find distance from the
    dist, _ = pp.distances.points_segments(nodes_new, start, end)
    # Look for points in the new grid with a small distance to the
    # line
    hit = np.argwhere(dist.ravel() < tol).reshape((1, -1))[0]

    # Depending on geometric tolerance and grid quality, hit
    # may contain nodes that are close to the 1d line, but not on it
    # To improve the results, also require that the faces are boundary faces

    # We know we are in 2d, thus all faces have two nodes
    # We can do the same trick in 3d, provided we have simplex grids
    # but this will fail on Cartesian or polyhedral grids
    fn = g_new.face_nodes.indices.reshape((2, g_new.num_faces), order="F")
    fn_in_hit = np.isin(fn, hit)
    # Faces where all points are found in hit
    faces_by_hit = np.where(np.all(fn_in_hit, axis=0))[0]
    faces_on_boundary_new = np.where(g_new.tags["fracture_faces"].ravel())[0]
    # Only consider faces both in hit, and that are boundary
    faces_on_boundary_new = np.intersect1d(faces_by_hit, faces_on_boundary_new)

    # Cells along the segment, from the new grid
    bound_cells_new = cells_from_faces(g_new, faces_on_boundary_new)
    #    assert bound_cells_new.size > 1, 'Have not implemented this. Not difficult though'
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

        if sn.size == 0 or so.size == 0:
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

        # Map from faces along segment in old to new grid. Consists of three
        # stages: faces in old to cells in 1d version of old, between 1d cells
        # in old and new, cells in new to faces in new

        # From faces to cells in old grid
        rows = face_to_cell_map(
            g_old, g_aux_old, loc_faces_old, loc_nodes_old[sort_ind_old]
        )
        cols = np.arange(rows.size)
        face_to_cell_old = sps.coo_matrix((np.ones(rows.size), (rows, cols)))

        # Mapping between cells in 1d grid
        weights, new_cells, old_cells = match_1d(g_aux_new, g_aux_old, tol)
        between_cells = sps.csr_matrix((weights, (old_cells, new_cells)))

        # From faces to cell in new grid
        rows = face_to_cell_map(
            g_new, g_aux_new, loc_faces_new, loc_nodes_new[sort_ind_new]
        )
        cols = np.arange(rows.size)
        cell_to_face_new = sps.coo_matrix((np.ones(rows.size), (rows, cols)))

        face_map_segment = face_to_cell_old * between_cells * cell_to_face_new

        face_map = face_map_old.T * face_map_segment * face_map_new

        matrix += face_map

    return matrix.tocsr()

