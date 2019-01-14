"""
Various utility functions related to computational geometry.

Some functions (add_point, split_edges, ...?) are mainly aimed at finding
intersection between lines, with grid generation in mind, and should perhaps
be moved to a separate module.

"""
from __future__ import division
import logging
import time
import numpy as np
from sympy import geometry as geom
import networkx as nx
from scipy.spatial import Delaunay

import shapely.geometry as shapely_geometry
import shapely.speedups as shapely_speedups

from porepy.utils import setmembership
import porepy as pp

# Module level logger
logger = logging.getLogger(__name__)

try:
    shapely_speedups.enable()
except AttributeError:
    pass

# -----------------------------------------------------------------------------
#
# START OF FUNCTIONS RELATED TO SPLITTING OF INTERSECTING LINES IN 2D
#
# ------------------------------------------------------------------------------#


def snap_to_grid(pts, tol=1e-3, box=None, **kwargs):
    """
    Snap points to an underlying Cartesian grid.
    Used e.g. for avoiding rounding issues when testing for equality between
    points.

    Anisotropy in the rounding rules can be enforced by the parameter box.

    >>> snap_to_grid([[0.2443], [0.501]])
    array([[ 0.244],
           [ 0.501]])

    >>> snap_to_grid([[0.2443], [0.501]], box=[[10], [1]])
    array([[ 0.24 ],
           [ 0.501]])

    >>> snap_to_grid([[0.2443], [0.501]], tol=0.01)
    array([[ 0.24],
           [ 0.5 ]])

    Parameters:
        pts (np.ndarray, nd x n_pts): Points to be rounded.
        box (np.ndarray, nd x 1, optional): Size of the domain, precision will
            be taken relative to the size. Defaults to unit box.
        precision (double, optional): Resolution of the underlying grid.

    Returns:
        np.ndarray, nd x n_pts: Rounded coordinates.

    """

    pts = np.asarray(pts)

    nd = pts.shape[0]

    if box is None:
        box = np.reshape(np.ones(nd), (nd, 1))
    else:
        box = np.asarray(box)

    # Precission in each direction
    delta = box * tol
    pts = np.rint(pts.astype(np.float64) / delta) * delta
    #    logging.debug('Snapped %i points to grid with tolerance %d', pts.shape[1],
    #                 tol)
    return pts


# ------------------------------------------------------------------------------#


def _split_edge(vertices, edges, edge_ind, new_pt, **kwargs):
    """
    Split a line into two by introcuding a new point.
    Function is used e.g. for gridding purposes.

    The input can be a set of points, and lines between them, of which one is
    to be split.

    New lines will be inserted, unless the new points coincide with the
    start or endpoint of the edge (under the given precision).

    The code is intended for 2D, in 3D use with caution.

    Examples:
        >>> p = np.array([[0, 0], [0, 1]])
        >>> edges = np.array([[0], [1]])
        >>> new_pt = np.array([[0], [0.5]])
        >>> v, e, nl, _ = _split_edge(p, edges, 0, new_pt, tol=1e-3)
        >>> e
        array([[0, 2],
               [2, 1]])

    Parameters:
        vertices (np.ndarray, nd x num_pt): Points of the vertex sets.
        edges (np.ndarray, n x num_edges): Connections between lines. If n>2,
            the additional rows are treated as tags, that are preserved under
            splitting.
        edge_ind (int): index of edges to be split.
        new_pt (np.ndarray, nd x n): new points to be inserted. Assumed to be
            on the edge to be split. If more than one point is inserted
            (segment intersection), it is assumed that new_pt[:, 0] is the one
            closest to edges[0, edge_ind] (conforming with the output of
            lines_intersect).
        **kwargs: Arguments passed to snap_to_grid

    Returns:
        np.ndarray, nd x n_pt: new point set, possibly with new point inserted.
        np.ndarray, n x n_con: new edge set, possibly with new lines defined.
        boolean: True if a new line is created, otherwise false.
        int: Splitting type, indicating which splitting strategy was used.
            Intended for debugging.

    """
    logger = logging.getLogger(__name__ + ".split_edge")
    tol = kwargs["tol"]

    # Some back and forth with the index of the edges to be split, depending on
    # whether it is one or two
    edge_ind = np.asarray(edge_ind)
    if edge_ind.size > 1:
        edge_ind_first = edge_ind[0]
    else:
        edge_ind_first = edge_ind

    # Start and end of the first (possibly only) edge
    start = edges[0, edge_ind_first]
    end = edges[1, edge_ind_first]

    # Number of points before edges is modified. Used for sanity check below.
    orig_num_pts = edges[:2].max()
    orig_num_edges = edges.shape[1]

    # Save tags associated with the edge.
    # NOTE: For segment intersetions where the edges have different tags, one
    # of the values will be overridden now. Fix later.
    tags = edges[2:, edge_ind_first]

    # Try to add new points
    vertices, pt_ind, _ = _add_point(vertices, new_pt, **kwargs)

    # Sanity check
    assert len(pt_ind) <= 2, "Splitting can at most create two new points"
    # Check for a single intersection point
    if len(pt_ind) < 2:
        pi = pt_ind[0]
        # Intersection at a point.
        if start == pi or end == pi:
            # Nothing happens, the intersection between the edges coincides
            # with a shared vertex for the edges
            new_line = 0
            split_type = 0
            logger.debug("Intersection on shared vertex")
            return vertices, edges, new_line, split_type
        else:
            # We may need to split the edge (start, end) into two
            new_edges = np.array([[start, pi], [pi, end]])
            # ... however, the new edges may already exist in the set (this
            # apparently can happen for complex networks with several fractures
            # sharing a line).
            # Check if the new candidate edges already are defined in the set
            # of edges
            ismem, _ = setmembership.ismember_rows(new_edges, edges[:2])
            if any(ismem):
                new_edges = np.delete(new_edges, np.squeeze(np.argwhere(ismem)), axis=0)
            if new_edges.shape[0] == 1:
                new_edges = new_edges.reshape((-1, 1))

        if new_edges.size == 0:
            new_line = 0
            split_type = 1
            logger.debug("Intersection on existing vertex")
            return vertices, edges, new_line, split_type

        # Add any tags to the new edge.
        if tags.size > 0:
            new_edges = np.vstack(
                (new_edges, np.tile(tags[:, np.newaxis], new_edges.shape[1]))
            )
        # Insert the new edge in the midle of the set of edges.
        edges = np.hstack(
            (edges[:, :edge_ind_first], new_edges, edges[:, edge_ind_first + 1 :])
        )
        # We have added as many new edges as there are columns in new_edges,
        # minus 1 (which was removed / ignored).
        new_line = new_edges.shape[1] - 1

        # Sanity check of new edge
        if np.any(np.diff(edges[:2], axis=0) == 0):
            raise ValueError("Have created a point edge")
        edge_copy = np.sort(edges[:2], axis=0)
        edge_unique, _, _ = setmembership.unique_columns_tol(edge_copy, tol=tol)
        if edge_unique.shape[1] < edges.shape[1]:
            raise ValueError("Have created the same edge twice")

        split_type = 2
        logger.debug("Intersection on new single vertex")
        return vertices, edges, new_line, split_type
    else:
        logger.debug("Splitting handles two points")
        # Without this, we will delete something we should not delete below.
        assert edge_ind[0] < edge_ind[1]

        # Intersection along a segment.
        # No new points should have been made
        assert pt_ind[0] <= orig_num_pts and pt_ind[1] <= orig_num_pts

        pt_ind = np.reshape(np.array(pt_ind), (-1, 1))

        # There are three (four) possible configurations
        # a) The intersection is contained within (start, end). edge_ind[0]
        # should be split into three segments, and edge_ind[1] should be
        # deleted (it will be identical to the middle of the new segments).
        # b) The intersection is identical with (start, end). edge_ind[1]
        # should be split into three segments, and edge_ind[0] is deleted.
        # c) and d) The intersection consists of one of (start, end), and another
        # point. Both edge_ind[0] and edge_ind[1] should be split into two
        # segments.

        i0 = pt_ind[0]
        i1 = pt_ind[1]
        if i0 != start and i1 != end:
            # Delete the second segment
            edges = np.delete(edges, edge_ind[1], axis=1)
            if edges.shape[0] == 1:
                edges = edges.reshape((-1, 1))
            # We know that i0 will be closest to start, thus (start, i0) is a
            # pair.
            # New segments (i0, i1) is identical to the old edge_ind[1]
            new_edges = np.array([[start, i0, i1], [i0, i1, end]])
            if tags.size > 0:
                new_edges = np.vstack(
                    (new_edges, np.tile(tags[:, np.newaxis], new_edges.shape[1]))
                )
            # Combine everything.
            edges = np.hstack(
                (edges[:, : edge_ind[0]], new_edges, edges[:, edge_ind[0] + 1 :])
            )

            logger.debug("Second edge split into two new parts")
            split_type = 4
        elif i0 == start and i1 == end:
            # We don't know if i0 is closest to the start or end of edges[:,
            # edges_ind[1]]. Find the nearest.
            if dist_point_pointset(
                vertices[:, i0], vertices[:, edges[0, edge_ind[1]]]
            ) < dist_point_pointset(
                vertices[:, i1], vertices[:, edges[0, edge_ind[1]]]
            ):
                other_start = edges[0, edge_ind[1]]
                other_end = edges[1, edge_ind[1]]
            else:
                other_start = edges[1, edge_ind[1]]
                other_end = edges[0, edge_ind[1]]
            # New segments (i0, i1) is identical to the old edge_ind[0]
            new_edges = np.array([[other_start, i0, i1], [i0, i1, other_end]])
            # For some reason we sometimes create point-edges here (start and
            # end are identical). Delete these if necessary
            del_ind = np.squeeze(np.where(np.diff(new_edges, axis=0)[0] == 0))
            new_edges = np.delete(new_edges, del_ind, axis=1)
            if tags.size > 0:
                new_edges = np.vstack(
                    (new_edges, np.tile(tags[:, np.newaxis], new_edges.shape[1]))
                )
            # Combine everything.
            edges = np.hstack(
                (edges[:, : edge_ind[1]], new_edges, edges[:, (edge_ind[1] + 1) :])
            )
            # Delete the second segment. This is most easily handled after
            # edges is expanded, to avoid accounting for changing edge indices.
            edges = np.delete(edges, edge_ind[0], axis=1)
            logger.debug("First edge split into 2 parts")
            split_type = 5

        # Note that we know that i0 is closest to start, thus no need to test
        # for i1 == start
        elif i0 == start and i1 != end:
            # The intersection starts in start of edge_ind[0], and end before
            # the end of edge_ind[0] (if not we would have i1==end).
            # The intersection should be split into intervals (start, i1), (i1,
            # end) and possibly (edge_ind[1][0 or 1], start); with the latter
            # representing the part of edge_ind[1] laying on the other side of
            # start compared than i1. The latter part will should not be
            # included if start is also a node of edge_ind[1].
            #
            # Examples in 1d (really needed to make this concrete right now):
            #  edge_ind[0] = (0, 2), edge_ind[1] = (-1, 1) is split into
            #   (0, 1), (1, 2) and (-1, 1) (listed in the same order as above).
            #
            # edge_ind[0] = (0, 2), edge_ind[1] = (0, 1) is split into
            #   (0, 1), (1, 2)
            if edges[0, edge_ind[1]] == i1:
                if edges[1, edge_ind[1]] == start:
                    logger.debug("First edge split into 2")
                    edges = np.delete(edges, edge_ind[1], axis=1)
                else:
                    edges[0, edge_ind[1]] = start
                    logger.debug("First and second edge split into 2")
            elif edges[1, edge_ind[1]] == i1:
                if edges[0, edge_ind[1]] == start:
                    edges = np.delete(edges, edge_ind[1], axis=1)
                    logger.debug("First edge split into 2")
                else:
                    edges[1, edge_ind[1]] = start
                    logger.debug("First and second edge split into 2")
            else:
                raise ValueError("This should not happen")

            new_edges = np.array([[start, i1], [i1, end]])
            if tags.size > 0:
                new_edges = np.vstack(
                    (new_edges, np.tile(tags[:, np.newaxis], new_edges.shape[1]))
                )

            edges = np.hstack(
                (edges[:, : edge_ind[0]], new_edges, edges[:, (edge_ind[0] + 1) :])
            )
            split_type = 6

        elif i0 != start and i1 == end:
            # Analogous configuration as the one above, but with i0 replaced by
            # i1 and start by end.
            if edges[0, edge_ind[1]] == i0:
                if edges[1, edge_ind[1]] == end:
                    edges = np.delete(edges, edge_ind[1], axis=1)
                    logger.debug("First edge split into 2")
                else:
                    edges[0, edge_ind[1]] = end
                    logger.debug("First and second edge split into 2")
            elif edges[1, edge_ind[1]] == i0:
                if edges[0, edge_ind[1]] == end:
                    edges = np.delete(edges, edge_ind[1], axis=1)
                    logger.debug("First edge split into 2")
                else:
                    edges[1, edge_ind[1]] = end
                    logger.debug("First and second edge split into 2")
            else:
                raise ValueError("This should not happen")
            new_edges = np.array([[start, i0], [i0, end]])
            if tags.size > 0:
                new_edges = np.vstack(
                    (new_edges, np.tile(tags[:, np.newaxis], new_edges.shape[1]))
                )

            edges = np.hstack(
                (edges[:, : edge_ind[0]], new_edges, edges[:, (edge_ind[0] + 1) :])
            )
            split_type = 7

        else:
            raise ValueError("How did it come to this")

        # Check validity of the new edge configuration
        if np.any(np.diff(edges[:2], axis=0) == 0):
            raise ValueError("Have created a point edge")

        # We may have created an edge that already existed in the grid. Remove
        # this by uniquifying the edges.
        # Hopefully, we do not mess up the edges here.
        edges_copy = np.sort(edges[:2], axis=0)
        edges_unique, new_2_old, _ = setmembership.unique_columns_tol(
            edges_copy, tol=tol
        )
        # Refer to unique edges if necessary
        if edges_unique.shape[1] < edges.shape[1]:
            # Copy tags
            edges = np.vstack((edges_unique, edges[2:, new_2_old]))
            # Also signify that we have carried out this operation.
            split_type = [split_type, 8]

        # Number of new lines created
        new_line = edges.shape[1] - orig_num_edges

        return vertices, edges, new_line, split_type


# ------------------------------------------------------**kwargs------------------------#


def _add_point(vertices, pt, tol=1e-3, snap=True, **kwargs):
    """
    Add a point to a point set, unless the point already exist in the set.

    Point coordinates are compared relative to an underlying Cartesian grid,
    see snap_to_grid for details.

    The function is created with gridding in mind, but may be useful for other
    purposes as well.

    Parameters:
        vertices (np.ndarray, nd x num_pts): existing point set
        pt (np.ndarray, nd x 1): Point to be added
        tol (double): Precision of underlying Cartesian grid
        **kwargs: Arguments passed to snap_to_grid

    Returns:
        np.ndarray, nd x n_pt: New point set, possibly containing a new point
        int: Index of the new point added (if any). If not, index of the
            closest existing point, i.e. the one that made a new point
            unnecessary.
        np.ndarray, nd x 1: The new point, or None if no new point was needed.

    """
    if "tol" not in kwargs:
        kwargs["tol"] = tol

    nd = vertices.shape[0]
    # Before comparing coordinates, snap both existing and new point to the
    # underlying grid
    if snap:
        vertices = snap_to_grid(vertices, **kwargs)
        pt = snap_to_grid(pt, **kwargs)

    new_pt = np.empty((nd, 0))
    ind = []
    # Distance
    for i in range(pt.shape[-1]):
        dist = dist_point_pointset(pt[:, i], vertices)
        min_dist = np.min(dist)

        # The tolerance parameter here turns out to be critical in an edge
        # intersection removal procedure. The scaling factor is somewhat
        # arbitrary, and should be looked into.
        if min_dist < tol * np.sqrt(3):
            # No new point is needed
            ind.append(np.argmin(dist))
        else:
            # Append the new point at the end of the point list
            ind.append(vertices.shape[1])
            vertices = np.append(vertices, pt, axis=1)
            new_pt = np.hstack((new_pt, pt[:, i].reshape((-1, 1))))
    if new_pt.shape[1] == 0:
        new_pt = None
    return vertices, ind, new_pt


# -----------------------------------------------------------------------------#


def remove_edge_crossings(vertices, edges, tol=1e-3, verbose=0, snap=True, **kwargs):
    """
    Process a set of points and connections between them so that the result
    is an extended point set and new connections that do not intersect.

    The function is written for gridding of fractured domains, but may be
    of use in other cases as well. The geometry is assumed to be 2D, (the
    corresponding operation in 3D requires intersection between planes, and
    is a rather complex, although doable, task).

    The connections are defined by their start and endpoints, and can also
    have tags assigned. If so, the tags are preserved as connections are split.

    Parameters:
        vertices (np.ndarray, 2 x n_pt): Coordinates of points to be processed
        edges (np.ndarray, n x n_con): Connections between lines. n >= 2, row
            0 and 1 are index of start and endpoints, additional rows are tags
        tol (double, optional, default=1e-8): Tolerance used for comparing
            equal points.
        **kwargs: Arguments passed to snap_to_grid.

    Returns:
    np.ndarray, (2 x n_pt), array of points, possibly expanded.
    np.ndarray, (n x n_edges), array of new edges. Non-intersecting.

    Raises:
    NotImplementedError if a 3D point array is provided.

    """
    # Sanity check of input specification edge endpoints
    assert np.all(np.diff(edges[:2], axis=0) != 0), (
        "Found point edge before" "removal of intersections"
    )

    # Use a non-standard naming convention for the logger to
    logger = logging.getLogger(__name__ + ".remove_edge_crossings")

    logger.debug("Find intersections between %i edges", edges.shape[1])
    nd = vertices.shape[0]

    # Only 2D is considered. 3D should not be too dificult, but it is not
    # clear how relevant it is
    if nd != 2:
        raise NotImplementedError("Only 2D so far")

    edge_counter = 0

    # Add tolerance to kwargs, this is later passed to split_edges, and further
    # on.
    kwargs["tol"] = tol
    kwargs["snap"] = snap
    if snap:
        vertices = snap_to_grid(vertices, **kwargs)

    # Field used for debugging of edge splits. To see the meaning of the values
    # of each split, look in the source code of split_edges.
    split_type = []

    # Loop over all edges, search for intersections. The number of edges can
    # change due to splitting.
    while edge_counter < edges.shape[1]:
        # The direct test of whether two edges intersect is somewhat
        # expensive, and it is hard to vectorize this part. Therefore,
        # we first identify edges which crosses the extention of this edge (
        # intersection of line and line segment). We then go on to test for
        # real intersections.
        logger.debug(
            "Remove intersection from edge with indices %i, %i",
            edges[0, edge_counter],
            edges[1, edge_counter],
        )
        # Find start and stop coordinates for all edges
        start_x = vertices[0, edges[0]]
        start_y = vertices[1, edges[0]]
        end_x = vertices[0, edges[1]]
        end_y = vertices[1, edges[1]]
        logger.debug(
            "Start point (%.5f, %.5f), End (%.5f, %.5f)",
            start_x[edge_counter],
            start_y[edge_counter],
            end_x[edge_counter],
            end_y[edge_counter],
        )

        a = end_y - start_y
        b = -(end_x - start_x)

        # Midpoint of this edge
        xm = (start_x[edge_counter] + end_x[edge_counter]) / 2.0
        ym = (start_y[edge_counter] + end_y[edge_counter]) / 2.0

        # For all lines, find which side of line i it's two endpoints are.
        # If c1 and c2 have different signs, they will be on different sides
        # of line i. See
        #
        # http://stackoverflow.com/questions/385305/efficient-maths-algorithm-to-calculate-intersections
        #
        # answer by PolyThinker and comments by Jason S, for more information.
        c1 = a[edge_counter] * (start_x - xm) + b[edge_counter] * (start_y - ym)
        c2 = a[edge_counter] * (end_x - xm) + b[edge_counter] * (end_y - ym)

        tol_scaled = tol * max(1, np.max([np.sqrt(np.abs(c1)), np.sqrt(np.abs(c2))]))

        # We check for three cases
        # 1) Lines crossing
        lines_cross = np.sign(c1) != np.sign(c2)
        # 2) Lines parallel
        parallel_lines = np.logical_and(
            np.abs(c1) < tol_scaled, np.abs(c2) < tol_scaled
        )
        # 3) One line look to end on the other
        lines_almost_cross = np.logical_or(
            np.abs(c1) < tol_scaled, np.abs(c2) < tol_scaled
        )
        # Any of the three above deserves a closer look
        line_intersections = np.logical_or(
            np.logical_or(parallel_lines, lines_cross), lines_almost_cross
        )

        # Find elements which may intersect.
        intersections = np.argwhere(line_intersections)
        # np.argwhere returns an array of dimensions (1, dim), so we reduce
        # this to truly 1D, or simply continue with the next edge if there
        # are no candidate edges
        if intersections.size > 0:
            intersections = intersections.ravel("C")
            logger.debug("Found %i candidate intersections", intersections.size)
        else:
            # There are no candidates for intersection
            edge_counter += 1
            logger.debug("Found no candidate intersections")
            continue

        size_before_splitting = edges.shape[1]

        int_counter = 0
        while intersections.size > 0 and int_counter < intersections.size:
            # Line intersect (inner loop) is an intersection if it crosses
            # the extension of line edge_counter (outer loop) (ie intsect it
            #  crosses the infinite line that goes through the endpoints of
            # edge_counter), but we don't know if it actually crosses the
            # line segment edge_counter. Now we do a more refined search to
            # find if the line segments intersects. Note that there is no
            # help in vectorizing lines_intersect and computing intersection
            #  points for all lines in intersections, since line i may be
            # split, and the intersection points must recalculated. It may
            # be possible to reorganize this while-loop by computing all
            # intersection points(vectorized), and only recompuing if line
            # edge_counter is split, but we keep things simple for now.
            intsect = intersections[int_counter]
            if intsect <= edge_counter:
                int_counter += 1
                continue

            logger.debug("Look for intersection with edge %i", intsect)
            logger.debug(
                "Outer edge: Start (%.5f, %.5f), End (%.5f, %.5f)",
                vertices[0, edges[0, edge_counter]],
                vertices[1, edges[0, edge_counter]],
                vertices[0, edges[1, edge_counter]],
                vertices[1, edges[1, edge_counter]],
            )
            logger.debug(
                "Inner edge: Start (%.5f, %.5f), End (%.5f, %.5f)",
                vertices[0, edges[0, intsect]],
                vertices[1, edges[0, intsect]],
                vertices[0, edges[1, intsect]],
                vertices[1, edges[1, intsect]],
            )

            # Check if this point intersects
            new_pt = lines_intersect(
                vertices[:, edges[0, edge_counter]],
                vertices[:, edges[1, edge_counter]],
                vertices[:, edges[0, intsect]],
                vertices[:, edges[1, intsect]],
                tol=tol,
            )

            def __min_dist(p):
                md = np.inf
                for pi in [
                    edges[0, edge_counter],
                    edges[1, edge_counter],
                    edges[0, intsect],
                    edges[1, intsect],
                ]:
                    md = min(md, dist_point_pointset(p, vertices[:, pi]))
                return md

            orig_vertex_num = vertices.shape[1]
            orig_edge_num = edges.shape[1]

            if new_pt is None:
                logger.debug("No intersection found")
            else:
                if snap:
                    new_pt = snap_to_grid(new_pt, tol=tol)
                # The case of segment intersections need special treatment.
                if new_pt.shape[-1] == 1:
                    logger.debug(
                        "Found intersection (%.5f, %.5f)", new_pt[0], new_pt[1]
                    )

                    # Split edge edge_counter (outer loop), unless the
                    # intersection hits an existing point (in practices this
                    # means the intersection runs through an endpoint of the
                    # edge in an L-type configuration, in which case no new point
                    # is needed)
                    md = __min_dist(new_pt)
                    vertices, edges, split_outer_edge, split = _split_edge(
                        vertices, edges, edge_counter, new_pt, **kwargs
                    )
                    split_type.append(split)
                    if split_outer_edge > 0:
                        logger.debug("Split outer edge")

                    if edges.shape[1] > orig_edge_num + split_outer_edge:
                        raise ValueError("Have created edge without bookkeeping")
                    # If the outer edge (represented by edge_counter) was split,
                    # e.g. inserted into the list of edges we need to increase the
                    # index of the inner edge
                    intsect += split_outer_edge

                    # Possibly split the inner edge
                    vertices, edges, split_inner_edge, split = _split_edge(
                        vertices, edges, intsect, new_pt, **kwargs
                    )
                    if (
                        edges.shape[1]
                        > orig_edge_num + split_inner_edge + split_outer_edge
                    ):
                        raise ValueError("Have created edge without bookkeeping")

                    split_type.append(split)
                    if split_inner_edge > 0:
                        logger.debug("Split inner edge")
                    intersections += split_outer_edge + split_inner_edge
                else:
                    # We have found an intersection along a line segment
                    logger.debug(
                        """Found two intersections: (%.5f, %.5f) and
                                    (%.5f, %.5f)""",
                        new_pt[0, 0],
                        new_pt[1, 0],
                        new_pt[0, 1],
                        new_pt[1, 1],
                    )
                    vertices, edges, splits, s_type = _split_edge(
                        vertices, edges, [edge_counter, intsect], new_pt, **kwargs
                    )
                    split_type.append(s_type)
                    intersections += splits
                    logger.debug("Split into %i parts", splits)

            # Sanity checks - turned out to be useful for debugging.
            if np.any(np.diff(edges[:2], axis=0) == 0):
                raise ValueError("Have somehow created a point edge")
            if intersections.max() > edges.shape[1]:
                raise ValueError("Intersection pointer outside edge array")

            # We're done with this candidate edge. Increase index of inner loop
            int_counter += 1

        # We're done with all intersections of this loop. increase index of
        # outer loop
        edge_counter += 1
        logger.debug(
            "Edge split into %i new parts", edges.shape[1] - size_before_splitting
        )

    return vertices, edges


def _axis_aligned_bounding_box_2d(p, e):
    """ For a set of lines in 2d, obtain the bounding box for each line.

    The lines are specified as a list of points, together with connections between
    the points.

    Parameters:
        p (np.ndarray, 2 x n_pt): Coordinates of points to be processed
        e (np.ndarray, n x n_con): Connections between lines. n >= 2, row
            0 and 1 are index of start and endpoints, additional rows are tags

    Returns:
        np.array (n_pt): Minimum x-coordinate for all lines.
        np.array (n_pt): Maximum x-coordinate for all lines.
        np.array (n_pt): Minimum y-coordinate for all lines.
        np.array (n_pt): Maximum y-coordinate for all lines.

    """
    x = p[0]
    y = p[1]

    x_0 = x[e[0]]
    x_1 = x[e[1]]
    y_0 = y[e[0]]
    y_1 = y[e[1]]

    x_min = np.minimum(x_0, x_1)
    x_max = np.maximum(x_0, x_1)
    y_min = np.minimum(y_0, y_1)
    y_max = np.maximum(y_0, y_1)

    return x_min, x_max, y_min, y_max


def _axis_aligned_bounding_box_3d(polys):
    """ For a set of polygons embedded in 3d, obtain the bounding box for each object.

    The polygons are specified as a list of numpy arrays.

    Parameters:
        p (list of np.ndarray, 3 x n_pt): Each list element specifies a
            polygon, described by its vertexes in a 3 x num_points np.array.

    Returns:
        np.array (n_pt): Minimum x-coordinate for all lines.
        np.array (n_pt): Maximum x-coordinate for all lines.
        np.array (n_pt): Minimum y-coordinate for all lines.
        np.array (n_pt): Maximum y-coordinate for all lines.
        np.array (n_pt): Minimum z-coordinate for all lines.
        np.array (n_pt): Maximum z-coordinate for all lines.

    """

    polys = list(polys)

    num_poly = len(polys)

    x_min = np.empty(num_poly)
    x_max = np.empty_like(x_min)
    y_min = np.empty_like(x_min)
    y_max = np.empty_like(x_min)
    z_min = np.empty_like(x_min)
    z_max = np.empty_like(x_min)

    for ind, p in enumerate(polys):
        x_min[ind] = np.min(p[0])
        x_max[ind] = np.max(p[0])
        y_min[ind] = np.min(p[1])
        y_max[ind] = np.max(p[1])
        z_min[ind] = np.min(p[2])
        z_max[ind] = np.max(p[2])

    return x_min, x_max, y_min, y_max, z_min, z_max


def _identify_overlapping_intervals(left, right):
    """ Based on a set of start and end coordinates for intervals, identify pairs of
    overlapping intervals.

    Parameters:
        left (np.array): Minimum coordinates of the intervals.
        right (np.array): Maximum coordinates of the intervals.

        For all items, left <= right (but equality is allowed).

    Returns:
        np.array, 2 x num_overlaps: Each column contains a pair of overlapping
            intervals, refering to their placement in left and right. The pairs
            are sorted so that the lowest index is in the first column.

    """
    # There can be no overlaps if there is less than two intervals
    if left.size < 2:
        return np.empty((2, 0))

    # Sort the coordinates
    sort_ind_left = np.argsort(left)
    sort_ind_right = np.argsort(right)

    # pointers to the next start and end point of an interval
    next_right = 0
    next_left = 0

    # List of pairs we have found
    pairs = []
    # List of intervals we are currently in. All intervals will join and leave this set.
    active = []

    num_lines = left.size

    # Loop through the line, add and remove intervals as we come across them.
    while True:
        # Check if the next start (left) point is before the next endpoint,
        # But only if there are more left points available.
        # Less or equal is critical here, or else cases where a point interval
        # is combined with the start of another interval may not be discovered.
        if (
            next_left < num_lines
            and left[sort_ind_left[next_left]] <= right[sort_ind_right[next_right]]
        ):
            # We have started a new interval. This will be paired with
            # all active intervals
            for a in active:
                pairs.append([a, sort_ind_left[next_left]])
            # Also join the new intervals to the active set.
            active.append(sort_ind_left[next_left])
            # Increase the index
            next_left += 1
        else:
            # We have reached the end of the interval - remove it from the
            # active ones.
            active.remove(sort_ind_right[next_right])
            next_right += 1
            # Check if we have come to the end
            if next_right == num_lines:
                break

    if len(pairs) == 0:
        return np.empty((2, 0))
    else:
        pairs = np.asarray(pairs).T
        # First sort the pairs themselves
        pairs.sort(axis=0)
        # Next, sort the columns so that the first row is non-decreasing
        sort_ind = np.argsort(pairs[0])
        pairs = pairs[:, sort_ind]
        return pairs


def _identify_overlapping_rectangles(xmin, xmax, ymin, ymax, tol=1e-8):
    """ Based on a set of start and end coordinates for bounding boxes, identify pairs of
    overlapping rectangles.

    The algorithm was found in 'A fast method for fracture intersection detection
    in discrete fracture networks' by Dong et al, omputers and Geotechniques 2018.

    Parameters:
        xmin (np.array): Minimum coordinates of the rectangle on the first axis.
        xmax (np.array): Maximum coordinates of the rectangle on the first axis.
        ymin (np.array): Minimum coordinates of the rectangle on the second axis.
        ymax (np.array): Maximum coordinates of the rectangle on the second axis.

        For all items, xmin <= xmax (but equality is allowed), correspondingly for
        the y-coordinates

    Returns:
        np.array, 2 x num_overlaps: Each column contains a pair of overlapping
            intervals, refering to their placement in left and right. The pairs
            are sorted so that the lowest index is in the first column.

    """
    # There can be no overlaps if there is less than two rectangles
    if xmin.size < 2:
        return np.empty((2, 0))

    # Sort the coordinates
    sort_ind_min = np.argsort(xmin)
    sort_ind_max = np.argsort(xmax)

    # pointers to the next start and end point of an interval
    next_min = 0
    next_max = 0

    # List of pairs we have found
    pairs = []
    # List of intervals we are currently in. All intervals will join and leave this set.
    active = []

    num_lines = xmax.size

    # Pass along the x-axis, identify the start and end of rectangles as we go.
    # The idea is then for each new interval to check which of the active intervals
    # also have overlap along the y-axis. These will be identified as pairs.
    while True:
        # Check if the next start (xmin) point is before the next endpoint,
        # but only if there are more left points available.
        # Less or equal is critical here, or else cases where a point interval
        # is combined with the start of another interval may not be discovered.
        if (
            next_min < num_lines
            and xmin[sort_ind_min[next_min]] <= xmax[sort_ind_max[next_max]]
        ):
            # Find active rectangles where the y-interval is also overlapping
            between = np.where(
                np.logical_and(
                    ymax[sort_ind_min[next_min]] >= ymin[active],
                    ymin[sort_ind_min[next_min]] <= ymax[active],
                )
            )[0]
            # For all identified overlaps, add the new pairs
            for a in between:
                pairs.append([active[a], sort_ind_min[next_min]])
            # Add this to the active rectangles, and increase the index
            active.append(sort_ind_min[next_min])
            next_min += 1
        else:
            # We are leaving a rectangle.
            active.remove(sort_ind_max[next_max])
            next_max += 1
            # Check if we have come to the end
            if next_max == num_lines:
                break

    if len(pairs) == 0:
        return np.empty((2, 0))
    else:
        pairs = np.asarray(pairs).T
        # First sort the pairs themselves
        pairs.sort(axis=0)
        # Next, sort the columns so that the first row is non-decreasing
        sort_ind = np.argsort(pairs[0])
        pairs = pairs[:, sort_ind]
        return pairs


def _intersect_pairs(p1, p2):
    """ For two lists containing pair of indices, find the intersection.

    Parameters:
        p1 (np.array, 2 x n): Each column contains a pair of indices.
        p2 (np.array, 2 x m): Each column contains a pair of indices.

    Returns:
        np.array, (2 x k, k <= min(n, m)): Each column contains a pair of
            indices that are found in both p1 and p2. The array is sorted so
            that items in the first row is less or equal to the second row.
            The columns are sorted according to the numbers in the first row.

    """
    # Special treatment of empty lists
    if p1.size == 0 or p2.size == 0:
        return np.empty((2, 0))
    else:
        # Do the intersection
        _, ind = pp.utils.setmembership.ismember_rows(p1, p2)
        pairs = p2[:, ind]

        # First sort the pairs themselves
        pairs.sort(axis=0)
        # Next, sort the columns so that the first row is non-decreasing
        sort_ind = np.argsort(pairs[0])
        pairs = pairs[:, sort_ind]
        return pairs


def remove_edge_crossings2(p, e, tol=1e-4):
    """ Process a set of points and connections between them so that the result
    is an extended point set and new connections that do not intersect.

    The function is written for gridding of fractured domains, but may be
    of use in other cases as well. The geometry is assumed to be 2D.

    The connections are defined by their start and endpoints, and can also
    have tags assigned. If so, the tags are preserved as connections are split.

    IMPLEMENTATION NOTE: This is a re-implementation of the old function
    remove_edge_crossings, based on a much faster algorithm. The two functions
    will coexist for a while.

    Parameters:
        p (np.ndarray, 2 x n_pt): Coordinates of points to be processed
        e (np.ndarray, n x n_con): Connections between lines. n >= 2, row
            0 and 1 are index of start and endpoints, additional rows are tags
        tol (double, optional, default=1e-8): Tolerance used for comparing
            equal points.

    Returns:
        np.ndarray, (2 x n_pt), array of points, possibly expanded.
        np.ndarray, (n x n_edges), array of new edges. Non-intersecting.

    """
    # Find the bounding box
    x_min, x_max, y_min, y_max = _axis_aligned_bounding_box_2d(p, e)
    # Identify fractures with overlapping bounding boxes
    pairs = _identify_overlapping_rectangles(x_min, x_max, y_min, y_max)

    # Identify all fractures that are the first (by index) of a potentially
    # crossing pair. A better way to group the fractures may be feasible,
    # but this has not been investigated.
    start_inds = np.unique(pairs[0])

    num_lines = e.shape[1]

    # Data structure for storage of intersection points. For each fracture,
    # we have an array that will contain the index of the intersections.
    isect_pt = np.empty(num_lines, dtype=np.object)
    for i in range(isect_pt.size):
        isect_pt[i] = np.empty(0, dtype=np.int)

    # Array of new points, found in the intersection of old ones.
    new_pts = []
    # The new points will be appended to the old ones, thus their index
    # must be adjusted.
    new_ind = p.shape[1]

    # Loop through all candidate pairs of intersecting fractures, check if
    # they do intersect. If so, store the point, and for each crossing fracture
    # take note of the index of the cross point.
    for di, line_ind in enumerate(start_inds):
        # First fracture in the candidate pair
        main = line_ind
        # Find all other fractures that is in a pair with the main as the first one.
        hit = np.where(pairs[0] == main)
        # Sort the other points; this makes debugging simpler if nothing else.
        other = np.sort(pairs[1, hit][0])

        # We will first do a coarse sorting, to rule out fractures that are clearly
        # not intersecting, and then do a finer search for an intersection below.

        # Utility function to pull out one or several points from an array based
        # on index
        def pt(p, ind):
            a = p[:, ind]
            if ind.size == 1:
                return a.reshape((-1, 1))
            else:
                return a

        # Obtain start and endpoint of the main and other fractures
        start_main = pt(p, e[0, main])
        end_main = pt(p, e[1, main])
        start_other = pt(p, e[0, other])
        end_other = pt(p, e[1, other])

        # Utility function to normalize the fracture length
        def normalize(v):
            nrm = np.sqrt(np.sum(v ** 2, axis=0))

            # If the norm of the vector is essentially zero, do not normalize the vector
            hit = nrm < tol
            nrm[hit] = 1
            return v / nrm

        def dist(a, b):
            return np.sqrt(np.sum((a - b) ** 2))

        # Vectors along the main fracture, and from the start of the main
        # to the start and end of the other fractures. All normalized.
        # If the other edges share start or endpoint with the main one, normalization
        # of the distance vector will make the vector nans. In this case, we
        # use another point along the other line, this works equally well for the
        # coarse identification (based on cross products).
        # If the segments are overlapping, there will still be issues with nans,
        # but these are dealt with below.
        main_vec = normalize(end_main - start_main)
        if dist(start_other, start_main) > 1e-4:
            main_other_start = normalize(start_other - start_main)
        else:
            main_other_start = normalize(0.5 * (start_other + end_other) - start_main)
        if dist(end_other, start_main) > 1e-4:
            main_other_end = normalize(end_other - start_main)
        else:
            main_other_end = normalize(0.3 * start_other + 0.7 * end_other - start_main)

        # Modified signum function: The value is 0 if it is very close to zero.
        def mod_sign(v, tol):
            sgn = np.sign(v)
            sgn[np.abs(v) < tol] = 0
            return sgn

        # Take the cross product between the vector along the main line, and the
        # vectors to the start and end of the other lines, respectively.
        start_cross = mod_sign(
            main_vec[0] * main_other_start[1] - main_vec[1] * main_other_start[0], tol
        )
        end_cross = mod_sign(
            main_vec[0] * main_other_end[1] - main_vec[1] * main_other_end[0], tol
        )

        # If the start and endpoint of the other fracture are clearly on the
        # same side of the main one, these are not crossing.
        # For completely ovrelapping edges, the normalization will leave the
        # vectors nan. There may be better ways of dealing with this, but we simply
        # run the intersection finder in this case.
        relevant = np.where(
            np.logical_or(
                (start_cross * end_cross < 1),
                np.any(np.isnan(main_other_start + main_other_end), axis=0),
            )
        )[0]

        # Loop over all relevant (possibly crossing) fractures, look closer
        # for an intersection.
        for ri in relevant:
            ipt = pp.cg.lines_intersect(
                start_main, end_main, pt(start_other, ri), pt(end_other, ri)
            )
            # Add the intersection point, if any.
            # If two intersection points are found, that is the edges are overlapping
            # both points are added.
            if ipt is not None:
                num_isect = ipt.shape[1]
                # Add indices of the new points to the main and other edge
                isect_pt[main] = np.append(
                    isect_pt[main], new_ind + np.arange(num_isect)
                )
                isect_pt[other[ri]] = np.append(
                    isect_pt[other[ri]], new_ind + np.arange(num_isect)
                )
                new_ind += num_isect

                # Add the one or two intertion points
                if num_isect == 1:
                    new_pts.append(ipt.squeeze())
                else:
                    # It turned out the transport was needed to get the code to work
                    new_pts.append(ipt.squeeze().T)

    # If we have found no intersection points, we can safely return the incoming
    # points and edges.
    if len(new_pts) == 0:
        return p, e
    # If intersection points are found, the intersecting lines must be split into
    # shorter segments.
    else:
        # The full set of points, both original and newly found intersection points
        all_pt = np.hstack((p, np.vstack((i for i in new_pts)).T))
        # Remove duplicates in the point set.
        # NOTE: The tolerance used here is a bit sensitive, if set too loose, this
        # may merge non-intersecting fractures.
        unique_all_pt, _, ib = pp.utils.setmembership.unique_columns_tol(all_pt, tol)
        # Data structure for storing the split edges.
        new_edge = np.empty((e.shape[0], 0))

        # Loop over all lines, split it into non-overlapping segments.
        for ei in range(num_lines):
            # Find indices of all points involved in this fracture.
            # Map them to the unique point set, and uniquify
            inds = np.unique(ib[np.hstack((e[:2, ei], isect_pt[ei]))])
            num_branches = inds.size - 1
            # Get the coordinates themselves.
            loc_pts = pt(unique_all_pt, inds)
            # Specifically get the start point: Pick one of the points of the
            # original edge, e[0, ei], which is known to be at an end of the edge.
            # map to the unique indices
            loc_start = pt(unique_all_pt, ib[e[0, ei]])
            # Measure the distance of the points from the start. This can be used
            # to sort the points along the line
            dist = np.sum((loc_pts - loc_start) ** 2, axis=0)
            order = np.argsort(dist)
            new_inds = inds[order]
            # All new segments share the tags of the old one.
            loc_tags = e[2:, ei].reshape((-1, 1)) * np.ones(num_branches)
            # Define the new segments, in terms of the unique points
            loc_edge = np.vstack((new_inds[:-1], new_inds[1:], loc_tags))

            # Add to the global list of segments
            new_edge = np.hstack((new_edge, loc_edge))

        # Finally, uniquify edges. This operation is necessary for overlapping edges.
        # Operate on sorted point indices per edge
        new_edge[:2] = np.sort(new_edge[:2], axis=0)
        # Uniquify.
        _, edge_map, _ = pp.utils.setmembership.unique_columns_tol(
            new_edge[:2].astype(np.int), tol
        )
        new_edge = new_edge[:, edge_map]

        return unique_all_pt, new_edge.astype(np.int)


def intersect_polygons_3d(polys, tol=1e-8):
    """ Compute the intersection between polygons embedded in 3d.

    Assumptions:
        * All polygons are convex. Non-convex polygons will simply be treated
          in a wrong way.
        * No polygon contains three points on a line, that is, an angle of pi.

    Parameters:
        polys (list of np.array): Each list item represents a polygon, specified
            by its vertexses as a numpy array, of dimension 3 x num_pts. There
            should be at least three vertexes in the polygon.
        tol (double, optional): Geometric tolerance for the computations.

    Returns:
        np.array: 3 x num_pt, intersection coordinates.
        np.array of lists: For each of the polygons, give the index of the intersection
            points, referring to the columns of the intersection coordinates.
        np.array of list: For each polygon, a list telling whether each of the intersections
            is on the boundary of the polygon or not. For polygon i, the first
            element in this list tells whether the point formed by point-indices
            0 and 1 in the previous return argument is on the boundary.
        list of tuples: Each list element is a 2-tuple with the indices of
            intersecting polygons.

    """
    # Obtain bounding boxes for the polygons
    x_min, x_max, y_min, y_max, z_min, z_max = pp.cg._axis_aligned_bounding_box_3d(
        polys
    )

    # Identify overlapping bounding boxes: First, use a fast method to find
    # overlapping rectangles in the xy-plane.
    pairs_xy = pp.cg._identify_overlapping_rectangles(x_min, x_max, y_min, y_max)
    # Next, find overlapping intervals in the z-directien
    pairs_z = pp.cg._identify_overlapping_intervals(z_min, z_max)

    # Finally, do the intersection
    pairs = pp.cg._intersect_pairs(pairs_xy, pairs_z)

    # Various utility functions
    def center(p):
        # Compute the mean coordinate of a set of points
        return p.mean(axis=1).reshape((-1, 1))

    def normalize(v):
        # Normalize a vector
        nrm = np.sqrt(np.sum(v ** 2, axis=0))
        return v / nrm

    def mod_sign(v, tol=1e-8):
        # Modified signum function: The value is 0 if it is very close to zero.
        sgn = np.sign(v)
        sgn[np.abs(v) < tol] = 0
        return sgn

    def intersection(start, end, normal, center):
        # Find a point p on the segment between start and end, so that the vector
        # p - center is perpendicular to normal

        # Vector along the segment
        dx = end - start
        dot_prod = np.sum(normal.ravel() * dx)
        assert np.abs(dot_prod) > 1e-6
        t = -np.sum((start - center.ravel()) * normal.ravel()) / dot_prod

        assert t >= 0 and t <= 1
        return start + t * dx

    def vector_pointset_point(a, b, tol=1e-4):
        # Create a set of non-zero vectors from a point in the plane spanned by
        # a, to all points in b
        found = None
        # Loop over all points in a, search for a point that is sufficiently
        # far away from b. Mainly this involves finding a point in a which is
        # not in b
        for i in range(a.shape[1]):
            dist = np.sqrt(np.sum((b - a[:, i].reshape((-1, 1))) ** 2, axis=0))
            if np.min(dist) > tol:
                found = i
                break
        if found is None:
            # All points in a are also in b. We could probably use some other
            # point in a, but this seems so strange that we will rather
            # raise an error, with the expectation that this should never happen.
            raise ValueError("Coinciding polygons")

        return b - a[:, found].reshape((-1, 1))

    num_polys = len(polys)

    # Storage array for storing the index of the intersection points for each polygon
    isect_pt = np.empty(num_polys, dtype=np.object)
    # Storage for whehter an intersection is on the boundary of a polygon
    is_bound_isect = np.empty_like(isect_pt)
    # Initialization
    for i in range(isect_pt.size):
        isect_pt[i] = []
        is_bound_isect[i] = []

    # Array for storing the newly found points
    new_pt = []
    new_pt_ind = 0

    # Index of the main fractures, to which the other ones will be compared.
    start_inds = np.unique(pairs[0])

    polygon_pairs = []

    # Loop over all fracture pairs (taking more than one simultaneously if an index
    # occurs several times in pairs[0]), and look for intersections
    for di, line_ind in enumerate(start_inds):
        # The algorithm first does a coarse filtering, to check if the candidate
        # pairs both crosses each others plane. For those pairs that passes
        # this test, we next compute the intersection points, and check if
        # they are contained within the fractures.

        # The main fracture, from the first row in pairs
        main = line_ind

        # Find the other fracture of all pairs starting with the main one
        hit = np.where(pairs[0] == main)
        other = pairs[1, hit][0]

        # Center point and normal vector of the main fracture
        main_center = center(polys[main])
        main_normal = pp.cg.compute_normal(polys[main]).reshape((-1, 1))

        # Create an expanded version of the main points, so that the start
        # and end points are the same. Thus the segments can be formed by
        # merging main_p_expanded[:-1] with main_p_expanded[1:]
        num_main = polys[main].shape[1]
        ind_main_cyclic = np.arange(num_main + 1) % num_main
        main_p_expanded = polys[main][:, ind_main_cyclic]

        # Loop over the other polygon in the pairs, look for intersections
        for o in other:
            # Expanded version of the other polygon
            num_other = polys[o].shape[1]
            ind_other_cyclic = np.arange(num_other + 1) % num_other
            other_p_expanded = polys[o][:, ind_other_cyclic]

            # Normal vector and cetner of the other polygon
            other_normal = pp.cg.compute_normal(polys[o]).reshape((-1, 1))
            other_center = center(polys[o])

            # Point a vector from the main center to the vertexes of the
            # other polygon. Then take the dot product with the normal vector
            # of the main fracture. If all dot products have the same sign,
            # the other fracture does not cross the plane of the main polygon.
            # Note that we use mod_sign to safeguard the computation - if
            # the vertexes are close, we will take a closer look at the combination
            vec_from_main = normalize(
                vector_pointset_point(polys[main], other_p_expanded)
            )
            dot_prod_from_main = mod_sign(np.sum(main_normal * vec_from_main, axis=0))

            # Similar procedure: Vector from ohter center to the main polygon,
            # then dot product.
            vec_from_other = normalize(vector_pointset_point(polys[o], main_p_expanded))
            dot_prod_from_other = mod_sign(
                np.sum(other_normal * vec_from_other, axis=0)
            )

            # If one of the polygons lie completely on one side of the other,
            # there can be no intersection.
            if (
                np.all(dot_prod_from_main > 0)
                or np.all(dot_prod_from_main < 0)
                or np.all(dot_prod_from_other > 0)
                or np.all(dot_prod_from_other < 0)
            ):
                continue

            # At this stage, we are fairly sure both polygons cross the plane of
            # the other polygon.
            # Identify the segments where the polygon crosses the plane
            sign_change_main = np.where(np.abs(np.diff(dot_prod_from_main)) > 0)[0]
            sign_change_other = np.where(np.abs(np.diff(dot_prod_from_other)) > 0)[0]

            # The default option is that the intersection is not on the boundary
            # of main or other, that is, the two intersection points are identical
            # to two vertexes of the polygon
            isect_on_boundary_main = False
            isect_on_boundary_other = False

            if np.all(dot_prod_from_main != 0):
                # In the case where one polygon does not have a vertex in the plane of
                # the other polygon, there should be exactly two segments crossing the plane.
                assert sign_change_main.size == 2
                # Compute the intersection points between the segments of the other polygon
                # and the plane of the main polygon.
                other_intersects_main_0 = intersection(
                    other_p_expanded[:, sign_change_main[0]],
                    other_p_expanded[:, sign_change_main[0] + 1],
                    main_normal,
                    main_center,
                )
                other_intersects_main_1 = intersection(
                    other_p_expanded[:, sign_change_main[1]],
                    other_p_expanded[:, sign_change_main[1] + 1],
                    main_normal,
                    main_center,
                )
            elif np.sum(dot_prod_from_main[:-1] == 0) == 1:
                # The first and last element represent the same point, thus include
                # only one of them when counting the number of points in the plane
                # of the other fracture.
                hit = np.where(dot_prod_from_main[:-1] == 0)[0]
                other_intersects_main_0 = other_p_expanded[:, hit[0]]
                sign_change_full = np.where(np.abs(np.diff(dot_prod_from_main)) > 1)[0]
                other_intersects_main_1 = intersection(
                    other_p_expanded[:, sign_change_full[0]],
                    other_p_expanded[:, sign_change_full[0] + 1],
                    main_normal,
                    main_center,
                )
            elif np.all(dot_prod_from_main[:-1] == 0):
                # The two polygons lie in the same plane. The intersection points will
                # be found on the segments of the polygons
                isect = np.zeros((3, 0))
                # Loop over both set of polygon segments, look for intersections
                for sm in range(polys[main].shape[1]):
                    # Store the intersection points found for this segment of the main
                    # polygon. If there are more than one, we know that the intersection
                    # is on the boundary of that polygon.
                    tmp_isect = np.zeros((3, 0))
                    for so in range(polys[o].shape[1]):
                        loc_isect = segments_intersect_3d(
                            main_p_expanded[:, sm],
                            main_p_expanded[:, sm + 1],
                            other_p_expanded[:, so],
                            other_p_expanded[:, so + 1],
                        )
                        if loc_isect is None:
                            continue
                        else:
                            isect = np.hstack((isect, loc_isect))
                            tmp_isect = np.hstack((tmp_isect, loc_isect))

                    # Uniquify the intersection points found on this segment of main.
                    # If more than one, the intersection is on the boundary of main.
                    tmp_unique_isect, *rest = pp.utils.setmembership.unique_columns_tol(
                        tmp_isect, tol=tol
                    )
                    if tmp_unique_isect.shape[1] > 1:
                        isect_on_boundary_main = True

                isect, *rest = pp.utils.setmembership.unique_columns_tol(isect, tol=tol)

                if isect.shape[1] == 0:
                    # The polygons share a plane, but no intersections
                    continue
                elif isect.shape[1] == 1:
                    # Point contact. Not really sure what to do with this, ignore for now
                    continue
                elif isect.shape[1] == 2:
                    other_intersects_main_0 = isect[:, 0]
                    other_intersects_main_1 = isect[:, 1]
                else:
                    raise ValueError("There should be at most two intersections")

            else:
                # Both of the intersection points are vertexes.
                # Check that there are only two points - if this assertion fails,
                # there is a hanging node of the other polygon, which is in the
                # plane of the other polygon. Extending to cover this case should
                # be possible, but further treatment is unclear at the moment.
                assert np.sum(dot_prod_from_main[:-1] == 0) == 2
                hit = np.where(dot_prod_from_main[:-1] == 0)[0]
                other_intersects_main_0 = other_p_expanded[:, hit[0]]
                # Pick the last of the intersection points. This is valid also for
                # multiple (>2) intersection points, but we keep the assertion for now.
                other_intersects_main_1 = other_p_expanded[:, hit[1]]
                # The other polygon has an edge laying in the plane of the main polygon.
                # This will be registered as a boundary intersection, but only if
                # the polygons (not only plane) intersect.
                if (
                    hit[0] + 1 == hit[-1]
                    or hit[0] == 0
                    and hit[-1] == (dot_prod_from_main.size - 2)
                ):
                    isect_on_boundary_other = True

            if np.all(dot_prod_from_other != 0):
                # In the case where one polygon does not have a vertex in the plane of
                # the other polygon, there should be exactly two segments crossing the plane.
                assert sign_change_other.size == 2
                # Compute the intersection points between the segments of the main polygon
                # and the plane of the other polygon.
                main_intersects_other_0 = intersection(
                    main_p_expanded[:, sign_change_other[0]],
                    main_p_expanded[:, sign_change_other[0] + 1],
                    other_normal,
                    other_center,
                )
                main_intersects_other_1 = intersection(
                    main_p_expanded[:, sign_change_other[1]],
                    main_p_expanded[:, sign_change_other[1] + 1],
                    other_normal,
                    other_center,
                )
            elif np.sum(dot_prod_from_other[:-1] == 0) == 1:
                # The first and last element represent the same point, thus include
                # only one of them when counting the number of points in the plane
                # of the other fracture.
                hit = np.where(dot_prod_from_other[:-1] == 0)[0]
                main_intersects_other_0 = main_p_expanded[:, hit[0]]
                sign_change_full = np.where(np.abs(np.diff(dot_prod_from_other)) > 1)[0]
                main_intersects_other_1 = intersection(
                    main_p_expanded[:, sign_change_full[0]],
                    main_p_expanded[:, sign_change_full[0] + 1],
                    other_normal,
                    other_center,
                )
            elif np.all(dot_prod_from_other[:-1] == 0):

                isect = np.zeros((3, 0))
                for so in range(polys[o].shape[1]):
                    tmp_isect = np.zeros((3, 0))
                    for sm in range(polys[main].shape[1]):
                        loc_isect = segments_intersect_3d(
                            main_p_expanded[:, sm],
                            main_p_expanded[:, sm + 1],
                            other_p_expanded[:, so],
                            other_p_expanded[:, so + 1],
                        )
                        if loc_isect is None:
                            continue
                        else:
                            isect = np.hstack((isect, loc_isect))
                            tmp_isect = np.hstack((tmp_isect, loc_isect))

                    tmp_unique_isect, *rest = pp.utils.setmembership.unique_columns_tol(
                        tmp_isect, tol=tol
                    )

                    if tmp_unique_isect.shape[1] > 1:
                        isect_on_boundary_other = True

                isect, *rest = pp.utils.setmembership.unique_columns_tol(isect, tol=tol)

                if isect.shape[1] == 0:
                    # The polygons share a plane, but no intersections
                    continue
                elif isect.shape[1] == 1:
                    # Point contact. Not really sure what to do with this, continue for now
                    continue
                elif isect.shape[1] == 2:
                    main_intersects_other_0 = isect[:, 0]
                    main_intersects_other_1 = isect[:, 1]
                else:
                    raise ValueError("There should be at most two intersections")

            else:
                # Both of the intersection points are vertexes.
                # Check that there are only two points - if this assertion fails,
                # there is a hanging node of the main polygon, which is in the
                # plane of the other polygon. Extending to cover this case should
                # be possible, but further treatment is unclear at the moment.
                # Do not count the last point here, this is identical to the
                # first one.
                assert np.sum(dot_prod_from_other[:-1] == 0) == 2
                hit = np.where(dot_prod_from_other[:-1] == 0)[0]
                main_intersects_other_0 = main_p_expanded[:, hit[0]]
                # Pick the last of the intersection points. This is valid also for
                # multiple (>2) intersection points, but we keep the assertion for now.
                main_intersects_other_1 = main_p_expanded[:, hit[-1]]
                # The main polygon has an edge laying in the plane of the other polygon.
                # If the two intersection points form a segment
                # This will be registered as a boundary intersection, but only if
                # the polygons (not only plane) intersect.
                # The two points can either be one apart in the main polygon,
                # or it can be the first and the penultimate point
                # (in the latter case, the final point, which is identical to the
                # first one, will also be in the plane, but this is disregarded
                # by the [:-1] above)
                if (
                    hit[0] + 1 == hit[-1]
                    or hit[0] == 0
                    and hit[-1] == (dot_prod_from_other.size - 2)
                ):
                    isect_on_boundary_main = True

            # Vectors from the intersection points in the main fracture to the
            # intersection point in the other fracture
            main_0_other_0 = other_intersects_main_0 - main_intersects_other_0
            main_0_other_1 = other_intersects_main_1 - main_intersects_other_0
            main_1_other_0 = other_intersects_main_0 - main_intersects_other_1
            main_1_other_1 = other_intersects_main_1 - main_intersects_other_1

            # To finalize the computation, we need to sort out how the intersection
            # points are located relative to each other. Only if there is an overlap
            # between the intersection points of the main and the other polygon
            # is there a real intersection (contained within the polygons, not only)
            # in their planes, but outside the features themselves.

            # e_1 is positive if both points of the other fracture lie on the same side of the
            # first intersection point of the main one
            e_1 = np.sum(main_0_other_0 * main_0_other_1)
            # e_2 is positive if both points of the other fracture lie on the same side of the
            # second intersection point of the main one
            e_2 = np.sum(main_1_other_0 * main_1_other_1)
            # e_3 is positive if both points of the main fracture lie on the same side of the
            # first intersection point of the other one
            e_3 = np.sum((-main_0_other_0) * (-main_1_other_0))
            # e_3 is positive if both points of the main fracture lie on the same side of the
            # second intersection point of the other one
            e_4 = np.sum((-main_0_other_1) * (-main_1_other_1))

            # This is in essence an implementation of the flow chart in Figure 9 in Dong et al,
            # However the inequality signs are changed a bit to make the logic clearer
            if e_1 > 0 and e_2 > 0 and e_3 > 0 and e_4 > 0:
                # The intersection points for the two fractures are separated.
                # There is no intersection
                continue
            if e_1 >= 0:
                # The first point on the main fracture is not involved in the intersection
                if e_2 >= 0:
                    # The second point on the main fracture is not involved
                    # We know that e_3 and e_4 are negative (positive is covered above
                    # and a combination is not possible)
                    isect_pt_loc = [other_intersects_main_0, other_intersects_main_1]
                else:
                    # The second point on the main fracture is surrounded by points on
                    # the other fracture. One of them will in turn be surrounded by the
                    # points on the main fracture, this is the intersecting one.
                    if e_3 <= 0:
                        isect_pt_loc = [
                            main_intersects_other_1,
                            other_intersects_main_0,
                        ]
                    elif e_4 <= 0:
                        isect_pt_loc = [
                            main_intersects_other_1,
                            other_intersects_main_1,
                        ]
                    else:
                        # We may eventually end up here for overlapping fractures
                        assert False
            elif e_2 >= 0:
                # The first point on the main fracture is not involved in the intersection
                # The case of e_1 also non-negative was covered above
                if e_1 < 0:  # Equality is covered above
                    # The first point on the main fracture is surrounded by points on
                    # the other fracture. One of them will in turn be surrounded by the
                    # points on the main fracture, this is the intersecting one.
                    if e_3 <= 0:
                        isect_pt_loc = [
                            main_intersects_other_0,
                            other_intersects_main_0,
                        ]
                    elif e_4 <= 0:
                        isect_pt_loc = [
                            main_intersects_other_0,
                            other_intersects_main_1,
                        ]
                    else:
                        # We may eventually end up here for overlapping fractures
                        assert False
            elif e_1 < 0 and e_2 < 0:
                # The points in on the main fracture are the intersection points
                isect_pt_loc = [main_intersects_other_0, main_intersects_other_1]
            else:
                # This should never happen
                assert False

            new_pt.append(np.array(isect_pt_loc).T)
            num_new = len(isect_pt_loc)
            isect_pt[main].append(new_pt_ind + np.arange(num_new))
            isect_pt[o].append(new_pt_ind + np.arange(num_new))
            new_pt_ind += num_new
            is_bound_isect[main].append(isect_on_boundary_main)
            is_bound_isect[o].append(isect_on_boundary_other)
            polygon_pairs.append((main, o))

    if len(new_pt) > 0:
        new_pt = np.hstack((v for v in new_pt))
        for i in range(isect_pt.size):
            if len(isect_pt[i]) > 0:
                isect_pt[i] = np.hstack((v for v in isect_pt[i]))
            else:
                isect_pt[i] = np.empty(0)

    else:
        new_pt = np.empty((3, 0))
        for i in range(isect_pt.size):
            isect_pt[i] = np.empty(0)

    return new_pt, isect_pt, is_bound_isect, polygon_pairs


# ----------------------------------------------------------
#
# END OF FUNCTIONS RELATED TO SPLITTING OF INTERSECTING LINES IN 2D
#
# -----------------------------------------------------------


def is_ccw_polygon(poly):
    """
    Determine if the vertices of a polygon are sorted counter clockwise.

    The method computes the winding number of the polygon, see
        http://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
    and
        http://blog.element84.com/polygon-winding.html

    for descriptions of the algorithm.

    The algorithm should work for non-convex polygons. If the polygon is
    self-intersecting (shaped like the number 8), the number returned will
    reflect whether the method is 'mostly' cw or ccw.

    Parameters:
        poly (np.ndarray, 2xn): Polygon vertices.

    Returns:
        boolean, True if the polygon is ccw.

    See also:
        is_ccw_polyline

    Examples:
        >>> is_ccw_polygon(np.array([[0, 1, 0], [0, 0, 1]]))
        True

        >>> is_ccw_polygon(np.array([[0, 0, 1], [0, 1, 0]]))
        False

    """
    p_0 = np.append(poly[0], poly[0, 0])
    p_1 = np.append(poly[1], poly[1, 0])

    num_p = poly.shape[1]
    value = 0
    for i in range(num_p):
        value += (p_1[i + 1] + p_1[i]) * (p_0[i + 1] - p_0[i])
    return value < 0


# ----------------------------------------------------------


def is_ccw_polyline(p1, p2, p3, tol=0, default=False):
    """
    Check if the line segments formed by three points is part of a
    conuter-clockwise circle.

    The test is positiv if p3 lies to left of the line running through p1 and
    p2.

    The function is intended for 2D points; higher-dimensional coordinates will
    be ignored.

    Extensions to lines with more than three points should be straightforward,
    the input points should be merged into a 2d array.

    Parameters:
        p1 (np.ndarray, length 2): Point on dividing line
        p2 (np.ndarray, length 2): Point on dividing line
        p3 (np.ndarray, length 2): Point to be tested
        tol (double, optional): Tolerance used in the comparison, can be used
            to account for rounding errors. Defaults to zero.
        default (boolean, optional): Mode returned if the point is within the
            tolerance. Should be set according to what is desired behavior of
            the function (will vary with application). Defaults to False.

    Returns:
        boolean, true if the points form a ccw polyline.

    See also:
        is_ccw_polygon

    """

    # Compute cross product between p1-p2 and p1-p3. Right hand rule gives that
    # p3 is to the left if the cross product is positive.
    cross_product = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (
        p3[0] - p1[0]
    )

    # Should there be a scaling of the tolerance relative to the distance
    # between the points?

    if np.abs(cross_product) <= tol:
        return default
    return cross_product > -tol


# -----------------------------------------------------------------------------


def is_inside_polygon(poly, p, tol=0, default=False):
    """
    Check if a set of points are inside a polygon.

    The method assumes that the polygon is convex.

    Paremeters:
        poly (np.ndarray, 2 x n): vertexes of polygon. The segments are formed by
            connecting subsequent columns of poly
        p (np.ndarray, 2 x n2): Points to be tested.
        tol (double, optional): Tolerance for rounding errors. Defaults to
            zero.
        default (boolean, optional): Default behavior if the point is close to
            the boundary of the polygon. Defaults to False.

    Returns:
        np.ndarray, boolean: Length equal to p, true if the point is inside the
            polygon.

    """
    if p.ndim == 1:
        pt = p.reshape((-1, 1))
    else:
        pt = p

    # The test uses is_ccw_polyline, and tacitly assumes that the polygon
    # vertexes is sorted in a ccw fashion. If this is not the case, flip the
    # order of the nodes on a copy, and use this for the testing.
    # Note that if the nodes are not cw nor ccw (e.g. they are crossing), the
    # test cannot be trusted anyhow.
    if not is_ccw_polygon(poly):
        poly = poly.copy()[:, ::-1]

    poly_size = poly.shape[1]

    inside = np.ones(pt.shape[1], dtype=np.bool)
    for i in range(pt.shape[1]):
        for j in range(poly.shape[1]):
            if not is_ccw_polyline(
                poly[:, j],
                poly[:, (j + 1) % poly_size],
                pt[:, i],
                tol=tol,
                default=default,
            ):
                inside[i] = False
                # No need to check the remaining segments of the polygon.
                break
    return inside


def is_inside_polyhedron(polyhedron, test_points, tol=1e-8):
    """ Test whether a set of point is inside a polyhedron.

    The actual algorithm and implementation used for the test can be found
    at

        https://github.com/mdickinson/polyhedron/blob/master/polyhedron.py

    By Mark Dickinson. From what we know, the implementation is only available
    on github (no pypi or similar), and we are also not aware of other
    implementations of algorithms for point-in-polyhedron problems that allows
    for non-convex polyhedra. Moreover, the file above has an unclear lisence.
    Therefore, to use the present function, download the above file and put it
    in the pythonpath with the name 'robust_point_in_polyhedron.py'
    (polyhedron.py seemed too general a name for this).

    Suggestions for better solutions to this are most welcome.

    Parameters:
        polyhedron (list of np.ndarray): Each list element represent a side
            of the polyhedron. Each side is assumed to be a convex polygon.
        test_points (np.ndarray, 3 x num_pt): Points to be tested.
        tol (double, optional): Geometric tolerance, used in comparison of
            points. Defaults to 1e-8.

    Returns:
        np.ndarray, size num_pt: Element i is 1 (True) if test_points[:, i] is
            inside the polygon.

    """
    # If you get an error message here, read documentation of the method.
    import robust_point_in_polyhedron

    # The actual test requires that the polyhedra surface is described by
    # a triangulation. To that end, loop over all polygons and compute
    # triangulation. This is again done by a projection to 2d

    # Data storage
    tri = np.zeros((0, 3))
    points = np.zeros((3, 0))

    num_points = 0
    for poly in polyhedron:
        R = project_plane_matrix(poly)
        p_2d = R.dot(poly)[:2]
        loc_tri = Delaunay(p_2d.T)
        simplices = loc_tri.simplices

        # Add the triangulation, with indices adjusted for the number of points
        # already added
        tri = np.vstack((tri, num_points + simplices))
        points = np.hstack((points, poly))
        num_points += simplices.max() + 1

    # Uniquify points, and update triangulation
    upoints, _, ib = pp.utils.setmembership.unique_columns_tol(points, tol=tol)
    ut = ib[tri.astype(np.int)]

    # The in-polyhedra algorithm requires a very particular ordering of the vertexes
    # in the triangulation. Fix this
    sorted_t = pp.utils.sort_points.sort_triangle_edges(ut.T).T

    # Generate tester for points
    test_object = robust_point_in_polyhedron.Polyhedron(sorted_t, upoints.T)

    if test_points.size < 4:
        test_points = test_points.reshape((-1, 1))

    is_inside = np.zeros(test_points.shape[1], dtype=np.bool)

    # Loop over all points, check if they are inside.
    for pi in range(test_points.shape[1]):
        # NOTE: The documentation of the test is a bit unclear, but it seems
        # a winding number of 0 means outside, non-zero is inside
        is_inside[pi] = np.abs(test_object.winding_number(test_points[:, pi])) > 0

    return is_inside


# -----------------------------------------------------------------------------


def lines_intersect(start_1, end_1, start_2, end_2, tol=1e-8):
    """
    Check if two line segments defined by their start end endpoints, intersect.

    The lines are assumed to be in 2D.

    Note that, oposed to other functions related to grid generation such as
    remove_edge_crossings, this function does not use the concept of
    snap_to_grid. This may cause problems at some point, although no issues
    have been discovered so far.

    Implementation note:
        This function can be replaced by a call to segments_intersect_3d. Todo.

    Example:
        >>> lines_intersect([0, 0], [1, 1], [0, 1], [1, 0])
        array([[ 0.5],
           [ 0.5]])

        >>> lines_intersect([0, 0], [1, 0], [0, 1], [1, 1])

    Parameters:
        start_1 (np.ndarray or list): coordinates of start point for first
            line.
        end_1 (np.ndarray or list): coordinates of end point for first line.
        start_2 (np.ndarray or list): coordinates of start point for first
            line.
        end_2 (np.ndarray or list): coordinates of end point for first line.

    Returns:
        np.ndarray (2 x num_pts): coordinates of intersection point, or the
            endpoints of the intersection segments if relevant. In the case of
            a segment, the first point (column) will be closest to start_1.  If
            the lines do not intersect, None is returned.

    Raises:
        ValueError if the start and endpoints of a line are the same.

    """
    start_1 = np.asarray(start_1).astype(np.float)
    end_1 = np.asarray(end_1).astype(np.float)
    start_2 = np.asarray(start_2).astype(np.float)
    end_2 = np.asarray(end_2).astype(np.float)

    # Vectors along first and second line
    d_1 = end_1 - start_1
    d_2 = end_2 - start_2

    length_1 = np.sqrt(np.sum(d_1 * d_1))
    length_2 = np.sqrt(np.sum(d_2 * d_2))

    # Vector between the start points
    d_s = start_2 - start_1

    # An intersection point is characterized by
    #   start_1 + d_1 * t_1 = start_2 + d_2 * t_2
    #
    # which on component form becomes
    #
    #   d_1[0] * t_1 - d_2[0] * t_2 = d_s[0]
    #   d_1[1] * t_1 - d_2[1] * t_2 = d_s[1]
    #
    # First check for solvability of the system (e.g. parallel lines) by the
    # determinant of the matrix.

    discr = d_1[0] * (-d_2[1]) - d_1[1] * (-d_2[0])

    # Check if lines are parallel.
    # The tolerance should be relative to the length of d_1 and d_2
    if np.abs(discr) < tol * length_1 * length_2:
        # The lines are parallel, and will only cross if they are also colinear
        logger.debug("The segments are parallel")
        # Cross product between line 1 and line between start points on line
        start_cross_line = d_s[0] * d_1[1] - d_s[1] * d_1[0]
        if np.abs(start_cross_line) < tol * max(length_1, length_2):
            logger.debug("Lines are colinear")
            # The lines are co-linear

            # Write l1 on the form start_1 + t * d_1, find the parameter value
            # needed for equality with start_2 and end_2
            if np.abs(d_1[0]) > tol * length_1:
                t_start_2 = (start_2[0] - start_1[0]) / d_1[0]
                t_end_2 = (end_2[0] - start_1[0]) / d_1[0]
            elif np.abs(d_1[1]) > tol * length_2:
                t_start_2 = (start_2[1] - start_1[1]) / d_1[1]
                t_end_2 = (end_2[1] - start_1[1]) / d_1[1]
            else:
                # d_1 is zero
                logger.error("Found what must be a point-edge")
                raise ValueError(
                    "Start and endpoint of line should be\
                                 different"
                )
            if t_start_2 < 0 and t_end_2 < 0:
                logger.debug("Lines are not overlapping")
                return None
            elif t_start_2 > 1 and t_end_2 > 1:
                logger.debug("Lines are not overlapping")
                return None
            # We have an overlap, find its parameter values
            t_min = max(min(t_start_2, t_end_2), 0)
            t_max = min(max(t_start_2, t_end_2), 1)

            if t_max - t_min < tol:
                # It seems this can only happen if they are also equal to 0 or
                # 1, that is, the lines share a single point
                p_1 = start_1 + d_1 * t_min
                logger.debug("Colinear lines share a single point")
                return p_1.reshape((-1, 1))

            logger.debug("Colinear lines intersect along segment")
            p_1 = start_1 + d_1 * t_min
            p_2 = start_1 + d_1 * t_max
            return np.array([[p_1[0], p_2[0]], [p_1[1], p_2[1]]])

        else:
            logger.debug("Lines are not colinear")
            # Lines are parallel, but not colinear
            return None
    else:
        # Solve linear system using Cramer's rule
        t_1 = (d_s[0] * (-d_2[1]) - d_s[1] * (-d_2[0])) / discr
        t_2 = (d_1[0] * d_s[1] - d_1[1] * d_s[0]) / discr

        isect_1 = start_1 + t_1 * d_1
        isect_2 = start_2 + t_2 * d_2
        # Safeguarding
        assert np.allclose(isect_1, isect_2, tol)

        # The intersection lies on both segments if both t_1 and t_2 are on the
        # unit interval.
        # Use tol to allow some approximations
        if t_1 >= -tol and t_1 <= (1 + tol) and t_2 >= -tol and t_2 <= (1 + tol):
            logger.debug("Segment intersection found in one point")
            return np.array([[isect_1[0]], [isect_1[1]]])

        return None


# ------------------------------------------------------------------------------#


def segments_intersect_3d(start_1, end_1, start_2, end_2, tol=1e-8):
    """
    Find intersection points (or segments) of two 3d lines.

    Note that, oposed to other functions related to grid generation such as
    remove_edge_crossings, this function does not use the concept of
    snap_to_grid. This may cause problems at some point, although no issues
    have been discovered so far.

    Parameters:
        start_1 (np.ndarray or list): coordinates of start point for first
            line.
        end_1 (np.ndarray or list): coordinates of end point for first line.
        start_2 (np.ndarray or list): coordinates of start point for first
            line.
        end_2 (np.ndarray or list): coordinates of end point for first line.

    Returns:
        np.ndarray, dimension 3xn_pts): coordinates of intersection points
            (number of columns will be either 1 for a point intersection, or 2
            for a segment intersection). If the lines do not intersect, None is
            returned.

    """

    # Convert input to numpy if necessary
    start_1 = np.asarray(start_1).astype(np.float).ravel()
    end_1 = np.asarray(end_1).astype(np.float).ravel()
    start_2 = np.asarray(start_2).astype(np.float).ravel()
    end_2 = np.asarray(end_2).astype(np.float).ravel()

    # Short hand for component of start and end points, as well as vectors
    # along lines.
    xs_1 = start_1[0]
    ys_1 = start_1[1]
    zs_1 = start_1[2]

    xe_1 = end_1[0]
    ye_1 = end_1[1]
    ze_1 = end_1[2]

    dx_1 = xe_1 - xs_1
    dy_1 = ye_1 - ys_1
    dz_1 = ze_1 - zs_1

    xs_2 = start_2[0]
    ys_2 = start_2[1]
    zs_2 = start_2[2]

    xe_2 = end_2[0]
    ye_2 = end_2[1]
    ze_2 = end_2[2]

    dx_2 = xe_2 - xs_2
    dy_2 = ye_2 - ys_2
    dz_2 = ze_2 - zs_2

    # The lines are parallel in the x-y plane, but we don't know about the
    # z-direction. CHeck this
    deltas_1 = np.array([dx_1, dy_1, dz_1])
    deltas_2 = np.array([dx_2, dy_2, dz_2])

    # Use masked arrays to avoid divisions by zero
    mask_1 = np.ma.greater(np.abs(deltas_1), tol)
    mask_2 = np.ma.greater(np.abs(deltas_2), tol)

    # Check for two dimensions that are not parallel with at least one line
    mask_sum = mask_1 + mask_2
    if mask_sum.sum() > 1:
        in_discr = np.argwhere(mask_sum)[:2]
    else:
        # We're going to have a zero discreminant anyhow, just pick some dimensions.
        in_discr = np.arange(2)

    not_in_discr = np.setdiff1d(np.arange(3), in_discr)[0]
    discr = (
        deltas_1[in_discr[0]] * deltas_2[in_discr[1]]
        - deltas_1[in_discr[1]] * deltas_2[in_discr[0]]
    )

    # An intersection will be a solution of the linear system
    #   xs_1 + dx_1 * t_1 = xs_2 + dx_2 * t_2 (1)
    #   ys_1 + dy_1 * t_1 = ys_2 + dy_2 * t_2 (2)
    #
    # In addition, the solution should satisfy
    #   zs_1 + dz_1 * t_1 = zs_2 + dz_2 * t_2 (3)
    #
    # The intersection is on the line segments if 0 <= (t_1, t_2) <= 1

    # Either the lines are parallel in two directions
    if np.abs(discr) < tol:
        # If the lines are (almost) parallel, there is no single intersection,
        # but it may be a segment

        # First check if the third dimension is also parallel, if not, no
        # intersection

        # A first, simple test
        if np.any(mask_1 != mask_2):
            return None

        t = deltas_1[mask_1] / deltas_2[mask_2]

        # Second, test for alignment in all directions
        if not np.allclose(t, t.mean(), tol):
            return None

        # If we have made it this far, the lines are indeed parallel. Next,
        # check that they lay along the same line.
        diff_start = start_2 - start_1

        dstart_x_delta_x = diff_start[1] * deltas_1[2] - diff_start[2] * deltas_1[1]
        if np.abs(dstart_x_delta_x) > tol:
            return None
        dstart_x_delta_y = diff_start[2] * deltas_1[0] - diff_start[0] * deltas_1[2]
        if np.abs(dstart_x_delta_y) > tol:
            return None
        dstart_x_delta_z = diff_start[0] * deltas_1[1] - diff_start[1] * deltas_1[0]
        if np.abs(dstart_x_delta_z) > tol:
            return None

        # For dimensions with an incline, the vector between segment start
        # points should be parallel to the segments.
        # Since the masks are equal, we can use any of them.
        # For dimensions with no incline, the start cooordinates should be the same
        if not np.allclose(start_1[~mask_1], start_2[~mask_1], tol):
            return None

        # We have overlapping lines! finally check if segments are overlapping.

        # Since everything is parallel, it suffices to work with a single coordinate
        s_1 = start_1[mask_1][0]
        e_1 = end_1[mask_1][0]
        s_2 = start_2[mask_1][0]
        e_2 = end_2[mask_1][0]

        max_1 = max(s_1, e_1)
        min_1 = min(s_1, e_1)
        max_2 = max(s_2, e_2)
        min_2 = min(s_2, e_2)

        # Rule out case with non-overlapping segments
        if max_1 < min_2:
            return None
        elif max_2 < min_1:
            return None

        # The lines are overlapping, we need to find their common line
        lines = np.array([s_1, e_1, s_2, e_2])
        sort_ind = np.argsort(lines)

        # The overlap will be between the middle two points in the sorted list
        target = sort_ind[1:3]

        # Array of the full coordinates - same order as lines
        lines_full = np.vstack((start_1, end_1, start_2, end_2)).transpose()
        # Our segment consists of the second and third column. We're done!
        return lines_full[:, target]

    # or we are looking for a point intersection
    else:
        # Solve 2x2 system by Cramer's rule

        discr = deltas_1[in_discr[0]] * (-deltas_2[in_discr[1]]) - deltas_1[
            in_discr[1]
        ] * (-deltas_2[in_discr[0]])
        t_1 = (
            (start_2[in_discr[0]] - start_1[in_discr[0]]) * (-deltas_2[in_discr[1]])
            - (start_2[in_discr[1]] - start_1[in_discr[1]]) * (-deltas_2[in_discr[0]])
        ) / discr

        t_2 = (
            deltas_1[in_discr[0]] * (start_2[in_discr[1]] - start_1[in_discr[1]])
            - deltas_1[in_discr[1]] * (start_2[in_discr[0]] - start_1[in_discr[0]])
        ) / discr

        # Check that we are on line segment
        if t_1 < 0 or t_1 > 1 or t_2 < 0 or t_2 > 1:
            return None

        # Compute the z-coordinates of the intersection points
        z_1_isect = start_1[not_in_discr] + t_1 * deltas_1[not_in_discr]
        z_2_isect = start_2[not_in_discr] + t_2 * deltas_2[not_in_discr]

        if np.abs(z_1_isect - z_2_isect) < tol:
            vec = np.zeros(3)
            vec[in_discr] = start_1[in_discr] + t_1 * deltas_1[in_discr]
            vec[not_in_discr] = z_1_isect
            return vec.reshape((-1, 1))
        else:
            return None


# ----------------------------------------------------------------------------#

# Represent the polygon as a sympy polygon
def _np2p(p):
    # Convert a numpy point array (3xn) to sympy points
    if p.ndim == 1:
        return geom.Point(p[:])
    else:
        return [geom.Point(p[:, i]) for i in range(p.shape[1])]


def _p2np(p):
    # Convert sympy points to numpy format. If more than one point, these should be sent as a list
    if isinstance(p, list):
        return np.array(list([i.args for i in p]), dtype="float").transpose()
    else:
        return np.array(list(p.args), dtype="float").reshape((-1, 1))


def _to3D(p):
    # Add a third dimension
    return np.vstack((p, np.zeros(p.shape[1])))


# -------------------------------------------------------------------


def polygon_boundaries_intersect(poly_1, poly_2, tol=1e-8):
    """
    Test for intersection between the bounding segments of two 3D polygons.

    No tests are done for intersections with polygons interiors. The code has
    only been tested for convex polygons, status for non-convex polygons is
    unknown.

    A segment can either be intersected by segments of the other polygon as
     i) a single point
     ii) two points, hit by different segments (cannot be more than 2 for
         convex polygons)
     iii) along a line, if the segments are parallel.

    Note that if the polygons share a vertex, this point will be found multiple
    times (depending on the configuration of the polygons).

    Each intersection is represented as a list of three items. The first two
    are the segment index (numbered according to start point) of the
    intersecting segments. The third is the coordinates of the intersection
    point, this can either be a single point (3x1 nd.array), or a 3x2 array
    with columns representing the same (shared vertex) or different (shared
    segment) points.

    Paremeters:
        poly_1 (np.array, 3 x n_pt): First polygon, assumed to be
        non-intersecting. The closing segment is defined by the last and first
            column.
        poly_2 (np.array, 3 x n_pt): Second polygon, assumed to be
        non-intersecting. The closing segment is defined by the last and first
            column.
        tol (float, optional): Tolerance used for equality.

    Returns:
        list: of intersections. See above for description of data format. If no
        intersections are found, an empty list is returned.

    """
    l_1 = poly_1.shape[1]
    ind_1 = np.append(np.arange(l_1), 0)
    l_2 = poly_2.shape[1]
    ind_2 = np.append(np.arange(l_2), 0)

    isect = []

    for i in range(l_1):
        p_1_1 = poly_1[:, ind_1[i]]
        p_1_2 = poly_1[:, ind_1[i + 1]]

        for j in range(l_2):
            p_2_1 = poly_2[:, ind_2[j]]
            p_2_2 = poly_2[:, ind_2[j + 1]]
            isect_loc = segments_intersect_3d(p_1_1, p_1_2, p_2_1, p_2_2)
            if isect_loc is not None:
                isect.append([i, j, isect_loc])

    return isect


# ----------------------------------------------------------


def polygon_segment_intersect(poly_1, poly_2, tol=1e-8, include_bound_pt=True):
    """
    Find intersections between polygons embeded in 3D.

    The intersections are defined as between the interior of the first polygon
    and the boundary of the second, although intersections on the boundary of
    both polygons can also be picked up sometimes. If you need to distinguish
    between the two, the safer option is to also call
    polygon_boundary_intersect(), and compare the results.

    Parameters:
        poly_1 (np.ndarray, 3xn1): Vertexes of polygon, assumed ordered as cw or
            ccw.
        poly_2 (np.ndarray, 3xn2): Vertexes of second polygon, assumed ordered
            as cw or ccw.
        tol (double, optional): Tolerance for when two points are equal.
            Defaults to 1e-8.
        include_bound_pt (boolean, optional): Include cases where a segment is
            in the plane of the first ploygon, and the segment crosses the
            polygon boundary. Defaults to True.

    Returns:
        np.ndarray, size 3 x num_isect, coordinates of intersection points; or
            None if no intersection is found (may change to empty array of size
            (3, 0)).

    Raises:
        NotImplementedError if the two polygons overlap in a 2D area. An
        extension should not be difficult, but the function is not intended for
        this use.

    """

    # First translate the points so that the first plane is located at the origin
    center_1 = np.mean(poly_1, axis=1).reshape((-1, 1))
    poly_1 = poly_1 - center_1
    poly_2 = poly_2 - center_1

    # Obtain the rotation matrix that projects p1 to the xy-plane
    rot_p_1 = project_plane_matrix(poly_1)
    irot = rot_p_1.transpose()
    poly_1_rot = rot_p_1.dot(poly_1)

    # Sanity check: The points should lay on a plane
    assert np.amax(np.abs(poly_1_rot[2])) / np.amax(np.abs(poly_1_rot[:2])) < tol

    # Drop the z-coordinate
    poly_1_xy = poly_1_rot[:2]

    # Make sure the xy-polygon is ccw.
    if not is_ccw_polygon(poly_1_xy):
        poly_1_xy = poly_1_xy[:, ::-1]

    # Rotate the second polygon with the same rotation matrix
    poly_2_rot = rot_p_1.dot(poly_2)

    # If the rotation of whole second point cloud lies on the same side of z=0,
    # there are no intersections.
    if poly_2_rot[2].min() > tol:
        return None
    elif poly_2_rot[2].max() < -tol:
        return None

    # Check if the second plane is parallel to the first (same xy-plane)
    dz_2 = poly_2_rot[2].max() - poly_2_rot[2].min()
    if dz_2 < tol:
        if poly_2_rot[2].max() < tol:
            # The polygons are parallel, and in the same plane
            # Represent second polygon by sympy, and use sympy function to
            # detect intersection.
            # Convert the first polygon to sympy format
            poly_1_sp = geom.Polygon(*_np2p(poly_1_xy))
            poly_2_sp = geom.Polygon(*_np2p(poly_2_rot[:2]))

            isect = poly_1_sp.intersection(poly_2_sp)
            if isinstance(isect, list) and len(isect) > 0:
                # It would have been possible to return the intersecting area,
                # but this is not the intended behavior of the function.
                # Instead raise an error, and leave it to the user to deal with
                # this.
                raise NotImplementedError
            else:
                return None
        else:
            # Polygons lies in different parallel planes. No intersection
            return None
    else:
        # Loop over all boundary segments of the second plane. Check if they
        # intersect with the first polygon.
        # TODO: Special treatment of the case where one or two vertexes lies in
        # the plane of the poly_1
        num_p2 = poly_2.shape[1]
        # Roling indexing
        ind = np.append(np.arange(num_p2), np.zeros(1)).astype("int")

        isect = np.empty((3, 0))

        for i in range(num_p2):

            # Coordinates of this segment
            pt_1 = poly_2_rot[:, ind[i]]
            pt_2 = poly_2_rot[:, ind[i + 1]]

            # Check if segment crosses z=0 in the rotated coordinates
            if max(pt_1[2], pt_2[2]) < -tol or min(pt_1[2], pt_2[2]) > tol:
                continue

            dx = pt_2[0] - pt_1[0]
            dy = pt_2[1] - pt_1[1]
            dz = pt_2[2] - pt_1[2]
            if np.abs(dz) > tol:
                # We are on a plane, and we know that dz_2 is non-zero, so all
                # individiual segments must have an incline.
                # Parametrize the line, find parameter value for intersection
                # with z=0.
                t = (-pt_1[2] - 0) / dz

                # Sanity check. We have ruled out segments not crossing the
                # origin above.
                if t < -tol or t > 1 + tol:
                    continue

                # x and y-coordinate for z=0
                x0 = pt_1[0] + dx * t
                y0 = pt_1[1] + dy * t
                # Representation as point
                p_00 = np.array([x0, y0]).reshape((-1, 1))

                # Check if the first polygon encloses the point. When applied
                # to fracture intersections of T-type (segment embedded in the
                # plane of another fracture), it turned out to be useful to be
                # somewhat generous with the definition of the intersection.
                # Therefore, allow for intersections that are slightly outside
                # the polygon, and use the projection onto the polygon.

                start = np.arange(poly_1_xy.shape[1])
                end = np.r_[np.arange(1, poly_1_xy.shape[1]), 0]

                poly_1_to3D = _to3D(poly_1_xy)
                p_00_to3D = _to3D(p_00)
                dist, cp = dist_points_segments(
                    p_00_to3D, poly_1_to3D[:, start], poly_1_to3D[:, end]
                )
                mask = np.where(dist[0] < tol)[0]
                if mask.size > 0:
                    cp = cp[0].T[:, mask]
                    isect = np.hstack((isect, irot.dot(cp) + center_1))
                else:
                    dist, cp, ins = dist_points_polygon(p_00_to3D, poly_1_to3D)
                    if (dist[0] < tol and include_bound_pt) or dist[0] < 1e-12:
                        isect = np.hstack((isect, irot.dot(cp) + center_1))

            elif np.abs(pt_1[2]) < tol and np.abs(pt_2[2]) < tol:
                # The segment lies completely within the polygon plane.
                both_pts = np.vstack((pt_1, pt_2)).T
                # Find points within tho polygon itself
                inside = is_inside_polygon(poly_1_xy, both_pts[:2], tol=tol)

                if inside.all():
                    # Both points are inside, add and go on
                    isect = np.hstack((isect, irot.dot(both_pts) + center_1))
                else:
                    # A single point is inside. Need to find the intersection between this line segment and the polygon
                    if inside.any():
                        isect_loc = both_pts[:2, inside].reshape((2, -1))
                        p1 = both_pts[:, inside]
                        p2 = both_pts[:, np.logical_not(inside)]
                    else:
                        isect_loc = np.empty((2, 0))
                        p1 = both_pts[:, 0]
                        p2 = both_pts[:, 1]

                    # If a single internal point is found
                    if isect_loc.shape[1] == 1 or include_bound_pt:
                        poly_1_start = poly_1_rot
                        poly_1_end = np.roll(poly_1_rot, 1, axis=1)
                        for j in range(poly_1.shape[1]):
                            ip = segments_intersect_3d(
                                p1, p2, poly_1_start[:, j], poly_1_end[:, j]
                            )
                            if ip is not None:
                                isect_loc = np.hstack((isect_loc, ip[:2]))

                    isect = np.hstack((isect, irot.dot(_to3D(isect_loc)) + center_1))

        if isect.shape[1] == 0:
            isect = None

        # For points lying in the plane of poly_1, the same points may be found
        # several times
        if isect is not None:
            isect, _, _ = setmembership.unique_columns_tol(isect, tol=tol)
        return isect


# ------------------------------------------------------------------------------#


def is_planar(pts, normal=None, tol=1e-5):
    """ Check if the points lie on a plane.

    Parameters:
    pts (np.ndarray, 3xn): the points.
    normal: (optional) the normal of the plane, otherwise three points are
        required.

    Returns:
    check, bool, if the points lie on a plane or not.

    """

    if normal is None:
        normal = compute_normal(pts)
    else:
        normal = normal.flatten() / np.linalg.norm(normal)

    check_all = np.zeros(pts.shape[1] - 1, dtype=np.bool)

    for idx, p in enumerate(pts[:, 1:].T):
        den = np.linalg.norm(pts[:, 0] - p)
        dotprod = np.dot(normal, (pts[:, 0] - p) / (den if den else 1))
        check_all[idx] = np.isclose(dotprod, 0, atol=tol, rtol=0)

    return np.all(check_all)


# ------------------------------------------------------------------------------#


def is_point_in_cell(poly, p, if_make_planar=True):
    """
    Check whatever a point is inside a cell. Note a similar behaviour could be
    reached using the function is_inside_polygon, however the current
    implementation deals with concave cells as well. Not sure which is the best,
    in term of performances, for convex cells.

    Parameters:
        poly (np.ndarray, 3xn): vertexes of polygon. The segments are formed by
            connecting subsequent columns of poly.
        p (np.array, 3x1): Point to be tested.
    if_make_planar (optional, default True): The cell needs to lie on (s, t)
        plane. If not already done, this flag need to be used.

    Return:
        boolean, if the point is inside the cell. If a point is on the boundary
        of the cell the result may be either True or False.
    """
    p.shape = (3, 1)
    if if_make_planar:
        R = project_plane_matrix(poly)
        poly = np.dot(R, poly)
        p = np.dot(R, p)

    j = poly.shape[1] - 1
    is_odd = False

    for i in np.arange(poly.shape[1]):
        if (poly[1, i] < p[1] and poly[1, j] >= p[1]) or (
            poly[1, j] < p[1] and poly[1, i] >= p[1]
        ):
            if (
                poly[0, i]
                + (p[1] - poly[1, i])
                / (poly[1, j] - poly[1, i])
                * (poly[0, j] - poly[0, i])
            ) < p[0]:
                is_odd = not is_odd
        j = i

    return is_odd


# ------------------------------------------------------------------------------#


def project_plane_matrix(
    pts, normal=None, tol=1e-5, reference=[0, 0, 1], check_planar=True
):
    """ Project the points on a plane using local coordinates.

    The projected points are computed by a dot product.
    example: np.dot( R, pts )

    Parameters:
    pts (np.ndarray, 3xn): the points.
    normal: (optional) the normal of the plane, otherwise three points are
        required.
    tol: (optional, float) tolerance to assert the planarity of the cloud of
        points. Default value 1e-5.
    reference: (optional, np.array, 3x1) reference vector to compute the angles.
        Default value [0, 0, 1].

    Returns:
    np.ndarray, 3x3, projection matrix.

    """

    if normal is None:
        normal = compute_normal(pts)
    else:
        normal = np.asarray(normal)
        normal = normal.flatten() / np.linalg.norm(normal)

    if check_planar:
        assert is_planar(pts, normal, tol)

    reference = np.asarray(reference, dtype=np.float)
    angle = np.arccos(np.dot(normal, reference))
    vect = np.cross(normal, reference)
    return rot(angle, vect)


# ------------------------------------------------------------------------------#


def project_line_matrix(pts, tangent=None, tol=1e-5, reference=[0, 0, 1]):
    """ Project the points on a line using local coordinates.

    The projected points are computed by a dot product.
    example: np.dot( R, pts )

    Parameters:
    pts (np.ndarray, 3xn): the points.
    tangent: (optional) the tangent unit vector of the plane, otherwise two
        points are required.

    Returns:
    np.ndarray, 3x3, projection matrix.

    """

    if tangent is None:
        tangent = compute_tangent(pts)
    else:
        tangent = tangent.flatten() / np.linalg.norm(tangent)

    reference = np.asarray(reference, dtype=np.float)
    angle = np.arccos(np.dot(tangent, reference))
    vect = np.cross(tangent, reference)
    return rot(angle, vect)


# ------------------------------------------------------------------------------#


def rot(a, vect):
    """ Compute the rotation matrix about a vector by an angle using the matrix
    form of Rodrigues formula.

    Parameters:
    a: double, the angle.
    vect: np.array, 1x3, the vector.

    Returns:
    matrix: np.ndarray, 3x3, the rotation matrix.

    """
    if np.allclose(vect, [0.0, 0.0, 0.0]):
        return np.identity(3)
    vect = vect / np.linalg.norm(vect)

    # Prioritize readability over PEP0008 whitespaces.
    # pylint: disable=bad-whitespace
    W = np.array(
        [[0.0, -vect[2], vect[1]], [vect[2], 0.0, -vect[0]], [-vect[1], vect[0], 0.0]]
    )
    return (
        np.identity(3)
        + np.sin(a) * W
        + (1.0 - np.cos(a)) * np.linalg.matrix_power(W, 2)
    )


# ------------------------------------------------------------------------------#


def normal_matrix(pts=None, normal=None):
    """ Compute the normal projection matrix of a plane.

    The algorithm assume that the points lie on a plane.
    Three non-aligned points are required.

    Either points or normal are mandatory.

    Parameters:
    pts (optional): np.ndarray, 3xn, the points. Need n > 2.
    normal (optional): np.array, 1x3, the normal.

    Returns:
    normal matrix: np.array, 3x3, the normal matrix.

    """
    if normal is not None:
        normal = normal / np.linalg.norm(normal)
    elif pts is not None:
        normal = compute_normal(pts)
    else:
        assert False, "Points or normal are mandatory"

    return np.tensordot(normal, normal, axes=0)


# ------------------------------------------------------------------------------#


def tangent_matrix(pts=None, normal=None):
    """ Compute the tangential projection matrix of a plane.

    The algorithm assume that the points lie on a plane.
    Three non-aligned points are required.

    Either points or normal are mandatory.

    Parameters:
    pts (optional): np.ndarray, 3xn, the points. Need n > 2.
    normal (optional): np.array, 1x3, the normal.

    Returns:
    tangential matrix: np.array, 3x3, the tangential matrix.

    """
    return np.eye(3) - normal_matrix(pts, normal)


# ------------------------------------------------------------------------------#


def compute_normal(pts):
    """ Compute the normal of a set of points.

    The algorithm assume that the points lie on a plane.
    Three non-aligned points are required.

    Parameters:
    pts: np.ndarray, 3xn, the points. Need n > 2.

    Returns:
    normal: np.array, 1x3, the normal.

    """

    assert pts.shape[1] > 2
    normal = np.cross(pts[:, 0] - pts[:, 1], compute_tangent(pts))
    if np.allclose(normal, np.zeros(3)):
        return compute_normal(pts[:, 1:])
    return normal / np.linalg.norm(normal)


# ------------------------------------------------------------------------------#


def compute_normals_1d(pts):
    t = compute_tangent(pts)
    n = np.array([t[1], -t[0], 0]) / np.sqrt(t[0] ** 2 + t[1] ** 2)
    return np.r_["1,2,0", n, np.dot(rot(np.pi / 2.0, t), n)]


# ------------------------------------------------------------------------------#


def compute_tangent(pts):
    """ Compute a tangent vector of a set of points.

    The algorithm assume that the points lie on a plane.

    Parameters:
    pts: np.ndarray, 3xn, the points.

    Returns:
    tangent: np.array, 1x3, the tangent.

    """

    mean_pts = np.mean(pts, axis=1).reshape((-1, 1))
    # Set of possible tangent vector. We can pick any of these, as long as it
    # is nonzero
    tangent = pts - mean_pts
    # Find the point that is furthest away from the mean point
    max_ind = np.argmax(np.sum(tangent ** 2, axis=0))
    tangent = tangent[:, max_ind]
    assert not np.allclose(tangent, np.zeros(3))
    return tangent / np.linalg.norm(tangent)


# ------------------------------------------------------------------------------#


def is_collinear(pts, tol=1e-5):
    """ Check if the points lie on a line.

    Parameters:
        pts (np.ndarray, 3xn): the points.
        tol (double, optional): Absolute tolerance used in comparison.
            Defaults to 1e-5.

    Returns:
        boolean, True if the points lie on a line.

    """

    if pts.shape[1] == 1 or pts.shape[1] == 2:
        return True

    pt0 = pts[:, 0]
    pt1 = pts[:, 1]

    dist = 1
    for i in np.arange(pts.shape[1]):
        for j in np.arange(i + 1, pts.shape[1]):
            dist = max(dist, np.linalg.norm(pts[:, i] - pts[:, j]))

    coll = (
        np.array([np.linalg.norm(np.cross(p - pt0, pt1 - pt0)) for p in pts[:, 1:-1].T])
        / dist
    )
    return np.allclose(coll, np.zeros(coll.size), atol=tol, rtol=0)


# ------------------------------------------------------------------------------#


def make_collinear(pts):
    """
    Given a set of points, return them aligned on a line.
    Useful to enforce collinearity for almost collinear points. The order of the
    points remain the same.
    NOTE: The first point in the list has to be on the extrema of the line.

    Parameter:
        pts: (3 x num_pts) the input points.

    Return:
        pts: (3 x num_pts) the corrected points.
    """
    assert pts.shape[1] > 1

    delta = pts - np.tile(pts[:, 0], (pts.shape[1], 1)).T
    dist = np.sqrt(np.einsum("ij,ij->j", delta, delta))
    end = np.argmax(dist)

    dist /= dist[end]

    return pts[:, 0, np.newaxis] * (1 - dist) + pts[:, end, np.newaxis] * dist


def project_points_to_line(p, tol=1e-4):
    """ Project a set of colinear points onto a line.

    The points should be co-linear such that a 1d description is meaningful.

    Parameters:
        p (np.ndarray, nd x n_pt): Coordinates of the points. Should be
            co-linear, but can have random ordering along the common line.
        tol (double, optional): Tolerance used for testing of co-linearity.

    Returns:
        np.ndarray, n_pt: 1d coordinates of the points, sorted along the line.
        np.ndarray (3x3): Rotation matrix used for mapping the points onto a
            coordinate axis.
        int: The dimension which onto which the point coordinates were mapped.
        np.ndarary (n_pt): Index array used to sort the points onto the line.

    Raises:
        ValueError if the points are not aligned on a line.

    """
    center = np.mean(p, axis=1).reshape((-1, 1))
    p -= center

    if p.shape[0] == 2:
        p = np.vstack((p, np.zeros(p.shape[1])))

    # Check that the points indeed form a line
    if not is_collinear(p, tol):
        raise ValueError("Elements are not colinear")
    # Find the tangent of the line
    tangent = compute_tangent(p)
    # Projection matrix
    rot = project_line_matrix(p, tangent)

    p_1d = rot.dot(p)
    # The points are now 1d along one of the coordinate axis, but we
    # don't know which yet. Find this.
    sum_coord = np.sum(np.abs(p_1d), axis=1)
    sum_coord /= np.amax(sum_coord)
    active_dimension = np.logical_not(np.isclose(sum_coord, 0, atol=tol, rtol=0))

    # Check that we are indeed in 1d
    assert np.sum(active_dimension) == 1
    # Sort nodes, and create grid
    coord_1d = p_1d[active_dimension]
    sort_ind = np.argsort(coord_1d)[0]
    sorted_coord = coord_1d[0, sort_ind]

    return sorted_coord, rot, active_dimension, sort_ind


# ------------------------------------------------------------------------------#


def map_grid(g, tol=1e-5):
    """ If a 2d or a 1d grid is passed, the function return the cell_centers,
    face_normals, and face_centers using local coordinates. If a 3d grid is
    passed nothing is applied. The return vectors have a reduced number of rows.

    Parameters:
    g (grid): the grid.

    Returns:
    cell_centers: (g.dim x g.num_cells) the mapped centers of the cells.
    face_normals: (g.dim x g.num_faces) the mapped normals of the faces.
    face_centers: (g.dim x g.num_faces) the mapped centers of the faces.
    R: (3 x 3) the rotation matrix used.
    dim: indicates which are the dimensions active.
    nodes: (g.dim x g.num_nodes) the mapped nodes.

    """
    cell_centers = g.cell_centers
    face_normals = g.face_normals
    face_centers = g.face_centers
    nodes = g.nodes
    R = np.eye(3)

    if g.dim == 0 or g.dim == 3:
        return (
            cell_centers,
            face_normals,
            face_centers,
            R,
            np.ones(3, dtype=bool),
            nodes,
        )

    if g.dim == 1 or g.dim == 2:

        if g.dim == 2:
            R = project_plane_matrix(g.nodes)
        else:
            R = project_line_matrix(g.nodes)

        face_centers = np.dot(R, face_centers)

        check = np.sum(np.abs(face_centers.T - face_centers[:, 0]), axis=0)
        check /= np.sum(check)
        dim = np.logical_not(np.isclose(check, 0, atol=tol, rtol=0))
        assert g.dim == np.sum(dim)
        face_centers = face_centers[dim, :]
        cell_centers = np.dot(R, cell_centers)[dim, :]
        face_normals = np.dot(R, face_normals)[dim, :]
        nodes = np.dot(R, nodes)[dim, :]

    return cell_centers, face_normals, face_centers, R, dim, nodes


# ------------------------------------------------------------------------------#


def dist_segment_set(start, end):
    """ Compute distance and closest points between sets of line segments.

    Parameters:
        start (np.array, nd x num_segments): Start points of segments.
        end (np.array, nd x num_segments): End points of segments.

    Returns:
        np.array, num_segments x num_segments: Distances between segments.
        np.array, num_segments x num_segments x nd: For segment i and j,
            element [i, j] gives the point on i closest to segment j.

    """
    if start.size < 4:
        start = start.reshape((-1, 1))
    if end.size < 4:
        end = end.reshape((-1, 1))

    nd = start.shape[0]
    ns = start.shape[1]

    d = np.zeros((ns, ns))
    cp = np.zeros((ns, ns, nd))

    for i in range(ns):
        cp[i, i, :] = start[:, i] + 0.5 * (end[:, i] - start[:, i])
        for j in range(i + 1, ns):
            dl, cpi, cpj = dist_two_segments(
                start[:, i], end[:, i], start[:, j], end[:, j]
            )
            d[i, j] = dl
            d[j, i] = dl
            cp[i, j, :] = cpi
            cp[j, i, :] = cpj

    return d, cp


# ------------------------------------------------------------------------------#


def dist_segment_segment_set(start, end, start_set, end_set):
    """ Compute distance and closest points between a segment and a set of
    segments.

    Parameters:
        start (np.array, nd x 1): Start point of the main segment
        end (np.array, nd x 1): End point of the main segment
        start_set (np.array, nd x n_segments): Start points for the segment set.
        end_set (np.array, nd x n_segments): End points for the segment set.

    Returns:
        np.array (n_segments): The distance from the main segment to each of the
            segments in the set.
        np.array (nd x n_segments): For each segment in the segment set, the
            point closest on the main segment
        np.array (nd x n_segments): For each segment in the segment set, the
            point closest on the secondary segment

    """
    start = np.squeeze(start)
    end = np.squeeze(end)

    nd = start.shape[0]
    ns = start_set.shape[1]

    d = np.zeros(ns)
    cp_set = np.zeros((nd, ns))
    cp = np.zeros((nd, ns))

    # Loop over all segments, compute the distance and closest point compared
    # to the main one.
    for i in range(ns):
        dl, cpi, cpj = dist_two_segments(start, end, start_set[:, i], end_set[:, i])
        d[i] = dl
        cp[:, i] = cpi
        cp_set[:, i] = cpj

    return d, cp, cp_set


# ------------------------------------------------------------------------------#


def dist_two_segments(s1_start, s1_end, s2_start, s2_end):
    """
    Compute the distance between two line segments.

    Also find the closest point on each of the two segments. In the case where
    the closest points are not unique (parallel lines), points somewhere along
    the segments are returned.

    The implementaion is based on http://geomalgorithms.com/a07-_distance.html
    (C++ code can be found somewhere on the page). Also confer that page for
    explanation of the algorithm.

    Implementation note:
        It should be possible to rewrite the algorithm to allow for one of (or
        both?) segments to be a set of segments, thus exploiting
        vectorization.

    Parameters:
        s1_start (np.array, size nd): Start point for the first segment
        s1_end (np.array, size nd): End point for the first segment
        s2_start (np.array, size nd): Start point for the second segment
        s2_end (np.array, size nd): End point for the second segment

    Returns:
        double: Minimum distance between the segments
        np.array (size nd): Closest point on the first segment
        np.array (size nd): Closest point on the second segment

    """

    # For the rest of the algorithm, see the webpage referred to above for details.
    d1 = s1_end - s1_start
    d2 = s2_end - s2_start
    d_starts = s1_start - s2_start

    dot_1_1 = d1.dot(d1)
    dot_1_2 = d1.dot(d2)
    dot_2_2 = d2.dot(d2)
    dot_1_starts = d1.dot(d_starts)
    dot_2_starts = d2.dot(d_starts)
    discr = dot_1_1 * dot_2_2 - dot_1_2 ** 2

    # Variable used to fine almost parallel lines. Sensitivity to this value has not been tested.
    SMALL_TOLERANCE = 1e-8 * np.minimum(dot_1_1, dot_2_2)

    # Sanity check
    assert discr >= -SMALL_TOLERANCE * dot_1_1 * dot_2_2

    sc = sN = sD = discr
    tc = tN = tD = discr

    if discr < SMALL_TOLERANCE:
        sN = 0
        sD = 1
        tN = dot_2_starts
        tD = dot_2_2
    else:
        sN = dot_1_2 * dot_2_starts - dot_2_2 * dot_1_starts
        tN = dot_1_1 * dot_2_starts - dot_1_2 * dot_1_starts
        if sN < 0.0:  # sc < 0 => the s=0 edge is visible
            sN = 0.0
            tN = dot_2_starts
            tD = dot_2_2

        elif sN > sD:  # sc > 1  => the s=1 edge is visible
            sN = sD
            tN = dot_1_2 + dot_2_starts
            tD = dot_2_2

    if tN < 0.0:  # tc < 0 => the t=0 edge is visible
        tN = 0.0
        # recompute sc for this edge
        if -dot_1_starts < 0.0:
            sN = 0.0
        elif -dot_1_starts > dot_1_1:
            sN = sD
        else:
            sN = -dot_1_starts
            sD = dot_1_1
    elif tN > tD:  # tc > 1  => the t=1 edge is visible
        tN = tD
        # recompute sc for this edge
        if (-dot_1_starts + dot_1_2) < 0.0:
            sN = 0
        elif (-dot_1_starts + dot_1_2) > dot_1_1:
            sN = sD
        else:
            sN = -dot_1_starts + dot_1_2
            sD = dot_1_1

    # finally do the division to get sc and tc
    if abs(sN) < SMALL_TOLERANCE:
        sc = 0.0
    else:
        sc = sN / sD
    if abs(tN) < SMALL_TOLERANCE:
        tc = 0.0
    else:
        tc = tN / tD

    # get the difference of the two closest points
    dist = d_starts + sc * d1 - tc * d2

    cp1 = s1_start + d1 * sc
    cp2 = s2_start + d2 * tc

    return np.sqrt(dist.dot(dist)), cp1, cp2


# -----------------------------------------------------------------------------


def dist_points_segments(p, start, end):
    """ Compute distances between points and line segments.

    Also return closest points on the segments.

    Parameters:
        p (np.array, ndxn): Individual points
        start (np.ndarray, nd x n_segments): Start points of segments.
        end (np.ndarray, nd x n_segments): End point of segments

    Returns:
        np.array, num_points x num_segments: Distances.
        np.array, num-points x num_segments x nd: Points on the segments
            closest to the individual points.

    """
    if start.size < 4:
        start = start.reshape((-1, 1))
        end = end.reshape((-1, 1))
    if p.size < 4:
        p = p.reshape((-1, 1))

    num_p = p.shape[1]
    num_l = start.shape[1]
    d = np.zeros((num_p, num_l))

    line = end - start
    lengths = np.sqrt(np.sum(line * line, axis=0))

    nd = p.shape[0]
    # Closest points
    cp = np.zeros((num_p, num_l, nd))

    # We need to compare all points to all segments.
    # The implemtation uses an outer for-loop, followed by an inner (almost)
    # vectorized part. Time consumption depends on the number of iterations in
    # the outer loop, so make a choice depending on whether there are more
    # segments or points. For borderline cases, more elaborate choices may be
    # better, but the current form should do for extreme cases.
    if num_p < num_l:
        for pi in range(num_p):
            # Project the vectors from start to point onto the line, and compute
            # relative length
            v = p[:, pi].reshape((-1, 1)) - start
            proj = np.sum(v * line, axis=0) / lengths ** 2

            # Projections with length less than zero have the closest point at
            # start
            less = np.ma.less_equal(proj, 0)
            d[pi, less] = dist_point_pointset(p[:, pi], start[:, less])
            cp[pi, less, :] = np.swapaxes(start[:, less], 1, 0)
            # Similarly, above one signifies closest to end
            above = np.ma.greater_equal(proj, 1)
            d[pi, above] = dist_point_pointset(p[:, pi], end[:, above])
            cp[pi, above, :] = np.swapaxes(end[:, above], 1, 0)

            # Points inbetween
            between = np.logical_not(np.logical_or(less, above).data)
            proj_p = start[:, between] + proj[between] * line[:, between]
            d[pi, between] = dist_point_pointset(p[:, pi], proj_p)
            cp[pi, between, :] = np.swapaxes(proj_p, 1, 0)
    else:
        for ei in range(num_l):
            # Project the vectors from start to point onto the line, and compute
            # relative length
            v = p - start[:, ei].reshape((-1, 1))
            proj = np.sum(v * line[:, ei].reshape((-1, 1)), axis=0) / lengths[ei] ** 2

            # Projections with length less than zero have the closest point at
            # start
            less = np.ma.less_equal(proj, 0)
            d[less, ei] = dist_point_pointset(start[:, ei], p[:, less])
            cp[less, ei, :] = start[:, ei]
            # Similarly, above one signifies closest to end
            above = np.ma.greater_equal(proj, 1)
            d[above, ei] = dist_point_pointset(end[:, ei], p[:, above])
            cp[above, ei, :] = end[:, ei]

            # Points inbetween
            between = np.logical_not(np.logical_or(less, above).data)
            proj_p = start[:, ei].reshape((-1, 1)) + proj[between] * line[
                :, ei
            ].reshape((-1, 1))
            for proj_i, bi in enumerate(np.where(between)[0]):
                d[bi, ei] = np.min(dist_point_pointset(proj_p[:, proj_i], p[:, bi]))
                cp[bi, ei, :] = proj_p[:, proj_i]

    return d, cp


# -----------------------------------------------------------------------------


def dist_point_pointset(p, pset, exponent=2):
    """
    Compute distance between a point and a set of points.

    Parameters:
        p (np.ndarray): Point from which distances will be computed
        pset (nd.array): Point cloud to which we compute distances
        exponent (double, optional): Exponent of the norm used. Defaults to 2.

    Return:
        np.ndarray: Array of distances.

    """

    # If p is 1D, do a reshape to facilitate broadcasting, but on a copy
    if p.ndim == 1:
        pt = p.reshape((-1, 1))
    else:
        pt = p

    # If the point cloud is a single point, it should still be a ndx1 array.
    if pset.size == 0:
        # In case of empty sets, return an empty zero
        return np.zeros(0)
    elif pset.size < 4:
        pset_copy = pset.reshape((-1, 1))
    else:
        # Call it a copy, even though it isn't
        pset_copy = pset

    return np.power(
        np.sum(np.power(np.abs(pt - pset_copy), exponent), axis=0), 1 / exponent
    )


# ----------------------------------------------------------------------------#


def dist_pointset(p, max_diag=False):
    """ Compute mutual distance between all points in a point set.

    Parameters:
        p (np.ndarray, 3xn): Points
        max_diag (boolean, defaults to False): If True, the diagonal values will
            are set to a large value, rather than 0.

    Returns:
        np.array (nxn): Distance between points.
    """
    if p.size > 3:
        n = p.shape[1]
    else:
        n = 1
        p = p.reshape((-1, 1))

    d = np.zeros((n, n))
    for i in range(n):
        d[i] = dist_point_pointset(p[:, i], p)

    if max_diag:
        for i in range(n):
            d[i, i] = 2 * np.max(d[i])

    return d


# ----------------------------------------------------------------------------#


def dist_points_polygon(p, poly, tol=1e-5):
    """ Compute distance from points to a polygon. Also find closest point on
    the polygon.

    The computation is split into two, the closest point can either be in the
    interior, or at the boundary of the point.

    Parameters:
        p (np.array, nd x n_pts): Points for which we will compute distances.
        poly (np.array, nd x n_vertexes): Vertexes of polygon. Edges are formed
            by subsequent points.

    Returns:
        np.array (n_pts): Distance from points to polygon
        np.array (nd x n_pts): For each point, the closest point on the
            polygon.
        np.array (n_pts, bool): True if the point is found in the interior,
            false if on a bounding segment.

    """

    if p.size < 4:
        p.reshape((-1, 1))

    num_p = p.shape[1]
    num_vert = poly.shape[1]
    nd = p.shape[0]

    # First translate the points so that the first plane is located at the origin
    center = np.mean(poly, axis=1).reshape((-1, 1))
    # Compute copies of polygon and point in new coordinate system
    orig_poly = poly
    poly = poly - center
    orig_p = p
    p = p - center

    # Obtain the rotation matrix that projects p1 to the xy-plane
    rot_p = project_plane_matrix(poly)
    irot = rot_p.transpose()
    poly_rot = rot_p.dot(poly)

    # Sanity check: The points should lay on a plane
    assert np.all(np.abs(poly_rot[2]) < tol)

    poly_xy = poly_rot[:2]

    # Make sure the xy-polygon is ccw.
    if not is_ccw_polygon(poly_xy):
        poly_1_xy = poly_xy[:, ::-1]

    # Rotate the point set, using the same coordinate system.
    p = rot_p.dot(p)

    in_poly = is_inside_polygon(poly_xy, p)

    # Distances
    d = np.zeros(num_p)
    # Closest points
    cp = np.zeros((nd, num_p))

    # For points that are located inside the extrusion of the polygon, the
    # distance is simply the z-coordinate
    d[in_poly] = np.abs(p[2, in_poly])

    # The corresponding closest points are found by inverse rotation of the
    # closest point on the polygon
    cp_inpoly = p[:, in_poly]
    if cp_inpoly.size == 3:
        cp_inpoly = cp_inpoly.reshape((-1, 1))
    cp_inpoly[2, :] = 0
    cp[:, in_poly] = center + irot.dot(cp_inpoly)

    if np.all(in_poly):
        return d, cp, in_poly

    # Next, points that are outside the extruded polygons. These will have
    # their closest point among one of the edges
    start = orig_poly
    end = orig_poly[:, (1 + np.arange(num_vert)) % num_vert]

    outside_poly = np.where(np.logical_not(in_poly))[0]

    d_outside, p_outside = dist_points_segments(orig_p[:, outside_poly], start, end)

    for i, pi in enumerate(outside_poly):
        mi = np.argmin(d_outside[i])
        d[pi] = d_outside[i, mi]
        cp[:, pi] = p_outside[i, mi, :]

    return d, cp, in_poly


# ----------------------------------------------------------------------------#


def dist_segments_polygon(start, end, poly, tol=1e-5):
    """ Compute the distance from line segments to a polygon.

    Parameters:
        start (np.array, nd x num_segments): One endpoint of segments
        end (np.array, nd x num_segments): Other endpoint of segments
        poly (np.array, nd x n_vert): Vertexes of polygon.

    Returns:
        np.ndarray, double: Distance from segment to polygon.
        np.array, nd x num_segments: Closest point.
    """
    if start.size < 4:
        start = start.reshape((-1, 1))
    if end.size < 4:
        end = end.reshape((-1, 1))

    num_p = start.shape[1]
    num_vert = poly.shape[1]
    nd = start.shape[0]

    d = np.zeros(num_p)
    cp = np.zeros((nd, num_p))

    # First translate the points so that the first plane is located at the origin
    center = np.mean(poly, axis=1).reshape((-1, 1))
    # Compute copies of polygon and point in new coordinate system
    orig_poly = poly
    orig_start = start
    orig_end = end

    poly = poly - center
    start = start - center
    end = end - center

    # Obtain the rotation matrix that projects p1 to the xy-plane
    rot_p = project_plane_matrix(poly)
    irot = rot_p.transpose()
    poly_rot = rot_p.dot(poly)

    # Sanity check: The points should lay on a plane
    assert np.all(np.abs(poly_rot[2]) < tol)

    poly_xy = poly_rot[:2]

    # Make sure the xy-polygon is ccw.
    if not is_ccw_polygon(poly_xy):
        poly_1_xy = poly_xy[:, ::-1]

    # Rotate the point set, using the same coordinate system.
    start = rot_p.dot(start)
    end = rot_p.dot(end)

    dz = end[2] - start[2]
    non_zero_incline = np.abs(dz) > tol

    # Parametrization along line of intersection point
    t = 0 * dz

    # Intersection point for segments with non-zero incline
    t[non_zero_incline] = -start[2, non_zero_incline] / dz[non_zero_incline]
    # Segments with z=0 along the segment
    zero_along_segment = np.logical_and(
        non_zero_incline, np.logical_and(t >= 0, t <= 1).astype(np.bool)
    )

    x0 = start + (end - start) * t
    # Check if zero point is inside the polygon
    inside = is_inside_polygon(poly_xy, x0[:2])
    crosses = np.logical_and(inside, zero_along_segment)

    # For points with zero incline, the z-coordinate should be zero for the
    # point to be inside
    segment_in_plane = np.logical_and(
        np.abs(start[2]) < tol, np.logical_not(non_zero_incline)
    )
    # Check if either start or endpoint is inside the polygon. This leaves the
    # option of the segment crossing the polygon within the plane, but this
    # will be handled by the crossing of segments below
    endpoint_in_polygon = np.logical_or(
        is_inside_polygon(poly_xy, start[:2]), is_inside_polygon(poly_xy, end[:2])
    )

    segment_in_polygon = np.logical_and(segment_in_plane, endpoint_in_polygon)

    intersects = np.logical_or(crosses, segment_in_polygon)

    x0[2, intersects] = 0
    cp[:, intersects] = center + irot.dot(x0[:, intersects])

    # Check if we're done, or if we should consider proximity to polygon
    # segments
    if np.all(intersects):
        # The distance is known to be zero, so no need to set it
        return d, cp

    not_found = np.where(np.logical_not(intersects))[0]

    # If we reach this, the minimum is not zero for all segments. The point
    # with minimum distance is then either 1) one endpoint of the segments
    # (point-polygon), 2) found as a segment-segment minimum (segment and
    # boundary of polygon), or 3) anywhere along the segment parallel with
    # polygon.
    poly = orig_poly
    start = orig_start
    end = orig_end

    # Distance from endpoints to
    d_start_poly, cp_s_p, s_in_poly = dist_points_polygon(start, poly)
    d_end_poly, cp_e_p, e_in_poly = dist_points_polygon(end, poly)

    # Loop over all segments that did not cross the polygon. The minimum is
    # found either by the endpoints, or as between two segments.
    for si in not_found:
        md = d_start_poly[si]
        cp_l = cp_s_p

        if d_end_poly[si] < md:
            md = d_end_poly
            cp_l = cp_e_p

        # Loop over polygon segments
        for poly_i in range(num_vert):
            ds, cp_s, _ = dist_two_segments(
                start[:, si],
                end[:, si],
                poly[:, poly_i],
                poly[:, (poly_i + 1) % num_vert],
            )
            if ds < md:
                md = ds
                cp_l = cp_s

        # By now, we have found the minimum
        d[si] = md
        cp[:, si] = cp_l.reshape((1, -1))

    return d, cp


# ----------------------------------------------------------------------------#

# ------------------------------------------------------------------------------#


def distance_point_segment(pt, start, end):
    """
    Compute the minimum distance between a point and a segment.

    Parameters:
        pt: the point
        start: a point representing one extreme of the segment.
        end: the second point representing the segment.
    Returns:
        distance: the minimum distance between the point and the segment.
        intersect: point of intersection
    """
    pt_shift = end - start
    length = np.dot(pt_shift, pt_shift)
    u = np.dot(pt - start, pt_shift) / (length if length != 0 else 1)
    dx = start + np.clip(u, 0.0, 1.0) * pt_shift - pt

    return np.sqrt(np.dot(dx, dx)), dx + pt


# ----------------------------------------------------------------------------#


def snap_points_to_segments(p_edges, edges, tol, p_to_snap=None):
    """
    Snap points in the proximity of lines to the lines.

    Note that if two vertices of two edges are close, they may effectively
    be co-located by the snapping. Thus, the modified point set may have
    duplicate coordinates.

    Parameters:
        p_edges (np.ndarray, nd x npt): Points defining endpoints of segments
        edges (np.ndarray, 2 x nedges): Connection between lines in p_edges.
            If edges.shape[0] > 2, the extra rows are ignored.
        tol (double): Tolerance for snapping, points that are closer will be
            snapped.
        p_to_snap (np.ndarray, nd x npt_snap, optional): The points to snap. If
            not provided, p_edges will be snapped, that is, the lines will be
            modified.

    Returns:
        np.ndarray (nd x n_pt_snap): A copy of p_to_snap (or p_edges) with
            modified coordinates.

    """

    if p_to_snap is None:
        p_to_snap = p_edges
        mod_edges = True
    else:
        mod_edges = False

    pn = p_to_snap.copy()

    nl = edges.shape[1]
    for ei in range(nl):

        # Find start and endpoint of this segment.
        # If we modify the edges themselves (mod_edges==True), we should use
        # the updated point coordinates. If not, we risk trouble for almost
        # coinciding vertexes.
        if mod_edges:
            p_start = pn[:, edges[0, ei]].reshape((-1, 1))
            p_end = pn[:, edges[1, ei]].reshape((-1, 1))
        else:
            p_start = p_edges[:, edges[0, ei]].reshape((-1, 1))
            p_end = p_edges[:, edges[1, ei]].reshape((-1, 1))
        d_segment, cp = dist_points_segments(pn, p_start, p_end)
        hit = np.argwhere(d_segment[:, 0] < tol)
        for i in hit:
            if mod_edges and (i == edges[0, ei] or i == edges[1, ei]):
                continue
            pn[:, i] = cp[i, 0, :].reshape((-1, 1))
    return pn


# ------------------------------------------------------------------------------#


def argsort_point_on_line(pts, tol=1e-5):
    """
    Return the indexes of the point according to their position on a line.

    Parameters:
        pts: the list of points
    Returns:
        argsort: the indexes of the points

    """
    if pts.shape[1] == 1:
        return np.array([0])
    assert is_collinear(pts, tol)

    nd, n_pts = pts.shape

    # Project into single coordinate
    rot = project_line_matrix(pts)
    p = rot.dot(pts)

    # Isolate the active coordinate

    mean = np.mean(p, axis=1)
    p -= mean.reshape((nd, 1))

    dx = p.max(axis=1) - p.min(axis=1)
    active_dim = np.where(dx > tol)[0]
    assert active_dim.size == 1, "Points should be co-linear"
    return np.argsort(p[active_dim])[0]


# ------------------------------------------------------------------------------#


def intersect_triangulations(p_1, p_2, t_1, t_2):
    """ Compute intersection of two triangle tessalation of a surface.

    The function will identify partly overlapping triangles between t_1 and
    t_2, and compute their common area. If parts of domain 1 or 2 is covered by
    one tessalation only, this will simply be ignored by the function.

    Implementation note: The function relies on the intersection algorithm in
    shapely.geometry.Polygon. It may be possible to extend the functionality
    to other cell shapes. This would require more general data structures, but
    should not be too much of an effort.

    Parameters:
        p_1 (np.array, 2 x n_p1): Points in first tessalation.
        p_2 (np.array, 2 x n_p2): Points in second tessalation.
        t_1 (np.array, 3 x n_tri_1): Triangles in first tessalation, referring
            to indices in p_1.
        t_2 (np.array, 3 x n_tri_1): Triangles in first tessalation, referring
            to indices in p_1.

    Returns:
        list of tuples: Each representing an overlap. The tuple contains index
            of the overlapping triangles in the first and second tessalation,
            and their common area.

    """
    n_1 = t_1.shape[1]
    n_2 = t_2.shape[1]
    t_1 = t_1.T
    t_2 = t_2.T

    # Find x and y coordinates of the triangles of first tessalation
    x_1 = p_1[0, t_1]
    y_1 = p_1[1, t_1]
    # Same with second tessalation
    x_2 = p_2[0, t_2]
    y_2 = p_2[1, t_2]

    intersections = []

    # Bounding box of each triangle for first and second tessalation
    min_x_1 = np.min(x_1, axis=1)
    max_x_1 = np.max(x_1, axis=1)
    min_y_1 = np.min(y_1, axis=1)
    max_y_1 = np.max(y_1, axis=1)

    min_x_2 = np.min(x_2, axis=1)
    max_x_2 = np.max(x_2, axis=1)
    min_y_2 = np.min(y_2, axis=1)
    max_y_2 = np.max(y_2, axis=1)

    # Represent the second tessalation using a Polygon from the shapely package
    poly_2 = [
        shapely_geometry.Polygon(
            [(x_2[j, 0], y_2[j, 0]), (x_2[j, 1], y_2[j, 1]), (x_2[j, 2], y_2[j, 2])]
        )
        for j in range(n_2)
    ]

    # Loop over all triangles in first tessalation, look for overlapping
    # members in second tessalation
    for i in range(n_1):
        # Polygon representation of the first triangle.
        poly_1 = shapely_geometry.Polygon(
            [(x_1[i, 0], y_1[i, 0]), (x_1[i, 1], y_1[i, 1]), (x_1[i, 2], y_1[i, 2])]
        )
        # Find triangles in the second tessalation that are outside the
        # bounding box of this triangle.
        right = np.squeeze(np.where(min_x_2 > max_x_1[i]))
        left = np.squeeze(np.where(max_x_2 < min_x_1[i]))
        above = np.squeeze(np.where(min_y_2 > max_y_1[i]))
        below = np.squeeze(np.where(max_y_2 < min_y_1[i]))

        # Candidates for intersection are only elements not outside
        outside = np.unique(np.hstack((right, left, above, below)))
        candidates = np.setdiff1d(np.arange(n_2), outside, assume_unique=True)

        # Loop over remaining candidates, call upon shapely to find
        # intersection
        for j in candidates:
            isect = poly_1.intersection(poly_2[j])
            if isinstance(isect, shapely_geometry.Polygon):
                intersections.append((i, j, isect.area))
    return intersections


# ------------------------------------------------------------------------------#


def constrain_lines_by_polygon(poly_pts, pts, edges):
    """
    Compute the intersections between a polygon (also not convex) and a set of lines.
    The computation is done line by line to avoid the splitting of edges caused by other
    edges. The implementation assume that the polygon and lines are on the plane (x, y).

    Parameters:
    poly_pts (np.ndarray, 3xn or 2xn): the points that define the polygon
    pts (np.ndarray, 3xn or 2xn): the points associated to the lines
    edges (np.ndarray, 2xn): for each column the id of the points for the line

    Returns:
    int_pts (np.ndarray, 2xn): the point associated to the lines after the intersection
    int_edges (np.ndarray, 2xn): for each column the id of the points for the line after the
        intersection. If the input edges have tags, stored in rows [2:], these will be
        preserved.

    """
    # it stores the points after the intersection
    int_pts = np.empty((2, 0))
    # define the polygon
    poly = shapely_geometry.Polygon(poly_pts[:2, :].T)

    # Kept edges
    edges_kept = []

    # we do the computation for each edge once at time, to avoid the splitting
    # caused by other edges.
    for ei, e in enumerate(edges.T):
        # define the line
        line = shapely_geometry.LineString([pts[:2, e[0]], pts[:2, e[1]]])
        # compute the intersections between the poligon and the current line
        int_lines = poly.intersection(line)
        # only line or multilines are considered, no points
        if type(int_lines) is shapely_geometry.LineString:
            # consider the case of single intersection by avoiding to consider
            # lines on the boundary of the polygon
            if not int_lines.touches(poly):
                int_pts = np.c_[int_pts, np.array(int_lines.xy)]
                edges_kept.append(ei)
        elif type(int_lines) is shapely_geometry.MultiLineString:
            # consider the case of multiple intersections by avoiding to consider
            # lines on the boundary of the polygon
            for int_line in int_lines:
                if not int_line.touches(poly):
                    int_pts = np.c_[int_pts, np.array(int_line.xy)]
                    edges_kept.append(ei)

    # define the list of edges
    int_edges = np.arange(int_pts.shape[1]).reshape((2, -1), order="F")

    # Also preserve tags, if any
    if len(edges_kept) > 0:
        edges_kept = np.array(edges_kept)
        edges_kept.sort()
        int_edges = np.vstack((int_edges, edges[2:, edges_kept]))
    else:
        # If no edges are kept, return an empty array with the right dimensions
        int_edges = np.empty((edges.shape[0], 0))

    return int_pts, int_edges


# ------------------------------------------------------------------------------#
def constrain_polygons_by_polyhedron(polygons, polyhedron, tol=1e-8):
    """ Constrain a seort of polygons in 3d to lie inside a, generally non-convex, polyhedron.

    Polygons not inside the polyhedron will be removed from descriptions.
    For non-convex polyhedra, polygons can be split in several parts.

    Parameters:
        polygons (np.ndarray, or list of arrays): Each element is a 3xnum_vertex
            array, describing the vertexes of a polygon.
        polyhedron (list of np.ndarray): Each element is a 3 x num_vertex array,
            describing the vertexes of the polygons that together form the
            polygon
        tol (double, optional): Tolerance used to compare points. Defaults to 1e-8.

    Returns:
        list of np.ndarray: Of polygons lying inside the polyhedra.

    """

    if isinstance(polygons, np.ndarray):
        polygons = [polygons]

    constrained_polygons = []

    # Loop over the polygons. For each, find the intersections with all
    # polygons on the side of the polyhedra.
    for poly in polygons:
        # Add this polygon to the list of constraining polygons. Put this first
        all_poly = [poly] + polyhedron

        # Find intersections
        coord, point_ind, is_bound, poly_ind = intersect_polygons_3d(all_poly)

        # Find indices of the intersection points for this polygon (the first one)
        isect_poly = point_ind[0]

        # If there are no intersection points, we just need to test if the
        # entire polygon is inside the polyhedral
        if isect_poly.size == 0:
            # Testing with a single point should suffice, but until the code
            # for in-polyhedron testing is more mature, we do some safeguarding:
            # Test for all points in the polygon, they should all be on the
            # inside or outside.
            inside = is_inside_polyhedron(polyhedron, poly)

            if inside.all():
                # Add the polygon to the constrained ones and continue
                constrained_polygons.append([poly])
                continue
            elif np.all(np.logical_not(inside)):
                # Do not add it.
                continue
            else:
                # This indicates that the inside_polyhedron test is bad
                assert False

        # At this point we know there are intersections between the polygon and
        # polyhedra

        # Find index of intersection points
        main_ind = point_ind[0]

        # Storage for intersection segments between the main polygon and the
        # polyhedron sides.
        segments = []

        # Loop over all sides of the polyhedral. Look for intersection points
        # that are both in main and the other
        for other in range(1, len(all_poly)):
            other_ip = point_ind[other]

            common = np.isin(other_ip, main_ind)
            if common.sum() < 2:
                # This is at most a point contact, no need to do anything
                continue
            # There is a real intersection between the segments. Add it.
            segments.append(other_ip[common])

        segments = np.array([i for i in segments]).T

        # Uniquify intersection coordinates, and update the segments
        unique_coords, _, ib = pp.utils.setmembership.unique_columns_tol(coord, tol=tol)
        unique_segments = ib[segments]

        # Represent the segments as a graph.
        graph = nx.Graph()
        for i in range(unique_segments.shape[1]):
            graph.add_edge(unique_segments[0, i], unique_segments[1, i])

        # If the segments are connected, which will always be the case if the
        # polyhedron is convex, the graph will have a single connected component.
        # If not, there will be multiple connected components. Find these, and
        # make a separate polygon for each.

        connected_poly = []

        # Loop over connected components
        for component in nx.connected_components(graph):
            # Extract subgraph of this cluster
            sg = graph.subgraph(component)
            # Make a list of edges of this subgraph
            el = []
            for e in sg.edges():
                el.append(e)
            el = np.array([e for e in el]).T

            # The vertexes of the polygon must be ordered. This is done slightly
            # differently depending on whether the polygon forms a closed circle
            # or not
            count = np.bincount(el.ravel())
            if np.any(count == 1):
                # There should be exactly two loose ends, if not, this is really
                # several polygons, and who knows how we ended up there.
                assert np.sum(count == 1) == 2
                sorted_pairs = pp.utils.sort_points.sort_point_pairs(
                    el, is_circular=False
                )
                inds = np.hstack((sorted_pairs[0], sorted_pairs[1, -1]))
            else:
                sorted_pairs = pp.utils.sort_points.sort_point_pairs(el)
                inds = sorted_pairs[0]
            inds = np.unique(el)
            # And there we are
            connected_poly.append(unique_coords[:, inds])

        constrained_polygons.append(connected_poly)

    return constrained_polygons


def bounding_box(pts, overlap=0):
    """ Obtain a bounding box for a point cloud.

    Parameters:
        pts: np.ndarray (nd x npt). Point cloud. nd should be 2 or 3
        overlap (double, defaults to 0): Extension of the bounding box outside
            the point cloud. Scaled with extent of the point cloud in the
            respective dimension.

    Returns:
        domain (dictionary): Containing keywords xmin, xmax, ymin, ymax, and
            possibly zmin and zmax (if nd == 3)

    """
    max_coord = pts.max(axis=1)
    min_coord = pts.min(axis=1)
    dx = max_coord - min_coord
    domain = {
        "xmin": min_coord[0] - dx[0] * overlap,
        "xmax": max_coord[0] + dx[0] * overlap,
        "ymin": min_coord[1] - dx[1] * overlap,
        "ymax": max_coord[1] + dx[1] * overlap,
    }

    if max_coord.size == 3:
        domain["zmin"] = min_coord[2] - dx[2] * overlap
        domain["zmax"] = max_coord[2] + dx[2] * overlap
    return domain


# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    import doctest

    doctest.testmod()
