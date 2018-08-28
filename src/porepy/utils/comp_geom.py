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

import shapely.geometry as shapely_geometry
import shapely.speedups as shapely_speedups

from porepy.utils import setmembership


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
    assert np.all(
        np.diff(edges[:2], axis=0) != 0
    ), "Found point edge before" "removal of intersections"

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
        xm = (start_x[edge_counter] + end_x[edge_counter]) / 2.
        ym = (start_y[edge_counter] + end_y[edge_counter]) / 2.

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
    if np.allclose(vect, [0., 0., 0.]):
        return np.identity(3)
    vect = vect / np.linalg.norm(vect)

    # Prioritize readability over PEP0008 whitespaces.
    # pylint: disable=bad-whitespace
    W = np.array(
        [[0., -vect[2], vect[1]], [vect[2], 0., -vect[0]], [-vect[1], vect[0], 0.]]
    )
    return (
        np.identity(3) + np.sin(a) * W + (1. - np.cos(a)) * np.linalg.matrix_power(W, 2)
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
    return np.r_["1,2,0", n, np.dot(rot(np.pi / 2., t), n)]


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

    """
    start = np.squeeze(start)
    end = np.squeeze(end)

    nd = start.shape[0]
    ns = start_set.shape[1]

    d = np.zeros(ns)
    cp_set = np.zeros((nd, ns))
    cp = np.zeros((nd, ns))

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

    # Variable used to fine almost parallel lines. Sensitivity to this value has not been tested.
    SMALL_TOLERANCE = 1e-6

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
    dx = start + np.clip(u, 0., 1.) * pt_shift - pt

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
