"""
Various utility functions related to computational geometry.

Some functions (add_point, split_edges, ...?) are mainly aimed at finding
intersection between lines, with grid generation in mind, and should perhaps
be moved to a separate module.

"""
import time
import numpy as np
from sympy import geometry as geom

from porepy.utils import setmembership

#-----------------------------------------------------------------------------
#
# START OF FUNCTIONS RELATED TO SPLITTING OF INTERSECTING LINES IN 2D
#
#------------------------------------------------------------------------------#

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
    return pts

#------------------------------------------------------------------------------#

def __nrm(v):
    return np.sqrt(np.sum(v * v, axis=0))

#------------------------------------------------------------------------------#

def __dist(p1, p2):
    return __nrm(p1 - p2)

#------------------------------------------------------------------------------#

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

    tol = kwargs['tol']

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
    assert len(pt_ind) <= 2
    # Check for a single intersection point
    if len(pt_ind) < 2:
        pi = pt_ind[0]
        # Intersection at a point.
        if start == pi or end == pi:
            # Nothing happens, the intersection between the edges coincides
            # with a shared vertex for the edges
            new_line = 0
            split_type = 0
            return vertices, edges, new_line, split_type
        else:
            # We may need to split the edge (start, end) into two
            new_edges = np.array([[start, pi],
                                  [pi, end]])
            # ... however, the new edges may already exist in the set (this
            # apparently can happen for complex networks with several fractures
            # sharing a line).
            # Check if the new candidate edges already are defined in the set
            # of edges
            ismem, _ = setmembership.ismember_rows(new_edges, edges[:2])
            if any(ismem):
                new_edges = np.delete(new_edges,
                                      np.squeeze(np.argwhere(ismem)),
                                      axis=0)
            if new_edges.shape[0] == 1:
                new_edges = new_edges.reshape((-1, 1))

        if new_edges.size == 0:
            new_line = 0
            split_type = 1
            return vertices, edges, new_line, split_type

        # Add any tags to the new edge.
        if tags.size > 0:
            new_edges = np.vstack((new_edges,
                                   np.tile(tags[:, np.newaxis],
                                           new_edges.shape[1])))
        # Insert the new edge in the midle of the set of edges.
        edges = np.hstack((edges[:, :edge_ind_first], new_edges,
                           edges[:, edge_ind_first+1:]))
        # We have added as many new edges as there are columns in new_edges,
        # minus 1 (which was removed / ignored).
        new_line = new_edges.shape[1] - 1

        # Sanity check of new edge
        if np.any(np.diff(edges[:2], axis=0) == 0):
            raise ValueError('Have created a point edge')
        edge_copy = np.sort(edges[:2], axis=0)
        edge_unique, *new_2_old = setmembership.unique_columns_tol(edge_copy,
                                                                   tol=tol)
        if edge_unique.shape[1] < edges.shape[1]:
            raise ValueError('Have created the same edge twice')

        split_type = 2
        return vertices, edges, new_line, split_type
    else:
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
            new_edges = np.array([[start, i0, i1],
                                  [i0, i1, end]])
            if tags.size > 0:
                new_edges = np.vstack((new_edges,
                                       np.tile(tags[:, np.newaxis],
                                               new_edges.shape[1])))
            # Combine everything.
            edges = np.hstack((edges[:, :edge_ind[0]],
                               new_edges,
                               edges[:, edge_ind[0]+1:]))

            split_type = 4
        elif i0 == start and i1 == end:
            # We don't know if i0 is closest to the start or end of edges[:,
            # edges_ind[1]]. Find the nearest.
            if __dist(np.squeeze(vertices[:, i0]),
                      vertices[:, edges[0, edge_ind[1]]]) < \
               __dist(np.squeeze(vertices[:, i1]),
                      vertices[:, edges[0, edge_ind[1]]]):
                other_start = edges[0, edge_ind[1]]
                other_end = edges[1, edge_ind[1]]
            else:
                other_start = edges[1, edge_ind[1]]
                other_end = edges[0, edge_ind[1]]
            # New segments (i0, i1) is identical to the old edge_ind[0]
            new_edges = np.array([[other_start, i0, i1],
                                  [i0, i1, other_end]])
            # For some reason we sometimes create point-edges here (start and
            # end are identical). Delete these if necessary
            del_ind = np.squeeze(np.where(np.diff(new_edges, axis=0)[0] == 0))
            new_edges = np.delete(new_edges, del_ind, axis=1)
            if tags.size > 0:
                new_edges = np.vstack((new_edges,
                                       np.tile(tags[:, np.newaxis],
                                               new_edges.shape[1])))
            # Combine everything.
            edges = np.hstack((edges[:, :edge_ind[1]],
                               new_edges,
                               edges[:, (edge_ind[1]+1):]))
            # Delete the second segment. This is most easily handled after
            # edges is expanded, to avoid accounting for changing edge indices.
            edges = np.delete(edges, edge_ind[0], axis=1)

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
                    edges = np.delete(edges, edge_ind[1], axis=1)
                else:
                    edges[0, edge_ind[1]] = start
            elif edges[1, edge_ind[1]] == i1:
                if edges[0, edge_ind[1]] == start:
                    edges = np.delete(edges, edge_ind[1], axis=1)
                else:
                    edges[1, edge_ind[1]] = start
            else:
                raise ValueError('This should not happen')

            new_edges = np.array([[start, i1],
                                  [i1, end]])
            if tags.size > 0:
                new_edges = np.vstack((new_edges,
                                       np.tile(tags[:, np.newaxis],
                                               new_edges.shape[1])))

            edges = np.hstack((edges[:, :edge_ind[0]],
                               new_edges,
                               edges[:, (edge_ind[0]+1):]))
            split_type = 6

        elif i0 != start and i1 == end:
            # Analogous configuration as the one above, but with i0 replaced by
            # i1 and start by end.
            if edges[0, edge_ind[1]] == i0:
                if edges[1, edge_ind[1]] == end:
                    edges = np.delete(edges, edge_ind[1], axis=1)
                else:
                    edges[0, edge_ind[1]] = end
            elif edges[1, edge_ind[1]] == i0:
                if edges[0, edge_ind[1]] == end:
                    edges = np.delete(edges, edge_ind[1], axis=1)
                else:
                    edges[1, edge_ind[1]] = end
            else:
                raise ValueError('This should not happen')
            new_edges = np.array([[start, i0],
                                  [i0, end]])
            if tags.size > 0:
                new_edges = np.vstack((new_edges,
                                       np.tile(tags[:, np.newaxis],
                                               new_edges.shape[1])))

            edges = np.hstack((edges[:, :edge_ind[0]],
                               new_edges,
                               edges[:, (edge_ind[0]+1):]))
            split_type = 7

        else:
            raise ValueError('How did it come to this')

        # Check validity of the new edge configuration
        if np.any(np.diff(edges[:2], axis=0) == 0):
            raise ValueError('Have created a point edge')

        # We may have created an edge that already existed in the grid. Remove
        # this by uniquifying the edges.
        # Hopefully, we do not mess up the edges here.
        edges_copy = np.sort(edges[:2], axis=0)
        edges_unique, new_2_old, _ \
                = setmembership.unique_columns_tol(edges_copy, tol=tol)
        # Refer to unique edges if necessary
        if edges_unique.shape[1] < edges.shape[1]:
            # Copy tags
            edges = np.vstack((edges_unique, edges[2, new_2_old]))
            # Also signify that we have carried out this operation.
            split_type = [split_type, 8]

        # Number of new lines created
        new_line = edges.shape[1] - orig_num_edges

        return vertices, edges, new_line, split_type

#------------------------------------------------------**kwargs------------------------#

def _add_point(vertices, pt, tol=1e-3, **kwargs):
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
    if 'tol' not in kwargs:
        kwargs['tol'] = tol

    nd = vertices.shape[0]
    # Before comparing coordinates, snap both existing and new point to the
    # underlying grid
    vertices = snap_to_grid(vertices, **kwargs)
    pt = snap_to_grid(pt, **kwargs)

    new_pt = np.empty((nd, 0))
    ind = []
    # Distance
    for i in range(pt.shape[-1]):
        dist = __dist(pt[:, i].reshape((-1, 1)), vertices)
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


#-----------------------------------------------------------------------------#

def remove_edge_crossings(vertices, edges, tol=1e-3, verbose=0, **kwargs):
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
    if verbose > 1:
        start_time = time.time()
        num_edges_orig = edges.shape[1]
        print('  Find intersections between ' + str(num_edges_orig) + ' edges')

    nd = vertices.shape[0]

    # Only 2D is considered. 3D should not be too dificult, but it is not
    # clear how relevant it is
    if nd != 2:
        raise NotImplementedError('Only 2D so far')

    edge_counter = 0

    # Add tolerance to kwargs, this is later passed to split_edges, and further
    # on.
    kwargs['tol'] = tol

    vertices = snap_to_grid(vertices, **kwargs)

    # Field used for debugging of edge splits. To see the meaning of the values
    # of each split, look in the source code of split_edges.
    split_type = []

    # Loop over all edges, search for intersections. The number of edges can
    #  change due to splitting.
    while edge_counter < edges.shape[1]:
        # The direct test of whether two edges intersect is somewhat
        # expensive, and it is hard to vectorize this part. Therefore,
        # we first identify edges which crosses the extention of this edge (
        # intersection of line and line segment). We then go on to test for
        # real intersections.

        # Find start and stop coordinates for all edges
        start_x = vertices[0, edges[0]]
        start_y = vertices[1, edges[0]]
        end_x = vertices[0, edges[1]]
        end_y = vertices[1, edges[1]]

        a = end_y - start_y
        b = -(end_x - start_x)

        # Midpoint of this edge
        xm = (start_x[edge_counter] + end_x[edge_counter]) / 2
        ym = (start_y[edge_counter] + end_y[edge_counter]) / 2

        # For all lines, find which side of line i it's two endpoints are.
        # If c1 and c2 have different signs, they will be on different sides
        # of line i. See
        #
        # http://stackoverflow.com/questions/385305/efficient-maths-algorithm-to-calculate-intersections
        #
        # answer by PolyThinker and comments by Jason S, for more information.
        c1 = a[edge_counter] * (start_x - xm) \
             + b[edge_counter] * (start_y - ym)
        c2 = a[edge_counter] * (end_x - xm) + b[edge_counter] * (end_y - ym)

        tol_scaled = tol * max(1, np.max([np.sqrt(np.abs(c1)),
                                          np.sqrt(np.abs(c2))]))

        # We check for three cases
        # 1) Lines crossing
        lines_cross = np.sign(c1) != np.sign(c2)
        # 2) Lines parallel
        parallel_lines = np.logical_and(np.abs(c1) < tol_scaled,
                                        np.abs(c2) < tol_scaled)
        # 3) One line look to end on the other
        lines_almost_cross = np.logical_or(np.abs(c1) < tol_scaled,
                                           np.abs(c2) < tol_scaled)
        # Any of the three above deserves a closer look
        line_intersections = np.logical_or(np.logical_or(parallel_lines,
                                                         lines_cross),
                                           lines_almost_cross)

        # Find elements which may intersect.
        intersections = np.argwhere(line_intersections)
        # np.argwhere returns an array of dimensions (1, dim), so we reduce
        # this to truly 1D, or simply continue with the next edge if there
        # are no candidate edges
        if intersections.size > 0:
            intersections = intersections.ravel(0)
        else:
            # There are no candidates for intersection
            edge_counter += 1
            continue

        if verbose > 2:
            print('    ------')
            print('    Splitting edge no ' + str(edge_counter) + '. Vertices:')
            print('    ' + str(vertices[:, edges[0, edge_counter]]))
            print('    ' + str(vertices[:, edges[1, edge_counter]]))

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

            if verbose > 2:
                print('    Intersection with edge ' + str(intsect)
                      + '. Vertices:')
                print('    ' + str(vertices[:, edges[0, intsect]]))
                print('    ' + str(vertices[:, edges[1, intsect]]))

            # Check if this point intersects
            new_pt = lines_intersect(vertices[:, edges[0, edge_counter]],
                                     vertices[:, edges[1, edge_counter]],
                                     vertices[:, edges[0, intsect]],
                                     vertices[:, edges[1, intsect]],
                                     tol=tol)

            def __min_dist(p):
                md = np.inf
                for pi in [edges[0, edge_counter],
                           edges[1, edge_counter],
                           edges[0, intsect], edges[1, intsect]]:
                    md = min(md, __dist(np.squeeze(p), vertices[:, pi]))
                return md

            orig_vertex_num = vertices.shape[1]
            orig_edge_num = edges.shape[1]

            if new_pt is not None:
                new_pt = snap_to_grid(new_pt, tol=tol)
                # The case of segment intersections need special treatment.
                if new_pt.shape[-1] == 1:
                    if verbose > 2:
                        print('    Found intersection: (' + str(new_pt[0]) +
                              ', '  + str(new_pt[1]))

                    # Split edge edge_counter (outer loop), unless the
                    # intersection hits an existing point (in practices this
                    # means the intersection runs through an endpoint of the
                    # edge in an L-type configuration, in which case no new point
                    # is needed)
                    md = __min_dist(new_pt)
                    vertices, edges, split_outer_edge,\
                            split = _split_edge(vertices, edges, edge_counter,
                                                new_pt, **kwargs)
                    split_type.append(split)
                    if verbose > 2 and split_outer_edge > 0 and \
                       vertices.shape[1] > orig_vertex_num:
                        print('      Introduced new point. Min length of edges:'
                              + str(md))
                        if md < tol:
                            import pdb
                            pdb.set_trace()

                    if edges.shape[1] > orig_edge_num + split_outer_edge:
                        raise ValueError('Have created edge without bookkeeping')
                    # If the outer edge (represented by edge_counter) was split,
                    # e.g. inserted into the list of edges we need to increase the
                    # index of the inner edge
                    intsect += split_outer_edge

                    # Possibly split the inner edge
                    vertices, edges, split_inner_edge, \
                            split = _split_edge(vertices, edges, intsect,
                                                new_pt, **kwargs)
                    if edges.shape[1] > \
                       orig_edge_num + split_inner_edge + split_outer_edge:
                        raise ValueError('Have created edge without bookkeeping')

                    split_type.append(split)
                    if verbose > 2 and split_inner_edge > 0 and \
                        vertices.shape[1] > orig_vertex_num:
                        print('      Introduced new point. Min length of edges:'
                              + str(md))
                        if md < tol:
                            import pdb
                            pdb.set_trace()
                    intersections += split_outer_edge + split_inner_edge
                else:
                    # We have found an intersection along a line segment
                    if verbose > 2:
                        print('    Found two intersections: ('
                              + str(new_pt[0, 0]) + ', '  + str(new_pt[1, 0]) +
                              'and ' + str(new_pt[0, 1]) + ', '  +
                              str(new_pt[1, 1]))

                    vertices, edges, splits,\
                            s_type = _split_edge(vertices, edges,
                                                 [edge_counter, intsect],
                                                 new_pt, **kwargs)
                    split_type.append(s_type)
                    intersections += splits
                    if verbose > 2 and (splits[0] > 0 or splits[1] > 0):
                        print('    Introduced new point')

            # Sanity checks - turned out to be useful for debugging.
            if np.any(np.diff(edges[:2], axis=0) == 0):
                if verbose > 3:
                    import pdb
                    pdb.set_trace()
                raise ValueError('Have somehow created a point edge')
            if intersections.max() > edges.shape[1]:
                raise ValueError('Intersection pointer outside edge array')

            # We're done with this candidate edge. Increase index of inner loop
            int_counter += 1

        # We're done with all intersections of this loop. increase index of
        # outer loop
        edge_counter += 1

    if verbose > 1:
        print('  Edge intersection removal complete. Elapsed time ' +\
              str(time.time() - start_time))
        print('  Introduced ' + str(edges.shape[1] - num_edges_orig) + \
              ' new edges')

    return vertices, edges

#----------------------------------------------------------
#
# END OF FUNCTIONS RELATED TO SPLITTING OF INTERSECTING LINES IN 2D
#
#-----------------------------------------------------------

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
        value += (p_1[i+1] + p_1[i]) * (p_0[i+1] - p_0[i])
    return value < 0

#----------------------------------------------------------

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
    cross_product = (p2[0] - p1[0]) * (p3[1] - p1[1])\
                   -(p2[1] - p1[1]) * (p3[0] - p1[0])

    # Should there be a scaling of the tolerance relative to the distance
    # between the points?

    if np.abs(cross_product) <= tol:
        return default

    return cross_product > 0

#-----------------------------------------------------------------------------

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

    poly_size = poly.shape[1]

    inside = np.ones(pt.shape[1], dtype=np.bool)
    for i in range(pt.shape[1]):
        for j in range(poly.shape[1]):
            if not is_ccw_polyline(poly[:, j], poly[:, (j+1) % poly_size],
                                   pt[:, i], tol=tol, default=default):
                inside[i] = False
                # No need to check the remaining segments of the polygon.
                break
    return inside


#-----------------------------------------------------------------------------

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

    return np.power(np.sum(np.power(np.abs(pt - pset), exponent),
                           axis=0), 1/exponent)


#------------------------------------------------------------------------------#

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

    discr = d_1[0] *(-d_2[1]) - d_1[1] * (-d_2[0])

    if np.abs(discr) < tol:
        # The lines are parallel, and will only cross if they are also colinear

        # Cross product between line 1 and line between start points on line
        start_cross_line = d_s[0] * d_1[1] - d_s[1] * d_1[0]
        if np.abs(start_cross_line) < tol:
            # The lines are co-linear

            # Write l1 on the form start_1 + t * d_1, find the parameter value
            # needed for equality with start_2 and end_2
            if np.abs(d_1[0]) > tol:
                t_start_2 = (start_2[0] - start_1[0])/d_1[0]
                t_end_2 = (end_2[0] - start_1[0])/d_1[0]
            elif np.abs(d_1[1]) > tol:
                t_start_2 = (start_2[1] - start_1[1])/d_1[1]
                t_end_2 = (end_2[1] - start_1[1])/d_1[1]
            else:
                # d_1 is zero
                raise ValueError('Start and endpoint of line should be\
                                 different')
            if t_start_2 < 0 and t_end_2 < 0:
                return None
            elif t_start_2 > 1 and t_end_2 > 1:
                return None
            # We have an overlap, find its parameter values
            t_min = max(min(t_start_2, t_end_2), 0)
            t_max = min(max(t_start_2, t_end_2), 1)

            if t_max - t_min < tol:
                # It seems this can only happen if they are also equal to 0 or
                # 1, that is, the lines share a single point
                p_1 = start_1 + d_1 * t_min
                return p_1.reshape((-1, 1))

            p_1 = start_1 + d_1 * t_min
            p_2 = start_1 + d_1 * t_max
            return np.array([[p_1[0], p_2[0]], [p_1[1], p_2[1]]])

        else:
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
        if t_1 >= -tol and t_1 <= (1 + tol) and \
           t_2 >= -tol and t_2 <= (1 + tol):
            return np.array([[isect_1[0]], [isect_1[1]]])

        return None

#------------------------------------------------------------------------------#

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
    start_1 = np.asarray(start_1).astype(np.float)
    end_1 = np.asarray(end_1).astype(np.float)
    start_2 = np.asarray(start_2).astype(np.float)
    end_2 = np.asarray(end_2).astype(np.float)

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
    discr = deltas_1[in_discr[0]] * deltas_2[in_discr[1]]\
            - deltas_1[in_discr[1]] * deltas_2[in_discr[0]]

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

        dstart_x_delta_x = diff_start[1] * deltas_1[2] -\
                           diff_start[2] * deltas_1[1]
        if np.abs(dstart_x_delta_x) > tol:
            return None
        dstart_x_delta_y = diff_start[2] * deltas_1[0] -\
                           diff_start[0] * deltas_1[2]
        if np.abs(dstart_x_delta_y) > tol:
            return None
        dstart_x_delta_z = diff_start[0] * deltas_1[1] -\
                           diff_start[1] * deltas_1[0]
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

        discr = deltas_1[in_discr[0]] * (-deltas_2[in_discr[1]]) -\
                deltas_1[in_discr[1]] * (-deltas_2[in_discr[0]])
        t_1 = ((start_2[in_discr[0]] - start_1[in_discr[0]]) \
               * (-deltas_2[in_discr[1]]) - \
               (start_2[in_discr[1]] - start_1[in_discr[1]]) \
               * (-deltas_2[in_discr[0]]))/discr

        t_2 = (deltas_1[in_discr[0]] * (start_2[in_discr[1]] -
                                        start_1[in_discr[1]]) - \
               deltas_1[in_discr[1]] * (start_2[in_discr[0]] -
                                        start_1[in_discr[0]])) / discr

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

#----------------------------------------------------------------------------#

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
        return np.array(list([i.args for i in p]), dtype='float').transpose()
    else:
        return np.array(list(p.args), dtype='float').reshape((-1, 1))

def _to3D(p):
    # Add a third dimension
    return np.vstack((p, np.zeros(p.shape[1])))

#-------------------------------------------------------------------


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
        p_1_2 = poly_1[:, ind_1[i+1]]

        for j in range(l_2):
            p_2_1 = poly_2[:, ind_2[j]]
            p_2_2 = poly_2[:, ind_2[j+1]]
            isect_loc = segments_intersect_3d(p_1_1, p_1_2, p_2_1, p_2_2)
            if isect_loc is not None:
                isect.append([i, j, isect_loc])

    return isect

#----------------------------------------------------------

def polygon_segment_intersect(poly_1, poly_2, tol=1e-8):
    """
    Find intersections between polygons embeded in 3D.

    The intersections are defined as between the interior of the first polygon
    and the boundary of the second.

    TODO:
        1) Also cover case where the one polygon ends in the plane of the other.

    Parameters:
        poly_1 (np.ndarray, 3xn1): Vertexes of polygon, assumed ordered as cw or
            ccw.
        poly_2 (np.ndarray, 3xn2): Vertexes of second polygon, assumed ordered
            as cw or ccw.
        tol (double, optional): Tolerance for when two points are equal.
            Defaults to 1e-8.

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
    poly_1_xy = rot_p_1.dot(poly_1)

    # Sanity check: The points should lay on a plane
    assert np.all(np.abs(poly_1_xy[2]) < tol)
    # Drop the z-coordinate
    poly_1_xy = poly_1_xy[:2]

    # Rotate the second polygon with the same rotation matrix
    poly_2_rot = rot_p_1.dot(poly_2)

    # If the rotation of whole second point cloud lies on the same side of z=0,
    # there are no intersections.
    if poly_2_rot[2].min() > 0:
        return None
    elif poly_2_rot[2].max() < 0:
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
        ind = np.append(np.arange(num_p2), np.zeros(1)).astype('int')

        isect = np.empty((3, 0))

        for i in range(num_p2):

            # Coordinates of this segment
            pt_1 = poly_2_rot[:, ind[i]]
            pt_2 = poly_2_rot[:, ind[i+1]]

            # Check if segment crosses z=0 in the rotated coordinates
            if max(pt_1[2], pt_2[2]) < 0 or min(pt_1[2], pt_2[2]) > 0:
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
                assert t >= 0 and t <= 1

                # x and y-coordinate for z=0
                x0 = pt_1[0] + dx * t
                y0 = pt_1[1] + dy * t
                # Representation as point
                p_00 = np.array([x0, y0]).reshape((-1, 1))

                # Check if the first polygon encloses the point. If the
                # intersection is on the border, this will not be detected.

                if is_inside_polygon(poly_1_xy, p_00, tol=tol):
                    # Back to physical coordinates by 1) expand to 3D, 2)
                    # inverse rotation, 3) translate to original coordinate.
                    isect = np.hstack((isect, irot.dot(_to3D(p_00)) +
                                       center_1))

        if isect.shape[1] == 0:
            isect = None

        return isect


def is_planar(pts, normal=None):
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
        normal = normal / np.linalg.norm(normal)

    check = np.array([np.isclose(np.dot(normal, pts[:, 0] - p), 0) \
                      for p in pts[:, 1:].T], dtype=np.bool)
    return np.all(check)

#------------------------------------------------------------------------------#

def project_plane_matrix(pts, normal=None):
    """ Project the points on a plane using local coordinates.

    The projected points are computed by a dot product.
    example: np.dot( R, pts )

    Parameters:
    pts (np.ndarray, 3xn): the points.
    normal: (optional) the normal of the plane, otherwise three points are
        required.

    Returns:
    np.ndarray, 3x3, projection matrix.

    """

    if normal is None:
        normal = compute_normal(pts)
    else:
        normal = normal / np.linalg.norm(normal)

    reference = np.array([0., 0., 1.])
    angle = np.arccos(np.dot(normal, reference))
    vect = np.cross(normal, reference)
    return rot(angle, vect)

#------------------------------------------------------------------------------#

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
    W = np.array( [[       0., -vect[2],  vect[1]],
                   [  vect[2],       0., -vect[0]],
                   [ -vect[1],  vect[0],       0. ]])
    return np.identity(3) + np.sin(a)*W + \
           (1.-np.cos(a)) * np.linalg.matrix_power(W, 2)

#------------------------------------------------------------------------------#

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

#------------------------------------------------------------------------------#

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

#------------------------------------------------------------------------------#

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

#------------------------------------------------------------------------------#

def compute_normals_1d(pts):
    t = compute_tangent(pts)
    n = np.array([t[1], -t[0], 0]) / np.sqrt(t[0]**2+t[1]**2)
    return np.r_['1,2,0', n, np.dot(rot(np.pi/2., t), n)]

#------------------------------------------------------------------------------#

def compute_tangent(pts):
    """ Compute a tangent of a set of points.

    The algorithm assume that the points lie on a plane.

    Parameters:
    pts: np.ndarray, 3xn, the points.

    Returns:
    tangent: np.array, 1x3, the tangent.

    """

    tangent = pts[:, 0] - np.mean(pts, axis=1)
    assert not np.allclose(tangent, np.zeros(3))
    return tangent / np.linalg.norm(tangent)

#------------------------------------------------------------------------------#

def is_collinear(pts, tol=1e-5):
    """ Check if the points lie on a line.

    Parameters:
        pts (np.ndarray, 3xn): the points.
        tol (double, optional): Tolerance used in comparison. Defaults to 1e-5.

    Returns:
        boolean, True if the points lie on a line.

    """

    assert pts.shape[1] > 1
    if pts.shape[1] == 2:
        return True

    pt0 = pts[:, 0]
    pt1 = pts[:, 1]

    coll = np.array([np.linalg.norm(np.cross(p - pt0, pt1 - pt0)) \
             for p in pts[:, 1:-1].T])
    return np.allclose(coll, np.zeros(coll.size), tol)

#------------------------------------------------------------------------------#

def map_grid(g):
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
        return cell_centers, face_normals, face_centers, R,\
               np.ones(3, dtype=bool), nodes

    if g.dim == 1 or g.dim == 2:
        v = compute_normal(g.nodes) if g.dim == 2 else compute_tangent(g.nodes)
        R = project_plane_matrix(g.nodes, v)
        face_centers = np.dot(R, face_centers)
        dim = np.logical_not(np.isclose(np.sum(np.abs(face_centers.T-
                                                      face_centers[:, 0]),
                                               axis=0), 0))
        assert g.dim == np.sum(dim)
        face_centers = face_centers[dim, :]
        cell_centers = np.dot(R, cell_centers)[dim, :]
        face_normals = np.dot(R, face_normals)[dim, :]
        nodes = np.dot(R, nodes)[dim, :]

    return cell_centers, face_normals, face_centers, R, dim, nodes

#------------------------------------------------------------------------------#

def distance_segment_segment(s1_start, s1_end, s2_start, s2_end):
    """
    Compute the distance between two line segments.

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
    assert discr >= 0

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
        if sN < 0.0:        # sc < 0 => the s=0 edge is visible
            sN = 0.0
            tN = dot_2_starts
            tD = dot_2_2

        elif sN > sD:   # sc > 1  => the s=1 edge is visible
            sN = sD
            tN = dot_1_2 + dot_2_starts
            tD = dot_2_2

    if tN < 0.0:            # tc < 0 => the t=0 edge is visible
        tN = 0.0
        # recompute sc for this edge
        if -dot_1_starts < 0.0:
            sN = 0.0
        elif -dot_1_starts > dot_1_1:
            sN = sD
        else:
            sN = -dot_1_starts
            sD = dot_1_1
    elif tN > tD:       # tc > 1  => the t=1 edge is visible
        tN = tD
        # recompute sc for this edge
        if (-dot_1_starts + dot_1_2) < 0.0:
            sN = 0
        elif (-dot_1_starts+ dot_1_2) > dot_1_1:
            sN = sD
        else:
            sN = (-dot_1_starts + dot_1_2)
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
    return np.sqrt(dist.dot(dist))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
