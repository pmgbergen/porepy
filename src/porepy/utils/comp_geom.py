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


def remove_edge_crossings(p, e, tol=1e-4):
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
            # Values 0.3 and 0.7 are quite random here.
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
                start_main, end_main, pt(start_other, ri), pt(end_other, ri), tol
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

    In addition to intersection points, the function also decides:
        1) Whether intersection points lie in the interior, on a segment or a vertex.
           If segment or vertex, the index of the segment or vertex is returned.
        2) Whether a pair of intersection points lie on the same boundary segment of a
           polygon, that is, if the polygon has a T or L-type intersection with another
           paolygon.

    Assumptions:
        * All polygons are convex. Non-convex polygons will simply be treated
          in a wrong way.
        * No polygon contains three points on a line, that is, an angle of pi. This can
            be included, possibly by temporarily stripping the hanging node from the
            polygon definition.
        * If two polygons meet in a vertex, this is not considered an intersection.
        * If two polygons lie in the same plane, intersection types (vertex, segment,
            interior) are not classified. This will be clear from the returned values.
            Inclusion of this should be possible, but it has not been a priority.
        * Contact between polygons in a single point may not be accurately calculated.

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
        list of list of tuples: For each polygon, for all intersection points (same
            order as the second return value), a 2-tuple, where the first value
            gives an index, the second is a Boolean, True if the intersection is on a
            segment, False if vertex. The index identifies the vertex, or the first
            vertex of the segment. If the intersection is in the interior of a polygon,
            the tuple is replaced by an empty list.

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
        if isinstance(v, np.ndarray):
            sgn = np.sign(v)
            sgn[np.abs(v) < tol] = 0
            return sgn
        else:
            if abs(v) < tol:
                return 0
            elif v < 0:
                return -1
            else:
                return 1

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
    # Storage for which segment or vertex of a polygon is intersected
    segment_vertex_intersection = np.empty_like(isect_pt)

    # Initialization
    for i in range(isect_pt.size):
        isect_pt[i] = []
        is_bound_isect[i] = []
        segment_vertex_intersection[i] = []

    # Array for storing the newly found points
    new_pt = []
    new_pt_ind = 0

    # Index of the main fractures, to which the other ones will be compared.
    start_inds = np.unique(pairs[0])

    # Store index of pairs of intersecting polygons
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

            # We know that the polygons at least are very close to intersecting each
            # others planes. There are four options, differing in whether the vertexes
            # are in the plane of the other polygon or not:
            #   1) The polygon has no vertex in the other plane. Intersection is found
            #      by computing interseciton between polygon segments and the other
            #      plane.
            #   2) The polygon has one vertex in the other plane. This is one intersection
            #      point. The other one should be on a segment, that is, the polygon
            #      should have points on both sides of the plane.
            #   3) The polygon has two vertexes in the other plane. These will be the
            #      intersection points. The remaining vertexes should be on the same
            #      side of the plane.
            #   4) All vertexes lie in the plane. The intersection points will be found
            #      by what is essentially a 2d algorithm. Note that the current
            #      implementation if this case is a bit rudimentary.
            #
            # NOTE: This part of the code only considers intersection between polygon
            # and plane. The analysis if whether the interseciton points are within
            # each polygon is done below.
            #
            # We first compute the interseciton of the other polygon with the plane of
            # the main one. The reverse operation is found below.
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
                # First indices, next is whether this refers to segment. False means vertex.
                seg_vert_other_0 = (sign_change_main[0], True)
                seg_vert_other_1 = (sign_change_main[1], True)

            elif np.sum(dot_prod_from_main[:-1] == 0) == 1:
                # The first and last element represent the same point, thus include
                # only one of them when counting the number of points in the plane
                # of the other fracture.
                hit = np.where(dot_prod_from_main[:-1] == 0)[0]
                other_intersects_main_0 = other_p_expanded[:, hit[0]]
                sign_change_full = np.where(np.abs(np.diff(dot_prod_from_main)) > 1)[0]
                if sign_change_full.size == 0:
                    # This corresponds to a point contact between one polygon and the
                    # other (at least other plane, perhaps also other polygon)
                    # Simply ignore this for now.S
                    continue
                other_intersects_main_1 = intersection(
                    other_p_expanded[:, sign_change_full[0]],
                    other_p_expanded[:, sign_change_full[0] + 1],
                    main_normal,
                    main_center,
                )
                seg_vert_other_0 = (hit[0], False)
                seg_vert_other_1 = (sign_change_full[0], True)

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

                seg_vert_other_0 = (0, "not implemented for shared planes")
                seg_vert_other_1 = (0, "not implemented for shared planes")

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

                seg_vert_other_0 = (hit[0], False)
                seg_vert_other_1 = (hit[1], False)

                # The other polygon has an edge laying in the plane of the main polygon.
                # This will be registered as a boundary intersection, but only if
                # the polygons (not only plane) intersect.
                if (
                    hit[0] + 1 == hit[-1]
                    or hit[0] == 0
                    and hit[-1] == (dot_prod_from_main.size - 2)
                ):
                    isect_on_boundary_other = True

            # Next, analyze intersection between main polygon and the plane of the other
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
                seg_vert_main_0 = (sign_change_other[0], True)
                seg_vert_main_1 = (sign_change_other[1], True)

            elif np.sum(dot_prod_from_other[:-1] == 0) == 1:
                # The first and last element represent the same point, thus include
                # only one of them when counting the number of points in the plane
                # of the other fracture.
                hit = np.where(dot_prod_from_other[:-1] == 0)[0]
                main_intersects_other_0 = main_p_expanded[:, hit[0]]
                sign_change_full = np.where(np.abs(np.diff(dot_prod_from_other)) > 1)[0]
                if sign_change_full.size == 0:
                    # This corresponds to a point contact between one polygon and the
                    # other (at least other plane, perhaps also other polygon)
                    # Simply ignore this for now.S
                    continue
                main_intersects_other_1 = intersection(
                    main_p_expanded[:, sign_change_full[0]],
                    main_p_expanded[:, sign_change_full[0] + 1],
                    other_normal,
                    other_center,
                )
                seg_vert_main_0 = (hit[0], False)
                seg_vert_main_1 = (sign_change_full[0], True)

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

                seg_vert_main_0 = (0, "not implemented for shared planes")
                seg_vert_main_1 = (0, "not implemented for shared planes")
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

                seg_vert_main_0 = (hit[0], False)
                seg_vert_main_1 = (hit[1], False)

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

            ###
            # We now have the intersections between polygons and planes.
            # To finalize the computation, we need to sort out how the intersection
            # points are located relative to each other. Only if there is an overlap
            # between the intersection points of the main and the other polygon
            # is there a real intersection (contained within the polygons, not only)
            # in their planes, but outside the features themselves.

            # Vectors from the intersection points in the main fracture to the
            # intersection point in the other fracture
            main_0_other_0 = other_intersects_main_0 - main_intersects_other_0
            main_0_other_1 = other_intersects_main_1 - main_intersects_other_0
            main_1_other_0 = other_intersects_main_0 - main_intersects_other_1
            main_1_other_1 = other_intersects_main_1 - main_intersects_other_1

            # e_1 is positive if both points of the other fracture lie on the same side of the
            # first intersection point of the main one
            # Use a mod_sign here to avoid issues related to rounding errors
            e_1 = mod_sign(np.sum(main_0_other_0 * main_0_other_1))
            # e_2 is positive if both points of the other fracture lie on the same side of the
            # second intersection point of the main one
            e_2 = mod_sign(np.sum(main_1_other_0 * main_1_other_1))
            # e_3 is positive if both points of the main fracture lie on the same side of the
            # first intersection point of the other one
            e_3 = mod_sign(np.sum((-main_0_other_0) * (-main_1_other_0)))
            # e_4 is positive if both points of the main fracture lie on the same side of the
            # second intersection point of the other one
            e_4 = mod_sign(np.sum((-main_0_other_1) * (-main_1_other_1)))

            # This is in essence an implementation of the flow chart in Figure 9 in Dong et al,
            # However the inequality signs are changed a bit to make the logic clearer
            if e_1 > 0 and e_2 > 0 and e_3 > 0 and e_4 > 0:
                # The intersection points for the two fractures are separated.
                # There is no intersection
                continue
            if (
                sum([e_1 == 0, e_2 == 0]) == 1
                and sum([e_1 > 0, e_2 > 0]) == 1
                and sum([e_3 == 0, e_4 == 0]) == 1
                and sum([e_3 > 0, e_4 > 0]) == 1
            ):
                # Contact in a single point
                continue
            if e_1 >= 0:
                # The first point on the main fracture is at most marginally involved in
                # the intersection (if e_1 == 0, two segments intersect)
                if e_2 >= 0:
                    # The second point on the main fracture is at most marginally involved
                    # We know that e_3 and e_4 are negative (positive is covered above
                    # and a combination is not possible)
                    isect_pt_loc = [other_intersects_main_0, other_intersects_main_1]

                    # Main is intersected in its interior, append two empty lists
                    if e_1 == 0:
                        segment_vertex_intersection[main].append(seg_vert_main_0)
                    else:
                        if isect_on_boundary_main:
                            ind = seg_vert_main_0[0]
                            if ind == 0:
                                ind = num_main - 1
                            segment_vertex_intersection[main].append((ind, True))
                        else:
                            segment_vertex_intersection[main].append([])
                    if e_2 == 0:
                        segment_vertex_intersection[main].append(seg_vert_main_1)
                    else:
                        if isect_on_boundary_main:
                            ind = seg_vert_main_1[0]
                            if not (ind == num_main - 1 and seg_vert_main_0[0] == 0):
                                ind -= 1

                            segment_vertex_intersection[main].append((ind, True))
                        else:
                            segment_vertex_intersection[main].append([])

                    # Other is intersected on two segments
                    segment_vertex_intersection[o].append(seg_vert_other_0)
                    segment_vertex_intersection[o].append(seg_vert_other_1)
                else:
                    # The second point on the main fracture is surrounded by points on
                    # the other fracture. One of them will in turn be surrounded by the
                    # points on the main fracture, this is the intersecting one.
                    if e_3 <= 0:
                        isect_pt_loc = [
                            main_intersects_other_1,
                            other_intersects_main_0,
                        ]

                        segment_vertex_intersection[main].append(seg_vert_main_1)
                        if e_1 == 0:
                            # The first point on the main fracture barely hits the other
                            # fracture
                            segment_vertex_intersection[main].append(seg_vert_main_0)
                        else:
                            if isect_on_boundary_main:
                                # No intersection for the first point of main
                                ind = seg_vert_main_0[0]
                                segment_vertex_intersection[main].append((ind, True))
                            else:
                                segment_vertex_intersection[main].append([])

                        # The second may hit, depending on e_4
                        if e_4 == 0:
                            segment_vertex_intersection[o].append(seg_vert_other_1)
                        else:
                            if isect_on_boundary_other:
                                ind = seg_vert_other_1[0]
                                if not (
                                    ind == num_other - 1 and seg_vert_other_0[0] == 0
                                ):
                                    ind -= 1
                                segment_vertex_intersection[o].append((ind, True))
                            else:
                                segment_vertex_intersection[o].append([])

                        # The first point of other surely hits
                        segment_vertex_intersection[o].append(seg_vert_other_0)

                    elif e_4 <= 0:
                        isect_pt_loc = [
                            main_intersects_other_1,
                            other_intersects_main_1,
                        ]
                        segment_vertex_intersection[main].append(seg_vert_main_1)

                        if e_1 == 0:
                            # The first point on the main fracture barely hits the other
                            # fracture
                            segment_vertex_intersection[main].append(seg_vert_main_0)
                        else:
                            # No intersection for the first point of main
                            segment_vertex_intersection[main].append([])

                        if e_3 == 0:
                            segment_vertex_intersection[o].append(seg_vert_other_0)
                        else:
                            segment_vertex_intersection[o].append([])

                        segment_vertex_intersection[o].append(seg_vert_other_1)

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
                        segment_vertex_intersection[main].append(seg_vert_main_0)
                        segment_vertex_intersection[main].append([])

                        segment_vertex_intersection[o].append([])
                        segment_vertex_intersection[o].append(seg_vert_other_0)

                    elif e_4 <= 0:
                        isect_pt_loc = [
                            main_intersects_other_0,
                            other_intersects_main_1,
                        ]
                        segment_vertex_intersection[main].append(seg_vert_main_0)
                        segment_vertex_intersection[main].append([])

                        segment_vertex_intersection[o].append([])
                        segment_vertex_intersection[o].append(seg_vert_other_1)

                    else:
                        # We may eventually end up here for overlapping fractures
                        assert False
            elif e_1 < 0 and e_2 < 0:
                # The points in on the main fracture are the intersection points
                isect_pt_loc = [main_intersects_other_0, main_intersects_other_1]
                segment_vertex_intersection[main].append(seg_vert_main_0)
                segment_vertex_intersection[main].append(seg_vert_main_1)

                if isect_on_boundary_other:
                    ind_0 = seg_vert_other_0[0]
                    ind_1 = seg_vert_other_1[0]
                    if abs(ind_0 - ind_1) == 1:
                        segment_vertex_intersection[o].append((min(ind_0, ind_1), True))
                        segment_vertex_intersection[o].append((min(ind_0, ind_1), True))
                    else:
                        segment_vertex_intersection[o].append((num_other - 1, True))
                        segment_vertex_intersection[o].append((num_other - 1, True))
                else:
                    segment_vertex_intersection[o].append([])
                    segment_vertex_intersection[o].append([])
            else:
                # This should never happen
                assert False

            # Append data for this combination of polygons.
            new_pt.append(np.array(isect_pt_loc).T)
            num_new = len(isect_pt_loc)
            isect_pt[main].append(new_pt_ind + np.arange(num_new))
            isect_pt[o].append(new_pt_ind + np.arange(num_new))
            new_pt_ind += num_new
            is_bound_isect[main].append(isect_on_boundary_main)
            is_bound_isect[o].append(isect_on_boundary_other)
            polygon_pairs.append((main, o))

    # Cleanup and return. Puh!
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

    return new_pt, isect_pt, is_bound_isect, polygon_pairs, segment_vertex_intersection


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
    try:
        import robust_point_in_polyhedron
    except:
        raise ImportError(
            """Cannot import robust_points_inside_polyhedron.
                          Read documentation of
                          pp.utils.comp_geom.is_inside_polyhedron for install
                          instructions.
                          """
        )

    # The actual test requires that the polyhedra surface is described by
    # a triangulation. To that end, loop over all polygons and compute
    # triangulation. This is again done by a projection to 2d

    # Data storage
    tri = np.zeros((0, 3))
    points = np.zeros((3, 0))

    num_points = 0
    for poly in polyhedron:
        R = project_plane_matrix(poly)
        # Project to 2d, Delaunay
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
    # in the triangulation. Fix this.
    # Note: We cannot do a standard CCW sorting here, since the polygons lie in
    # different planes, and projections to 2d may or may not rotate the polygon.
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
        try:
            is_inside[pi] = np.abs(test_object.winding_number(test_points[:, pi])) > 0
            # If the given point is on the boundary, this will produce an error informing
            # about coplanar or collinear points. Interpret this as a False (not inside)
        except ValueError as err:
            if str(err) in [
                "vertices coplanar with origin",
                "vertices collinear with origin",
                "vertex coincides with origin",
            ]:
                is_inside[pi] = False
            else:
                # If the error is about something else, raise it again.
                raise err

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
            R = project_plane_matrix(g.nodes, tol=tol)
        else:
            R = project_line_matrix(g.nodes, tol=tol)

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
    s1_start = s1_start.ravel()
    s2_start = s2_start.ravel()

    s1_end = s1_end.ravel()
    s2_end = s2_end.ravel()

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
        np.ndarray: For each constrained polygon, corresponding list of its original
            polygon

    """

    if isinstance(polygons, np.ndarray):
        polygons = [polygons]

    constrained_polygons = []
    orig_poly_ind = []

    # Loop over the polygons. For each, find the intersections with all
    # polygons on the side of the polyhedra.
    for pi, poly in enumerate(polygons):
        # Add this polygon to the list of constraining polygons. Put this first
        all_poly = [poly] + polyhedron

        # Find intersections
        coord, point_ind, is_bound, pairs, seg_vert = intersect_polygons_3d(all_poly)

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
                constrained_polygons.append(poly)
                orig_poly_ind.append(pi)
                continue
            elif np.all(np.logical_not(inside)):
                # Do not add it.
                continue
            else:
                # This indicates that the inside_polyhedron test is bad
                assert False

        # At this point we know there are intersections between the polygon and
        # polyhedra. The constrained polygon can have up to three types of segments:
        # 1) Both vertexes are on the boundary. The segment is formed by the pair of
        # intersection points between two polygons.
        # 2) Both vertexes are in the interior. This is one of the original segments
        # of the polygon.
        # 3) A segment of the original polygon crosses on or more of the polyhedron
        # boundaries. This includes the case where the original polygon has a vertex
        # on the polyhedron boundary. This can produce one or several segments.

        # Case 1): Find index of intersection points
        main_ind = point_ind[0]

        # Storage for intersection segments between the main polygon and the
        # polyhedron sides.
        boundary_segments = []

        # First find segments fully on the boundary.
        # Loop over all sides of the polyhedral. Look for intersection points
        # that are both in main and the other
        for other in range(1, len(all_poly)):
            other_ip = point_ind[other]

            common = np.isin(other_ip, main_ind)
            if common.sum() < 2:
                # This is at most a point contact, no need to do anything
                continue
            # There is a real intersection between the segments. Add it.
            boundary_segments.append(other_ip[common])

        boundary_segments = np.array([i for i in boundary_segments]).T

        # For segments with at least one interior point, we need to jointly consider
        # intersection points and the original vertexes
        num_coord = coord.shape[1]
        coord_extended = np.hstack((coord, poly))

        # Convenience arrays for navigating between vertexes in the polygon
        num_vert = poly.shape[1]
        ind = np.arange(num_vert)
        next_ind = 1 + ind
        next_ind[-1] = 0
        prev_ind = np.arange(num_vert) - 1
        prev_ind[0] = num_vert - 1

        # Case 2): Find segments that are defined by two interior points
        points_inside_polyhedron = is_inside_polyhedron(polyhedron, poly)
        # segment_inside[0] tells whehter the point[:, -1] - point[:, 0] is fully inside
        # the remaining elements are point[:, 0] - point[:, 1] etc.
        segments_inside = np.logical_and(
            points_inside_polyhedron, points_inside_polyhedron[next_ind]
        )
        # Temporary list of interior segments, it will be adjusted below
        interior_segments = np.vstack((ind[segments_inside], next_ind[segments_inside]))

        # From here on, we will lean heavily on information on segments that cross the
        # boundary.
        # Only consider segment-vertex information for the first polygon
        seg_vert = seg_vert[0]

        # The test for interior points does not check if the segment crosses the
        # domain boundary due to a convex domain; these must be removed.
        # What we really want is multiple small segments, excluding those that are on
        # the outside of the domain. These are identified below, under case 3.

        # First, count the number of times a segment of the polygon is associated with
        # an intersection point
        count_boundary_segment = np.zeros(num_vert, dtype=np.int)
        for isect in seg_vert:
            # Only consider segment intersections, not interior (len==0), and vertexes
            if len(isect) > 0 and isect[1]:
                count_boundary_segment[isect[0]] += 1

        # Find presumed interior segments that crosses the boundary
        segment_crosses_boundary = np.where(
            np.logical_and(count_boundary_segment > 0, segments_inside)
        )[0]
        # Sanity check: If both points are interior, there must be an even number of
        # segment crossings
        assert np.all(count_boundary_segment[segment_crosses_boundary] % 2 == 0)
        # The index of the segments are associated with the first row of the interior_segments
        # Find the columns to keep by using invert argument to isin
        keep_ind = np.isin(interior_segments[0], segment_crosses_boundary, invert=True)
        # Delete false interior segments.
        interior_segments = interior_segments[:, keep_ind]

        # Adjust index so that it refers to the extended coordinate array
        interior_segments += num_coord

        # Case 3: Where a segment of the original polygon crosses (including start and
        # end point) the polyhedron an unknown number of times. This gives rise to
        # at least one segment, but can be multiple.

        # Storage of identified segments in the constrained polygon
        segments_interior_boundary = []

        # Check if individual vertexs are on the boundary
        vertex_on_boundary = np.zeros(num_vert, np.bool)
        for isect in seg_vert:
            if len(isect) > 0 and not isect[1]:
                vertex_on_boundary[isect[0]] = 1

        # Storage of the intersections associated with each segment of the original polygon
        isects_of_segment = np.zeros(num_vert, np.object)
        for i in range(num_vert):
            isects_of_segment[i] = []

        # Identify intersections of each segment.
        # This is a bit involved, possibly because of a poor choice of data formats:
        # The actual identification of the sub-segments (next for-loop) uses the
        # identified intersection points, with an empty point list signifying that there
        # are no intersections (that is, no sub-segments from this original segment).
        # The only problem is the case where the original segment runs from a vertex
        # on the polyhedron boundary to an interior point: This segment must be processed
        # despite there being no intersections. We achieve that by adding an empty
        # list to the relevant data field, and then remove the list if a true
        # intersection is found later
        for isect_ind, isect in enumerate(seg_vert):
            if len(isect) > 0:
                if isect[1]:
                    # intersection point lies ona segment
                    if len(isects_of_segment[isect[0]]) == 0:
                        isects_of_segment[isect[0]] = [isect_ind]
                    else:
                        # Remove empty list if necessary, then add the information
                        if isinstance(isects_of_segment[isect[0]][0], list):
                            isects_of_segment[isect[0]] = [isect_ind]
                        else:
                            isects_of_segment[isect[0]].append(isect_ind)
                else:
                    # intersection point is on a segment
                    # This segment can be connected to both the previous and next point
                    if len(isects_of_segment[isect[0]]) == 0:
                        isects_of_segment[isect[0]].append([])
                    if len(isects_of_segment[prev_ind[isect[0]]]) == 0:
                        isects_of_segment[prev_ind[isect[0]]].append([])

        # For all original segments that have intersection points (or vertex on a
        # polyhedron boundary), find all points along the segment (original endpoints
        # and intersection points. Find out which of these sub-segments are inside and
        # outside the polyhedron, remove exterior parts
        for seg_ind in range(num_vert):
            if len(isects_of_segment[seg_ind]) == 0:
                continue
            # Index and coordinate of intersection points on this segment
            loc_isect_ind = np.asarray(isects_of_segment[seg_ind], dtype=np.int).ravel()
            isect_coord = coord[:, loc_isect_ind]

            # Start and end of the full segment
            start = poly[:, seg_ind].reshape((-1, 1))
            end = poly[:, next_ind[seg_ind]].reshape((-1, 1))

            # Sanity check
            assert is_collinear(np.hstack((start, isect_coord, end)))
            # Sort the intersection points according to their distance from the start
            sorted_ind = np.argsort(np.sum((isect_coord - start) ** 2, axis=0))

            # Indices (in terms of columns in coords_extended) along the segment
            index_along_segment = np.hstack(
                (
                    num_coord + seg_ind,
                    loc_isect_ind[sorted_ind],
                    num_coord + next_ind[seg_ind],
                )
            )
            # Since the sub-segments are formed by intersection points, every second
            # will be in the interior of the polyhedron. The first one is interior if
            # the start point is in the interior or on the boundary of the polyhedron.
            if points_inside_polyhedron[seg_ind] or vertex_on_boundary[seg_ind]:
                start_pairs = 0
            else:
                start_pairs = 1
            # Define the vertex pairs of the sub-segmetns, and add the relevant ones.
            pairs = np.vstack((index_along_segment[:-1], index_along_segment[1:]))
            for pair_ind in range(start_pairs, pairs.shape[1], 2):
                segments_interior_boundary.append(pairs[:, pair_ind])

        # Clean up boundary-interior segments
        if len(segments_interior_boundary) > 0:
            segments_interior_boundary = np.array(
                [i for i in segments_interior_boundary]
            ).T
        else:
            segments_interior_boundary = np.zeros((2, 0), dtype=np.int)

        # At this stage, we have identified all segments, possibly with duplicates.
        # Next task is to arrive at a unique representation of the segments.
        # To that end, first collect the segments in a single list
        segments = np.sort(
            np.hstack(
                (boundary_segments, interior_segments, segments_interior_boundary)
            ),
            axis=0,
        )
        # Uniquify intersection coordinates, and update the segments
        unique_coords, _, ib = pp.utils.setmembership.unique_columns_tol(
            coord_extended, tol=tol
        )
        unique_segments = ib[segments]
        # Then uniquify the segments, in terms of the unique coordinates
        unique_segments, *rest = pp.utils.setmembership.unique_columns_tol(
            unique_segments
        )

        # The final stage is to collect the constrained polygons.
        # If the segments are connected, which will always be the case if the
        # polyhedron is convex, the graph will have a single connected component.
        # If not, there will be multiple connected components. Find these, and
        # make a separate polygon for each.

        # Represent the segments as a graph.
        graph = nx.Graph()
        for i in range(unique_segments.shape[1]):
            graph.add_edge(unique_segments[0, i], unique_segments[1, i])

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

            # And there we are
            constrained_polygons.append(unique_coords[:, inds])
            orig_poly_ind.append(pi)

    return constrained_polygons, np.array(orig_poly_ind)


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
