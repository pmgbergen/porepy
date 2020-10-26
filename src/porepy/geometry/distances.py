"""
Module contains functions for distance computations.

"""

import numpy as np

import porepy as pp


def segment_set(start, end):
    """Compute distance and closest points between sets of line segments.

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
            dl, cpi, cpj = two_segments(start[:, i], end[:, i], start[:, j], end[:, j])
            d[i, j] = dl
            d[j, i] = dl
            cp[i, j, :] = cpi
            cp[j, i, :] = cpj

    return d, cp


def segment_segment_set(start, end, start_set, end_set):
    """Compute distance and closest points between a segment and a set of
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
        dl, cpi, cpj = two_segments(start, end, start_set[:, i], end_set[:, i])
        d[i] = dl
        cp[:, i] = cpi
        cp_set[:, i] = cpj

    return d, cp, cp_set


def two_segments(s1_start, s1_end, s2_start, s2_end):
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

    # Variable used to fine almost parallel lines. Sensitivity to this value has not
    # been tested.
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


def points_segments(p, start, end):
    """Compute distances between points and line segments.

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
            d[pi, less] = point_pointset(p[:, pi], start[:, less])
            cp[pi, less, :] = np.swapaxes(start[:, less], 1, 0)
            # Similarly, above one signifies closest to end
            above = np.ma.greater_equal(proj, 1)
            d[pi, above] = point_pointset(p[:, pi], end[:, above])
            cp[pi, above, :] = np.swapaxes(end[:, above], 1, 0)

            # Points inbetween
            between = np.logical_not(np.logical_or(less, above).data)
            proj_p = start[:, between] + proj[between] * line[:, between]
            d[pi, between] = point_pointset(p[:, pi], proj_p)
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
            d[less, ei] = point_pointset(start[:, ei], p[:, less])
            cp[less, ei, :] = start[:, ei]
            # Similarly, above one signifies closest to end
            above = np.ma.greater_equal(proj, 1)
            d[above, ei] = point_pointset(end[:, ei], p[:, above])
            cp[above, ei, :] = end[:, ei]

            # Points inbetween
            between = np.logical_not(np.logical_or(less, above).data)
            proj_p = start[:, ei].reshape((-1, 1)) + proj[between] * line[
                :, ei
            ].reshape((-1, 1))
            for proj_i, bi in enumerate(np.where(between)[0]):
                d[bi, ei] = np.min(point_pointset(proj_p[:, proj_i], p[:, bi]))
                cp[bi, ei, :] = proj_p[:, proj_i]

    return d, cp


def point_pointset(p, pset, exponent=2):
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


def pointset(p, max_diag=False):
    """Compute mutual distance between all points in a point set.

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
        d[i] = point_pointset(p[:, i], p)

    if max_diag:
        for i in range(n):
            d[i, i] = 2 * np.max(d[i])

    return d


def points_polygon(p, poly, tol=1e-5):
    """Compute distance from points to a polygon. Also find closest point on
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
    rot_p = pp.map_geometry.project_plane_matrix(poly)
    irot = rot_p.transpose()
    poly_rot = rot_p.dot(poly)

    # Sanity check: The points should lay on a plane
    assert np.all(np.abs(poly_rot[2]) < tol)

    poly_xy = poly_rot[:2]

    # Rotate the point set, using the same coordinate system.
    p = rot_p.dot(p)

    in_poly = pp.geometry_property_checks.point_in_polygon(poly_xy, p)

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

    d_outside, p_outside = points_segments(orig_p[:, outside_poly], start, end)

    for i, pi in enumerate(outside_poly):
        mi = np.argmin(d_outside[i])
        d[pi] = d_outside[i, mi]
        cp[:, pi] = p_outside[i, mi, :]

    return d, cp, in_poly


def segments_polygon(start, end, poly, tol=1e-5):
    """Compute the distance from line segments to a polygon.

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
    rot_p = pp.map_geometry.project_plane_matrix(poly)
    irot = rot_p.transpose()
    poly_rot = rot_p.dot(poly)

    # Sanity check: The points should lay on a plane
    assert np.all(np.abs(poly_rot[2]) < tol)

    poly_xy = poly_rot[:2]

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
    inside = pp.geometry_property_checks.point_in_polygon(poly_xy, x0[:2])
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
        pp.geometry_property_checks.point_in_polygon(poly_xy, start[:2]),
        pp.geometry_property_checks.point_in_polygon(poly_xy, end[:2]),
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
    d_start_poly, cp_s_p, _ = points_polygon(start, poly)
    d_end_poly, cp_e_p, _ = points_polygon(end, poly)

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
            ds, cp_s, _ = two_segments(
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
