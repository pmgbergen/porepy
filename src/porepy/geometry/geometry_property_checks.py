"""
Functions for (boolean) inquires about geometric objects, and relations between
objects.
"""
from typing import List, Union

import numpy as np
import scipy

import porepy as pp


def is_ccw_polygon(poly: np.ndarray) -> bool:
    """
    Determine if the vertices of a polygon are sorted counterclockwise.

    The method computes the winding number of the polygon, see
        http://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
    and
        http://blog.element84.com/polygon-winding.html

    for descriptions of the algorithm.

    The algorithm should work for non-convex polygons. If the polygon is
    self-intersecting (e.g. shaped like the number 8), the number returned will
    reflect whether the method is 'mostly' cw or ccw.

    NOTE: The test can *not* be used to determine whether the vertexes of a polygon are
    ordered in a natural fashion, that is, not self-intersecting.

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
    conuterclockwise circle. The function can also test several points vs a
    single segment.

    The test is positive if p3 lies to left of the line running through p1 and
    p2.

    The function is intended for 2D points; higher-dimensional coordinates will
    be ignored.

    Extensions to lines with more than three points should be straightforward,
    the input points should be merged into a 2d array.

    Parameters:
        p1 (np.ndarray, length 2): Point on dividing line
        p2 (np.ndarray, length 2): Point on dividing line
        p3 (np.ndarray): Points to be tested
        tol (double, optional): Tolerance used in the comparison, can be used
            to account for rounding errors. Defaults to zero.

        default (boolean, optional): Mode returned if the point is within the
            tolerance. Should be set according to what is desired behavior of
            the function (will vary with application). Defaults to False.

    Returns:
        boolean or np.array of booleans. true if the points form a ccw polyline.
            An array is returned if more than one point is tested.

    See also:
        is_ccw_polygon

    """
    if p3.size > 2:
        num_points = p3.shape[1]
    else:
        num_points = 1

    # Compute cross product between p1-p2 and p1-p3. Right-hand rule gives that
    # p3 is to the left if the cross product is positive.
    cross_product = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (
        p3[0] - p1[0]
    )

    # Should there be a scaling of the tolerance relative to the distance
    # between the points?
    is_ccw = np.ones(num_points, dtype=bool)
    is_ccw[np.abs(cross_product) <= tol] = default

    is_ccw[cross_product < -tol] = False
    is_ccw[cross_product > tol] = True

    if num_points == 1:
        return is_ccw[0]
    else:
        return is_ccw


def point_in_polygon(poly, p, default=False):
    """
    Check if a set of points are inside a polygon.

    The polygon need not be convex.

    Credit: The implementation is a translation of an implementation by
        Mark Dickenson to the PorePy data format for polygons. The original
        source code can be found at:

        https://github.com/mdickinson/polyhedron/blob/master/polygon.py

    Parameters:
        poly (np.ndarray, 2 x n): vertexes of polygon. The segments are formed by
            connecting subsequent columns of poly
        p (np.ndarray, 2 x n2): Points to be tested.
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

    # Roll the polygon vertexes once.
    next_vert = np.roll(poly, -1, axis=1)

    num_pt = pt.shape[1]
    poly_size = poly.shape[1]

    inside = default * np.ones(num_pt, dtype=bool)

    # Loop over all vertexes, find the status of each of them.
    for i in range(num_pt):
        # For description of the method, see the original source code, link above.

        pi = pt[:, i].reshape((-1, 1))

        poly_pi_x = poly[0] - pi[0]
        poly_pi_y = poly[1] - pi[1]

        next_pi_x = next_vert[0] - pi[0]
        next_pi_y = next_vert[1] - pi[1]

        if np.logical_or(
            np.logical_and(poly_pi_x == 0, poly_pi_y == 0),
            np.logical_and(next_pi_x == 0, next_pi_y == 0),
        ).any():
            # The point is on a vertex of the polygon. In this case, we keep
            # the default value.
            continue

        vertex_sgn_poly = np.sign(poly_pi_x)
        hit = vertex_sgn_poly == 0
        vertex_sgn_poly[hit] = np.sign(poly_pi_y)[hit]

        vertex_sgn_next = np.sign(next_pi_x)
        hit = vertex_sgn_next == 0
        vertex_sgn_next[hit] = np.sign(next_pi_y)[hit]

        edge_boundary = vertex_sgn_next - vertex_sgn_poly

        edge_sgn = np.sign(poly_pi_x * next_pi_y - poly_pi_y * next_pi_x)
        if np.any(edge_sgn == 0):
            # The point is on an edge of the polygon. In this case, we keep
            # the default value.
            continue

        contrib = np.zeros(poly_size)

        edge_sgn_active = edge_boundary != 0

        contrib[edge_sgn_active] = edge_sgn[edge_sgn_active]
        winding_number = np.sum(contrib) / 2
        inside[i] = np.abs(winding_number) > 0

    return inside


def point_in_polyhedron(
    polyhedron: Union[np.ndarray, List[np.ndarray]],
    test_points: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    """Test whether a set of point is inside a polyhedron.

    Parameters:
        polyhedron (nested np.ndarray): Each outer element represents a side
            of the polyhedron. Each side is assumed to be a convex polygon.
        test_points (np.ndarray, 3 x num_pt): Points to be tested.
        tol (double, optional): Geometric tolerance, used in comparison of
            points. Defaults to 1e-8.

    Returns:
        np.ndarray, size num_pt: Element i is 1 (True) if test_points[:, i] is
            inside the polygon.

    """

    # The actual test requires that the polyhedra surface is described by
    # a triangulation. To that end, loop over all polygons and compute
    # triangulation. This is again done by a projection to 2d

    # Data storage
    tri = np.zeros((0, 3))
    points = np.zeros((3, 0))

    num_points = 0
    for poly in polyhedron:
        # Shortcut if the polygon already is a triangle
        if poly.shape[1] == 3:
            simplices = np.array([0, 1, 2])
        else:
            R = pp.map_geometry.project_plane_matrix(poly)
            # Project to 2d, Delaunay
            p_2d = R.dot(poly)[:2]
            loc_tri = scipy.spatial.Delaunay(p_2d.T)
            simplices = loc_tri.simplices

        # Add the triangulation, with indices adjusted for the number of points
        # already added
        tri = np.vstack((tri, num_points + simplices))
        points = np.hstack((points, poly))
        num_points += simplices.max() + 1

    # Uniquify points, and update triangulation
    upoints, ia, ib = pp.utils.setmembership.uniquify_point_set(points, tol)
    ut = ib[tri.astype(int)]

    # The in-polyhedra algorithm requires a very particular ordering of the vertexes
    # in the triangulation. Fix this.
    # Note: We cannot do a standard CCW sorting here, since the polygons lie in
    # different planes, and projections to 2d may or may not rotate the polygon.
    sorted_t = pp.utils.sort_points.sort_triangle_edges(ut.T).T

    # Generate tester for points
    test_object = pp.point_in_polyhedron_test.PointInPolyhedronTest(upoints.T,sorted_t,tol)

    if test_points.size < 4:
        test_points = test_points.reshape((-1, 1))

    is_inside = np.zeros(test_points.shape[1], dtype=bool)

    # Loop over all points, check if they are inside.
    for pi in range(test_points.shape[1]):
        # NOTE: The documentation of the test is a bit unclear, but it seems
        # a winding number of 0 means outside, non-zero is inside
        try:
            # Winding number is a real number. It is zero when the point is outside
            is_inside[pi] = np.abs(test_object.winding_number(test_points[:, pi])) > tol

            # If the given point is on the boundary is considered outside.
            # For the point being tested Coniciding vertex, collinearity and coplanarity are check
        except ValueError as err:
            if str(err) in [
                "Test point coincides with a vertex",
                "Test point is collinear with the vertices",
                "Test point is coplanar with the vertices",
            ]:
                is_inside[pi] = False
            else:
                # If the error is about something else, raise it again.
                raise err

    return is_inside


def points_are_planar(pts, normal=None, tol=1e-5):
    """Check if the points lie on a plane.

    Parameters:
    pts (np.ndarray, 3xn): the points.
    normal: (optional) the normal of the plane, otherwise three points are
        required.

    Returns:
    check, bool, if the points lie on a plane or not.

    """

    if normal is None:
        normal = pp.map_geometry.compute_normal(pts)
    else:
        normal = normal.flatten() / np.linalg.norm(normal)

    # Force normal vector to be a column vector
    normal = normal.reshape((-1, 1))
    # Mean point in the point cloud
    cp = np.mean(pts, axis=1).reshape((-1, 1))
    # Dot product between the normal vector and the vector from the center point to the
    # individual points
    dot_prod = np.linalg.norm(np.sum(normal * (pts - cp), axis=0))
    # The points are planar if all the dot products are essentially zero.
    return np.all(np.isclose(dot_prod, 0, atol=tol, rtol=0))


def point_in_cell(poly, p, if_make_planar=True):
    """
    Check whatever a point is inside a cell. Note a similar behaviour could be
    reached using the function is_inside_polygon, however the current
    implementation deals with concave cells as well. Not sure which is the best,
    in terms of performance, for convex cells.

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
        R = pp.map_geometry.project_plane_matrix(poly)
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


def points_are_collinear(pts, tol=1e-5):
    """Check if the points lie on a line.

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


def polygon_hanging_nodes(p, edges, tol=1e-8):
    """
    Find hanging nodes of a polygon

    Parameters:
        p (np.ndarray, nd x n_pt): Point coordinates
        edges (TYPE): Indices, referring to columns in p, of edges in the polygon.
            Should be ordered so that edges[1, i] == edges[0, i+1], and
            edges[0, 0] == edges[1, -1].
        tol (TYPE, optional): Tolerance for when two segments will be considered
            parallel. Defaults to 1e-8.

    Returns:
        np.array: Index of edges with hanging nodes. For an index i, the lines defined
            by edges[:, i] and edges[:, i+1] (or edges[:, 0] if i == edges.shape[1] - 1)
            are parallel.

    """
    # Data structure for storing indices of the hanging nodes
    ind = []

    edges_expanded = np.hstack((edges, edges[:, 0].reshape((-1, 1))))
    # Loop over all edges
    for i in range(edges.shape[1]):
        # Find the vector along this edge, and along the following
        v_this = p[:, edges_expanded[1, i]] - p[:, edges_expanded[0, i]]
        v_next = p[:, edges_expanded[1, i + 1]] - p[:, edges_expanded[0, i + 1]]

        nrm_this = np.linalg.norm(v_this)
        nrm_next = np.linalg.norm(v_next)
        # If the dot product of the normalized vectors is (almost) unity, this is
        # a hanging node
        dot_prod = (v_this / nrm_this).dot(v_next / nrm_next)
        if dot_prod > 1 - tol:
            ind.append(i)

    return np.asarray(ind)
