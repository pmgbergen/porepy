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



# ------------------------------------------------------------------------------#




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
        coord, point_ind, is_bound, pairs, seg_vert = pp.intersections.polygons_3d(all_poly)

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
