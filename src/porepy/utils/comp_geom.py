"""
Various utility functions related to computational geometry.

Some functions (add_point, split_edges, ...?) are mainly aimed at finding
intersection between lines, with grid generation in mind, and should perhaps
be moved to a separate module.

"""
from __future__ import division
import logging
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
