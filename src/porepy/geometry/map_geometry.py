"""
Collection of functions related to geometry mappings, rotations etc.

"""
import numpy as np

import porepy as pp

module_sections = ["geometry"]


@pp.time_logger(sections=module_sections)
def force_point_collinearity(pts):
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


@pp.time_logger(sections=module_sections)
def map_grid(g, tol=1e-5, R=None):
    """If a 2d or a 1d grid is passed, the function return the cell_centers,
    face_normals, and face_centers using local coordinates. If a 3d grid is
    passed nothing is applied. The return vectors have a reduced number of rows.

    Parameters:
    g (grid): the grid.
    tol (double, optional): Tolerance used to check that the grid is linear or planar.
        Defaults to 1e-5.
    R (np.array size 3x3, optional ): Rotation matrix. The first dim rows should map
        vectors onto the tangential space of the grid. If not provided, a rotation
        matrix will be computed.

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

    if g.dim == 0 or g.dim == 3:
        if R is None:
            R = np.eye(3)

        return (
            cell_centers,
            face_normals,
            face_centers,
            R,
            np.ones(3, dtype=bool),
            nodes,
        )

    else:  # g.dim == 1 or g.dim == 2:
        if R is None:
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


@pp.time_logger(sections=module_sections)
def sort_points_on_line(pts, tol=1e-5):
    """
    Return the indexes of the point according to their position on a line.

    Parameters:
        pts: the list of points
    Returns:
        argsort: the indexes of the points

    """
    if pts.shape[1] == 1:
        return np.array([0])
    assert pp.geometry_property_checks.points_are_collinear(pts, tol)

    nd, _ = pts.shape

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


@pp.time_logger(sections=module_sections)
def project_points_to_line(p, tol=1e-4):
    """Project a set of colinear points onto a line.

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
    p = p - center

    if p.shape[0] == 2:
        p = np.vstack((p, np.zeros(p.shape[1])))

    # Check that the points indeed form a line
    if not pp.geometry_property_checks.points_are_collinear(p, tol):
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
    # Sort nodes
    coord_1d = p_1d[active_dimension]
    sort_ind = np.argsort(coord_1d)[0]
    sorted_coord = coord_1d[0, sort_ind]

    return sorted_coord, rot, active_dimension, sort_ind


@pp.time_logger(sections=module_sections)
def project_plane_matrix(pts, normal=None, tol=1e-5, reference=None, check_planar=True):
    """Project the points on a plane using local coordinates.

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
    if reference is None:
        reference = [0, 0, 1]

    if normal is None:
        normal = compute_normal(pts)
    else:
        normal = np.asarray(normal)
        normal = normal.flatten() / np.linalg.norm(normal)

    if check_planar:
        assert pp.geometry_property_checks.points_are_planar(pts, normal, tol)

    reference = np.asarray(reference, dtype=np.float)
    angle = np.arccos(np.dot(normal, reference))
    vect = np.array(
        [
            normal[1] * reference[2] - normal[2] * reference[1],
            normal[2] * reference[0] - normal[0] * reference[2],
            normal[0] * reference[1] - normal[1] * reference[0],
        ]
    )
    return rotation_matrix(angle, vect)


@pp.time_logger(sections=module_sections)
def project_line_matrix(pts, tangent=None, tol=1e-5, reference=None):
    """Project the points on a line using local coordinates.

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

    if reference is None:
        reference = [0, 0, 1]

    reference = np.asarray(reference, dtype=np.float)
    angle = np.arccos(np.dot(tangent, reference))
    vect = np.array(
        [
            tangent[1] * reference[2] - tangent[2] * reference[1],
            tangent[2] * reference[0] - tangent[0] * reference[2],
            tangent[0] * reference[1] - tangent[1] * reference[0],
        ]
    )
    return rotation_matrix(angle, vect)


@pp.time_logger(sections=module_sections)
def rotation_matrix(a, vect):
    """Compute the rotation matrix about a vector by an angle using the matrix
    form of Rodrigues formula.

    Parameters:
    a: double, the angle.
    vect: np.array, 1x3, the vector.

    Returns:
    matrix: np.ndarray, 3x3, the rotation matrix.

    NOTE: If vect is a zero vector, the returned rotation matrix will be the
    identify matrix.

    """
    if np.allclose(vect, np.zeros(3)):
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


@pp.time_logger(sections=module_sections)
def normal_matrix(pts=None, normal=None):
    """Compute the normal projection matrix of a plane.

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


@pp.time_logger(sections=module_sections)
def tangent_matrix(pts=None, normal=None):
    """Compute the tangential projection matrix of a plane.

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


@pp.time_logger(sections=module_sections)
def compute_normal(pts, check=True):
    """Compute the normal of a set of points. The sign of the normal is arbitary

    The algorithm assume that the points lie on a plane.
    Three non-aligned points are required.

    Parameters:
    pts: np.ndarray, 3xn, the points. Need n > 2.
    check (boolean, optional): Do sanity check on the results. Defaults to True.

    Returns:
    normal: np.array, 1x3, the normal.

    """
    if pts.shape[1] <= 2:
        raise ValueError("in compute_normal: pts.shape[1] must be larger than 2")
    normal = np.cross(pts[:, 0] - pts[:, 1], pts[:, 2] - pts[:, 1])
    count = 0
    max_count = pts.shape[1] - 3
    while np.allclose(normal, np.zeros(3)) and count <= max_count and check:
        count += 1
        normal = np.cross(pts[:, 0] - pts[:, 1], np.mean(pts, axis=1) - pts[:, 2])
        pts = pts[:, 1:]
    if check and np.allclose(normal, np.zeros(3)):
        raise RuntimeError(
            "Unable to calculate normal from point set. Are all points collinear?"
        )
    return normal / np.linalg.norm(normal)


@pp.time_logger(sections=module_sections)
def compute_normals_1d(pts):
    t = compute_tangent(pts)
    n = np.array([t[1], -t[0], 0]) / np.sqrt(t[0] ** 2 + t[1] ** 2)
    return np.r_["1,2,0", n, np.dot(rotation_matrix(np.pi / 2.0, t), n)]


@pp.time_logger(sections=module_sections)
def compute_tangent(pts, check=True):
    """Compute a tangent vector of a set of points.

    The algorithm assume that the points lie on a plane.

    Parameters:
    pts: np.ndarray, 3xn, the points.
    check: boolean, optional. Do sanity check on the result. Defaults to True.

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
    if check:
        assert not np.allclose(tangent, np.zeros(3))
    return tangent / np.linalg.norm(tangent)
