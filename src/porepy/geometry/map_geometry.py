"""
Collection of functions related to geometry mappings, rotations etc.

"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

import porepy as pp


def force_point_collinearity(
    pts: np.ndarray[Any, np.dtype[np.float64]]
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Given a set of points, return them aligned on a line.
    Useful to enforce collinearity for almost collinear points. The order of the
    points remain the same.

    Parameter:
        pts: (3 x num_pts) the input points. The first point should be on one
            extremum of the line.

    Return:
        pts: (3 x num_pts) the corrected points.

    """
    assert pts.shape[1] > 1

    delta = pts - np.tile(pts[:, 0], (pts.shape[1], 1)).T
    dist = np.sqrt(np.einsum("ij,ij->j", delta, delta))
    end = np.argmax(dist)

    dist /= dist[end]

    return pts[:, 0, np.newaxis] * (1 - dist) + pts[:, end, np.newaxis] * dist


def map_grid(
    sd: pp.Grid, tol: float = 1e-5, R=None
) -> Tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """If a 2d or a 1d grid is passed, the function returns the cell_centers,
    face_normals, and face_centers using local coordinates. If a 3d grid is
    passed nothing is applied. The return vectors have a reduced number of rows.

    Parameters:
        sd (grid): the subdomain grid.
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
    cell_centers = sd.cell_centers
    face_normals = sd.face_normals
    face_centers = sd.face_centers
    nodes = sd.nodes

    if sd.dim == 0 or sd.dim == 3:
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
            if sd.dim == 2:
                R = project_plane_matrix(sd.nodes, tol=tol)
            else:
                R = project_line_matrix(sd.nodes)

        face_centers = np.dot(R, face_centers)

        check = np.sum(np.abs(face_centers.T - face_centers[:, 0]), axis=0)
        check /= np.sum(check)
        dim = np.logical_not(np.isclose(check, 0, atol=tol, rtol=0))
        assert sd.dim == np.sum(dim)
        face_centers = face_centers[dim, :]
        cell_centers = np.dot(R, cell_centers)[dim, :]
        face_normals = np.dot(R, face_normals)[dim, :]
        nodes = np.dot(R, nodes)[dim, :]

    return cell_centers, face_normals, face_centers, R, dim, nodes


def sort_points_on_line(
    pts: np.ndarray[Any, np.dtype[np.float64]], tol: float = 1e-5
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Return the indexes of the point according to their position on a line.

    Parameters:
        pts: Array of points
        tol: Tolerance used in check for point collinearity.

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


def project_points_to_line(
    p: np.ndarray[Any, np.dtype[np.float64]], tol: float = 1e-4
) -> Tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    int,
    np.ndarray[Any, np.dtype[np.int64]],
]:
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


def project_plane_matrix(
    pts: np.ndarray[Any, np.dtype[np.float64]],
    normal: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None,
    tol: float = 1e-5,
    reference: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None,
    check_planar: bool = True,
) -> np.ndarray[Any, np.dtype[np.float64]]:
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
        check_planar:
            Whether to check for planarity.

    Returns:
        np.ndarray, 3x3, projection matrix.

    """
    if reference is None:
        reference = np.array([0, 0, 1])

    if normal is None:
        normal = compute_normal(pts, tol=tol)
    else:
        normal = np.asarray(normal)
        normal = normal.flatten() / np.linalg.norm(normal)
    assert normal is not None

    if check_planar:
        assert pp.geometry_property_checks.points_are_planar(pts, normal, tol)

    reference = np.asarray(reference, dtype=float)
    angle = np.arccos(np.dot(normal, reference))
    vect = np.array(
        [
            normal[1] * reference[2] - normal[2] * reference[1],
            normal[2] * reference[0] - normal[0] * reference[2],
            normal[0] * reference[1] - normal[1] * reference[0],
        ]
    )
    return rotation_matrix(angle, vect)


def project_line_matrix(
    pts: np.ndarray[Any, np.dtype[np.float64]],
    tangent: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None,
    reference: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None,
) -> np.ndarray[Any, np.dtype[np.float64]]:
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
        reference = np.array([0.0, 0.0, 1.0])

    # Appease mypy
    assert type(tangent) == np.ndarray

    reference = np.asarray(reference, dtype=float)
    angle = np.arccos(np.dot(tangent, reference))
    vect = np.array(
        [
            tangent[1] * reference[2] - tangent[2] * reference[1],
            tangent[2] * reference[0] - tangent[0] * reference[2],
            tangent[0] * reference[1] - tangent[1] * reference[0],
        ]
    )
    return rotation_matrix(angle, vect)


def rotation_matrix(
    a: float, vect: np.ndarray[Any, np.dtype[np.float64]]
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute the rotation matrix about a vector by an angle using the matrix
    form of Rodrigues formula.

    Parameters:
        a: double, the angle.
        vect: np.array, 1x3, the vector.

    Returns:
        matrix: np.ndarray, 3x3, the rotation matrix.

    NOTE: If vect is a zero vector, the returned rotation matrix will be the
    identity matrix.

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


def normal_matrix(
    pts: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None,
    normal: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None,
):
    """Compute the normal projection matrix of a plane.

    The algorithm assume that the points lie on a plane.

    Three non-aligned points are required.

    Parameters:
        pts (optional): np.ndarray, 3xn, the points. Need n > 2.
        normal (optional): np.array, 3x1, the normal.

        Either points or normal are mandatory.

    Returns:
        normal matrix: np.array, 3x3, the normal matrix.

    Raises:
        ValueError if neither pts nor normal is provided.

    """
    if normal is not None:
        normal = normal / np.linalg.norm(normal)
    elif pts is not None:
        normal = compute_normal(pts)
    else:
        raise ValueError(
            "Need either points or normal vector to compute normal matrix."
        )
    assert type(normal) == np.ndarray

    # Ravel normal vector for the calculation to work
    return np.tensordot(normal, normal.ravel(), axes=0)


def tangent_matrix(
    pts: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None,
    normal: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute the tangential projection matrix of a plane.

    The algorithm assume that the points lie on a plane.
    Three non-aligned points are required.

    Either points or normal are mandatory.

    Parameters:
        pts: np.ndarray, 3xn, the points. Need n > 2.
        normal: np.array, 3x1, the normal.

    Returns:
        tangential matrix: np.array, 3x3, the tangential matrix.

    Raises:
        ValueError if neither pts nor normal is provided.

    """
    if pts is None and normal is None:
        raise ValueError(
            "Need either points or normal vector to compute tangent matrix."
        )

    return np.eye(3) - normal_matrix(pts, normal)


def compute_normal(
    pts: np.ndarray[Any, np.dtype[np.float64]], tol: float = 1e-5
) -> np.ndarray:
    """Compute the normal of a set of points. The sign of the normal is arbitrary

    The algorithm assumes that the points lie on a plane.

    Three non-aligned points are needed. If the points are almost collinear,
    the algorithm will attempt to find a combination of points that minimizes
    rounding errors. To ensure stable results, make sure to provide points
    that truly span a 2d plane.

    Parameters:
        pts: np.ndarray, 3xn, the points. Need n > 2.
        tol: Absolute tolerance used to detect essentially collinear points.

    Returns:
        normal: np.array, 1x3, the normal.

    Raises:
        ValueError if less than three points are provided.
        ValueError if all points provided are collinear (relative to the specified
            tolerance)

    """
    if pts.shape[1] <= 2:
        raise ValueError("in compute_normal: pts.shape[1] must be larger than 2")

    # Center of the point cloud, and vectors from the center to all points
    center = pts.mean(axis=1).reshape((-1, 1))
    v = pts - center

    # To do the cross product, we need two vectors in the plane of the point cloud.
    # In an attempt at minimizing the vulnerabilities with respect to rounding errors,
    # the vectors should be carefully chosen.
    # As the first vector, choose the longest one.

    # Norm of all vectors
    nrm = np.linalg.norm(v, axis=0)
    # Index of the longest vector (will be needed below)
    v1_ind = np.argmax(nrm)
    v1 = v[:, v1_ind]

    # Next, compute the cross product between the longest vector and all vectors in
    # the plane
    cross = np.array(
        [
            v1[1] * v[2] - v1[2] * v[1],
            v1[2] * v[0] - v1[0] * v[2],
            v1[0] * v[1] - v1[1] * v[0],
        ]
    )

    # Find the index of the longest cross product, and thereby of the vector in v that
    # produced the longest vector.
    cross_ind = np.argmax(np.linalg.norm(cross, axis=0))

    # Pick out the normal vector, using the longest normal
    normal = cross[:, cross_ind]

    # Check on the computation, if the cross product is essentially zero, the points
    # are collinear, and the computation is not to be trusted.
    # We need to use absolute tolerance when invoking np.allclose, since relative
    # tolerance makes no sense when comparing with a zero vector - see numpy.allclose
    # documentation for details.
    # Still, the tolerance should be scaled with the size of v1 and v[:, cross_ind]
    # (the cross product a x b = |a||b|sin(theta) - to detect a small theta, we need to
    # scale the cross product by the lengths of a and b).
    nrm_scaling = nrm[v1_ind] * nrm[cross_ind]

    if np.allclose(normal, np.zeros(3), atol=tol * nrm_scaling):
        raise RuntimeError(
            "Unable to calculate normal from point set. Are all points collinear?"
        )
    return normal / np.linalg.norm(normal)


def compute_normals_1d(
    pts: np.ndarray[Any, np.dtype[np.float64]]
) -> np.ndarray[Any, np.dtype[np.float64]]:
    t = compute_tangent(pts)
    n = np.array([t[1], -t[0], 0]) / np.sqrt(t[0] ** 2 + t[1] ** 2)
    return np.r_["1,2,0", n, np.dot(rotation_matrix(np.pi / 2.0, t), n)]


def compute_tangent(
    pts: np.ndarray[Any, np.dtype[np.float64]], check: bool = True
) -> np.ndarray[Any, np.dtype[np.float64]]:
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
    # Find the point that is the furthest away from the mean point
    max_ind = np.argmax(np.sum(tangent**2, axis=0))
    tangent = tangent[:, max_ind]
    if check:
        assert not np.allclose(tangent, np.zeros(3))
    return tangent / np.linalg.norm(tangent)
