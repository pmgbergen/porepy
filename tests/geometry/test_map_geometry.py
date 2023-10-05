import numpy as np
import pytest

from porepy import map_geometry

# ---------- Testing compute_normal ----------


def test_axis_normal():
    """Test that we get the correct normal in x, y and z direction."""
    # pts in xy-plane
    pts_xy = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]).T
    # pts in xz-plane
    pts_xz = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1]]).T
    # pts in yz-plane
    pts_yz = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]]).T

    nz = map_geometry.compute_normal(pts_xy)
    ny = map_geometry.compute_normal(pts_xz)
    nx = map_geometry.compute_normal(pts_yz)

    # The method does not guarantee normal direction. Should it?
    assert np.allclose(nz, np.array([0, 0, 1])) or np.allclose(nz, -np.array([0, 0, 1]))
    assert np.allclose(ny, np.array([0, 1, 0])) or np.allclose(ny, -np.array([0, 1, 0]))
    assert np.allclose(nx, np.array([1, 0, 0])) or np.allclose(nx, -np.array([1, 0, 0]))


def test_compute_normal_2d():
    pts = np.array([[0.0, 2.0, -1.0], [0.0, 4.0, 2.0], [0.0, 0.0, 0.0]])
    normal = map_geometry.compute_normal(pts)

    # Known normal vector, up to direction
    normal_test = np.array([0.0, 0.0, 1.0])
    pt = pts[:, 0]

    assert np.allclose(np.linalg.norm(normal), 1.0)
    assert np.allclose(
        [np.dot(normal, p - pt) for p in pts[:, 1:].T],
        np.zeros(pts.shape[1] - 1),
    )
    # Normal vector should be equal to the known one, up to a flip or direction
    assert np.allclose(normal, normal_test) or np.allclose(-normal, normal_test)


def test_compute_normal_3d():
    pts = np.array(
        [[2.0, 0.0, 1.0, 1.0], [1.0, -2.0, -1.0, 1.0], [-1.0, 0.0, 2.0, -8.0]]
    )
    normal_test = np.array([7.0, -5.0, -1.0])
    normal_test = normal_test / np.linalg.norm(normal_test)
    normal = map_geometry.compute_normal(pts)
    pt = pts[:, 0]

    assert np.allclose(np.linalg.norm(normal), 1.0)
    assert np.allclose(
        [np.dot(normal, p - pt) for p in pts[:, 1:].T],
        np.zeros(pts.shape[1] - 1),
    )
    assert np.allclose(normal, normal_test) or np.allclose(-normal, normal_test)


def test_collinear_points():
    """Test that giving collinear points throws an assertion."""
    # pts in xy-plane
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]).T

    with pytest.raises(RuntimeError):
        _ = map_geometry.compute_normal(pts)


# ---------- Testing project_plane_matrix ----------


def test_project_plane():
    pts = np.array(
        [[2.0, 0.0, 1.0, 1.0], [1.0, -2.0, -1.0, 1.0], [-1.0, 0.0, 2.0, -8.0]]
    )
    R = map_geometry.project_plane_matrix(pts)
    P_pts = np.dot(R, pts)

    # Known z-coordinates (up to a sign change) of projected points
    known_z = 1.15470054

    assert np.allclose(P_pts[2], known_z) or np.allclose(P_pts[2], -known_z)
