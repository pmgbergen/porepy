import numpy as np
from porepy import map_geometry
import pytest

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


def test_collinear_points():
    """Test that giving collinear points throws an assertion."""
    # pts in xy-plane
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]).T

    with pytest.raises(RuntimeError):
        _ = map_geometry.compute_normal(pts)
