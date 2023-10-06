"""Test functionality related to plane_fracture module."""
import numpy as np
import pytest

from porepy.fracs import plane_fracture


def test_plane_fracture_center_normal():
    # Simple plane, known center.
    fracture = plane_fracture.PlaneFracture(
        np.array([[0, 2, 2, 0], [0, 2, 2, 0], [-1, -1, 1, 1]]),
        check_convexity=False,
    )
    center_known = np.array([1, 1, 0]).reshape((3, 1))
    normal_known = np.array([1, -1, 0]).reshape((1, 3))
    assert np.allclose(center_known, fracture.center)
    assert np.allclose(np.cross(normal_known, fracture.normal.T), 0)


@pytest.mark.parametrize(
    "points__center_known",
    [
        # Center away from xy-plane
        ([[0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 1, 1]], [1, 1, 0.5]),
        # Fracture plane defined by x + y + z = 1
        ([[0, 1, 1, 0], [0, 0, 1, 1], [1, 0, -1, 0]], [0.5, 0.5, 0]),
        # Fracture plane defined by x + y + z = 4
        ([[0, 1, 1, 0], [0, 0, 1, 1], [4, 3, 2, 3]], [0.5, 0.5, 3]),
    ],
)
def test_plane_fracture_center(points__center_known):
    points = np.array(points__center_known[0])
    center_known = np.array(points__center_known[1]).reshape((3, 1))
    fracture = plane_fracture.PlaneFracture(points, check_convexity=False)
    assert np.allclose(center_known, fracture.center)


def test_plane_fracture_center_normal_random():
    # Random normal vector.
    random = np.random.rand(4)
    x = np.array([0, 1, 1, 0])
    y = np.array([0, 0, 1, 1])
    z = (random[0] - random[1] * x - random[2] * y) / random[3]
    fracture = plane_fracture.PlaneFracture(np.vstack((x, y, z)), check_convexity=False)
    z_center = (random[0] - 0.5 * random[1] - 0.5 * random[2]) / random[3]
    center_known = np.array([0.5, 0.5, z_center]).reshape((3, 1))
    assert np.allclose(center_known, fracture.center)


@pytest.mark.parametrize(
    "indices_expected",
    [
        # Equal
        (1, 1, True),
        # Unequal
        (1, 4, False),
        # One fracture has no index set. Should not be equal
        (1, None, False),
    ],
)
def test_fracture_index(indices_expected):
    """Test index attribute of Fracture. The attribute may need to change later, these
    tests should ensure that we don't rush it."""
    i0, i1, expected = indices_expected
    array = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1]])
    f1 = plane_fracture.PlaneFracture(array, index=i0, check_convexity=False)
    f2 = plane_fracture.PlaneFracture(array, index=i1, check_convexity=False)
    assert (f1 == f2) is expected


@pytest.fixture
def frac() -> plane_fracture.PlaneFracture:
    return plane_fracture.PlaneFracture(
        np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]), check_convexity=False
    )


@pytest.mark.parametrize(
    "point_expected_tol",
    [
        #  Not vertex
        ([0.5, 0.5, 0], False, 1e-4),
        # Vertex
        ([0, 0, 0.0], True, 1e-4),
        # Borderline, is a vertex depending on the tolerance.
        ([1e-4, 0, 0], False, 1e-5),
        ([1e-4, 0, 0], True, 1e-3),
    ],
)
def test_fracture_is_vertex(frac: plane_fracture.PlaneFracture, point_expected_tol):
    """Testing is_vertex method."""
    point = np.array(point_expected_tol[0])
    expected_is_vertex = point_expected_tol[1]
    tol = point_expected_tol[2]
    is_vert, ind = frac.is_vertex(point, tol=tol)
    assert is_vert == expected_is_vertex
    assert (ind == 0) if expected_is_vertex else (ind is None)


def test_fracture_copy(frac: plane_fracture.PlaneFracture):
    """Testing copy method. Should make a deep copy."""
    f2 = frac.copy()
    assert id(frac) != id(f2)

    # Points should be identical
    assert np.allclose(frac.pts, f2.pts)

    f2.pts[0, 0] = 7
    assert not np.allclose(frac.pts, f2.pts)
