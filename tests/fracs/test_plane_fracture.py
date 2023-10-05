import numpy as np

from porepy.fracs import plane_fracture
import pytest

# Test computation of fracture centroids.


def test_frac_3d_1():
    # Simple plane, known center.
    f_1 = plane_fracture.PlaneFracture(
        np.array([[0, 2, 2, 0], [0, 2, 2, 0], [-1, -1, 1, 1]]),
        check_convexity=False,
    )
    c_known = np.array([1, 1, 0]).reshape((3, 1))
    n_known = np.array([1, -1, 0]).reshape((1, 3))
    assert np.allclose(c_known, f_1.center)
    assert np.allclose(np.cross(n_known, f_1.normal.T), 0)


def test_frac_3d_2():
    # Center away from xy-plane
    f_1 = plane_fracture.PlaneFracture(
        np.array([[0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 1, 1]]), check_convexity=False
    )
    c_known = np.array([1, 1, 0.5]).reshape((3, 1))
    assert np.allclose(c_known, f_1.center)


def test_frac_3d_3():
    # Fracture plane defined by x + y + z = 1
    f_1 = plane_fracture.PlaneFracture(
        np.array([[0, 1, 1, 0], [0, 0, 1, 1], [1, 0, -1, 0]]), check_convexity=False
    )
    c_known = np.array([0.5, 0.5, 0]).reshape((3, 1))
    assert np.allclose(c_known, f_1.center)


def test_frac_3d_4():
    # Fracture plane defined by x + y + z = 4
    f_1 = plane_fracture.PlaneFracture(
        np.array([[0, 1, 1, 0], [0, 0, 1, 1], [4, 3, 2, 3]]), check_convexity=False
    )
    c_known = np.array([0.5, 0.5, 3]).reshape((3, 1))
    assert np.allclose(c_known, f_1.center)


def test_frac_3d_rand():
    # Random normal vector.
    r = np.random.rand(4)
    x = np.array([0, 1, 1, 0])
    y = np.array([0, 0, 1, 1])
    z = (r[0] - r[1] * x - r[2] * y) / r[3]
    f = plane_fracture.PlaneFracture(np.vstack((x, y, z)), check_convexity=False)
    z_cc = (r[0] - 0.5 * r[1] - 0.5 * r[2]) / r[3]
    c_known = np.array([0.5, 0.5, z_cc]).reshape((3, 1))
    assert np.allclose(c_known, f.center)


# Test index attribute of Fracture. The attribute may need to change later, these tests
# should ensure that we don't rush it.


def make_plane_fracture(index: int | None):
    array = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1]])
    return plane_fracture.PlaneFracture(array, index=index, check_convexity=False)


def test_equal_index():
    f1 = make_plane_fracture(index=1)
    f2 = make_plane_fracture(index=1)
    assert f1 == f2


def test_unequal_index():
    f1 = make_plane_fracture(index=1)
    f2 = make_plane_fracture(index=4)
    assert f1 != f2


def test_no_index():
    # One fracture has no index set. Should not be equal
    f1 = make_plane_fracture(index=1)
    f2 = make_plane_fracture(index=None)
    assert f1 != f2


# Testing is_vertex method.


@pytest.fixture
def frac():
    return plane_fracture.PlaneFracture(
        np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]), check_convexity=False
    )


def test_not_vertex(frac):
    p = np.array([0.5, 0.5, 0])
    is_vert, ind = frac.is_vertex(p)
    assert is_vert == False
    assert ind is None


def test_is_vertex(frac):
    p = np.array([0.0, 0.0, 0])
    is_vert, ind = frac.is_vertex(p)
    assert is_vert == True
    assert ind == 0


def test_tolerance_sensitivity(frac):
    # Borderline, is a vertex depending on the tolerance
    p = np.array([1e-4, 0, 0])
    is_vert, ind = frac.is_vertex(p, tol=1e-5)
    assert is_vert == False
    assert ind is None

    is_vert, ind = frac.is_vertex(p, tol=1e-3)
    assert is_vert == True
    assert ind == 0


# Testing copy method.


def test_copy(frac):
    f2 = frac.copy()
    assert id(frac) != id(f2)


def test_deep_copy(frac):
    f2 = frac.copy()

    # Points should be identical
    assert np.allclose(frac.pts, f2.pts)

    f2.pts[0, 0] = 7
    assert not np.allclose(frac.pts, f2.pts)


# Testing __repr__ and __str__


def test_str(frac):
    frac.__str__()


def test_repr(frac):
    frac.__repr__()
