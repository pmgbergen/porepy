import numpy as np
import pytest

from porepy import geometry_property_checks

# --------- Testing point_in_polygon ---------


@pytest.fixture
def poly():
    return np.array([[0, 1, 1, 0], [0, 0, 1, 1]])


def test_inside(poly):
    p = np.array([0.5, 0.5])
    assert np.all(geometry_property_checks.point_in_polygon(poly, p))


def test_outside(poly):
    p = np.array([2, 2])

    inside = geometry_property_checks.point_in_polygon(poly, p)

    assert not inside[0]


def test_on_line(poly):
    # Point on the line, but not the segment, of the polygon
    p = np.array([2, 0])

    inside = geometry_property_checks.point_in_polygon(poly, p)
    assert not inside[0]


def test_on_boundary(poly):
    p = np.array([0, 0.5])

    inside = geometry_property_checks.point_in_polygon(poly, p)
    assert not inside[0]


def test_just_inside(poly):
    p = np.array([0.5, 1e-6])
    assert geometry_property_checks.point_in_polygon(poly, p)


def test_multiple_points(poly):
    p = np.array([[0.5, 0.5], [0.5, 1.5]])

    inside = geometry_property_checks.point_in_polygon(poly, p)

    assert inside[0]
    assert not inside[1]


def test_large_polygon():
    a = np.array(
        [
            [
                -2.04462568e-01,
                -1.88898782e-01,
                -1.65916617e-01,
                -1.44576869e-01,
                -7.82444375e-02,
                -1.44018389e-16,
                7.82444375e-02,
                1.03961083e-01,
                1.44576869e-01,
                1.88898782e-01,
                2.04462568e-01,
                1.88898782e-01,
                1.44576869e-01,
                7.82444375e-02,
                1.44018389e-16,
                -7.82444375e-02,
                -1.44576869e-01,
                -1.88898782e-01,
            ],
            [
                -1.10953147e-16,
                7.82444375e-02,
                1.19484803e-01,
                1.44576869e-01,
                1.88898782e-01,
                2.04462568e-01,
                1.88898782e-01,
                1.76059749e-01,
                1.44576869e-01,
                7.82444375e-02,
                1.31179355e-16,
                -7.82444375e-02,
                -1.44576869e-01,
                -1.88898782e-01,
                -2.04462568e-01,
                -1.88898782e-01,
                -1.44576869e-01,
                -7.82444375e-02,
            ],
        ]
    )
    b = np.array([[0.1281648, 0.04746067], [-0.22076491, 0.16421546]])
    inside = geometry_property_checks.point_in_polygon(a, b)
    assert not inside[0]
    assert inside[1]


# --------- Testing point_in_polyhedron ---------


@pytest.fixture
def cart_polyhedron():
    west = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
    east = np.array([[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
    south = np.array([[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 1, 1]])
    north = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]])
    bottom = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
    top = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
    return [west, east, south, north, bottom, top]


@pytest.fixture
def non_convex_polyhedron():
    west = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
    east = np.array([[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
    south_w = np.array([[0, 0.5, 0.5, 0], [0, 0, 0, 0], [0, 0.5, 1, 1]])
    south_e = np.array([[0.5, 1, 1, 0.5], [0, 0, 0, 0], [0.5, 0, 1, 1]])
    north_w = np.array([[0, 0.5, 0.5, 0], [1, 1, 1, 1], [0, 0.5, 1, 1]])
    north_e = np.array([[0.5, 1, 1, 0.5], [1, 1, 1, 1], [0.5, 0, 1, 1]])
    bottom_w = np.array([[0, 0.5, 0.5, 0], [0, 0, 1, 1], [0, 0.5, 0.5, 0]])
    bottom_e = np.array([[0.5, 1, 1, 0.5], [0, 0, 1, 1], [0.5, 0.0, 0, 0.5]])
    top_w = np.array([[0, 0.5, 0.5, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
    top_e = np.array([[0.5, 1, 1, 0.5], [0, 0, 1, 1], [1, 1, 1, 1]])
    return [
        west,
        east,
        south_w,
        south_e,
        north_w,
        north_e,
        bottom_w,
        bottom_e,
        top_w,
        top_e,
    ]


def test_point_inside_box(cart_polyhedron):
    p = np.array([0.3, 0.5, 0.5])
    is_inside = geometry_property_checks.point_in_polyhedron(cart_polyhedron, p)
    assert is_inside.size == 1
    assert is_inside[0] == 1


def test_two_points_inside_box(cart_polyhedron):
    p = np.array([[0.3, 0.5, 0.5], [0.5, 0.5, 0.5]]).T
    is_inside = geometry_property_checks.point_in_polyhedron(cart_polyhedron, p)
    assert is_inside.size == 2
    assert np.all(is_inside[0] == 1)


def test_point_outside_box(cart_polyhedron):
    p = np.array([1.5, 0.5, 0.5])
    is_inside = geometry_property_checks.point_in_polyhedron(cart_polyhedron, p)
    assert is_inside.size == 1
    assert is_inside[0] == 0


def test_point_inside_non_convex(non_convex_polyhedron):
    p = np.array([0.5, 0.5, 0.7])
    is_inside = geometry_property_checks.point_in_polyhedron(non_convex_polyhedron, p)
    assert is_inside.size == 1
    assert is_inside[0] == 1


def test_point_outside_non_convex_inside_box(non_convex_polyhedron):
    p = np.array([0.5, 0.5, 0.3])
    is_inside = geometry_property_checks.point_in_polyhedron(non_convex_polyhedron, p)
    assert is_inside.size == 1
    assert is_inside[0] == 0


# --------- Testing point_in_cell ---------


def test_planar_square():
    pts = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=float)

    pt = np.array([0.2, 0.3, 0])
    assert geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([1.1, 0.5, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([-0.1, 0.5, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([0.5, 1.1, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([0.5, -0.1, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([1.1, 1.1, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([-0.1, 1.1, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([-0.1, -0.1, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([1.1, -0.1, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)


def test_planar_square_1():
    pts = np.array([[0, 0.5, 1, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0]], dtype=float)

    pt = np.array([0.2, 0.3, 0])
    assert geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([1.1, 0.5, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([-0.1, 0.5, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([0.5, 1.1, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([0.5, -0.1, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([1.1, 1.1, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([-0.1, 1.1, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([-0.1, -0.1, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([1.1, -0.1, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)


def test_planar_convex():
    pts = np.array(
        [[0, 1, 1.2, 0.5, -0.2], [0, 0, 1, 1.2, 1], [0, 0, 0, 0, 0]], dtype=float
    )

    pt = np.array([0.2, 0.3, 0])
    assert geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([0.5, 1, 0])
    assert geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([1.3, 0.5, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([-0.1, 0.5, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([1.1, -0.1, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)


def test_planar_convex_1():
    pts = np.array(
        [[0, 0.5, 1, 1.2, 0.5, -0.2], [0, 0, 0, 1, 1.2, 1], [0, 0, 0, 0, 0, 0]],
        dtype=float,
    )

    pt = np.array([0.2, 0.3, 0])
    assert geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([0.5, 1, 0])
    assert geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([1.3, 0.5, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([-0.1, 0.5, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([1.1, -0.1, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)


def test_planar_concave():
    pts = np.array(
        [[0, 0.5, 1, 0.4, 0.5, -0.2], [0, 0, 0, 1, 1.2, 1], [0, 0, 0, 0, 0, 0]],
        dtype=float,
    )

    pt = np.array([0.2, 0.3, 0])
    assert geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([0.5, 1, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([1.3, 0.5, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([-0.1, 0.5, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)

    pt = np.array([1.1, -0.1, 0])
    assert not geometry_property_checks.point_in_cell(pts, pt)


# --------- Testing points_are_planar ---------


def test_is_planar_2d():
    pts = np.array([[0.0, 2.0, -1.0], [0.0, 4.0, 2.0], [2.0, 2.0, 2.0]])
    assert geometry_property_checks.points_are_planar(pts)


def test_is_planar_3d():
    pts = np.array(
        [
            [0.0, 1.0, 0.0, 4.0 / 7.0],
            [0.0, 1.0, 1.0, 0.0],
            [5.0 / 8.0, 7.0 / 8.0, 7.0 / 4.0, 1.0 / 8.0],
        ]
    )
    assert geometry_property_checks.points_are_planar(pts)


# --------- Testing is_ccw_polyline ---------


def make_ccw_vectors():
    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    p3 = np.array([1, 1])
    return p1, p2, p3


def make_ccw_close_vectors(y):
    p1 = np.array([0, 0])
    p2 = np.array([2, 0])
    p3 = np.array([1, y])
    return p1, p2, p3


def test_is_ccw():
    p1, p2, p3 = make_ccw_vectors()
    assert geometry_property_checks.is_ccw_polyline(p1, p2, p3)


def test_not_ccw():
    p1, p2, p3 = make_ccw_vectors()
    assert not geometry_property_checks.is_ccw_polyline(p1, p3, p2)


def test_ccw_on_boundary():
    p1, p2, p3 = make_ccw_close_vectors(0)
    assert geometry_property_checks.is_ccw_polyline(p1, p2, p3, default=True)
    assert not geometry_property_checks.is_ccw_polyline(p1, p2, p3, default=False)


def test_tolerance():
    p1, p2, p3 = make_ccw_close_vectors(1e-6)

    # Safety margin saves ut
    assert geometry_property_checks.is_ccw_polyline(p1, p2, p3, tol=1e-4, default=True)

    # Default kills us, even though we're inside safety margin
    assert not geometry_property_checks.is_ccw_polyline(
        p1, p2, p3, tol=1e-4, default=False
    )

    # Outside safety margin, and on the ccw side
    assert geometry_property_checks.is_ccw_polyline(p1, p2, p3, tol=1e-8, default=False)


def test_tolerance_outside():
    p1, p2, p3 = make_ccw_close_vectors(-1e-6)

    # Safety margin saves ut
    assert geometry_property_checks.is_ccw_polyline(p1, p2, p3, tol=1e-4, default=True)

    # Default kills us, even though we're inside safety margin
    assert not geometry_property_checks.is_ccw_polyline(
        p1, p2, p3, tol=1e-4, default=False
    )

    # Outside safety margin, and not on the ccw side
    assert not geometry_property_checks.is_ccw_polyline(
        p1, p2, p3, tol=1e-8, default=False
    )


def test_several_points():
    p1, p2, _ = make_ccw_vectors()

    p_test = np.array([[0.5, 0.5], [1, -1]])
    known = np.array([1, 0], dtype=bool)
    assert np.allclose(
        known,
        geometry_property_checks.is_ccw_polyline(
            p1, p2, p_test, tol=1e-8, default=False
        ),
    )
