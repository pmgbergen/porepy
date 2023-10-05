import numpy as np

from porepy import distances

# Test segment_segment_set


def setup_2d_unit_square():
    p00 = np.array([0, 0])
    p10 = np.array([1, 0])
    p01 = np.array([0, 1])
    p11 = np.array([1, 1])
    return p00, p10, p11, p01


def test_segment_no_intersect_2d():
    p00, p10, p11, p01 = setup_2d_unit_square()
    d, _, _ = distances.segment_segment_set(p00, p01, p11, p10)
    assert d == 1


def test_segment_intersect_2d():
    p00, p10, p11, p01 = setup_2d_unit_square()
    d, cp_1, cp_2 = distances.segment_segment_set(p00, p11, p10, p01)
    assert d == 0

    assert np.allclose(cp_1, np.array([0.5, 0.5]))
    assert np.allclose(cp_2, np.array([0.5, 0.5]))


def test_line_passing():
    # Lines not crossing
    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    p3 = np.array([2, -1])
    p4 = np.array([2, 1])
    d, cp1, cp2 = distances.segment_segment_set(p1, p2, p3, p4)
    assert d == 1
    assert np.allclose(cp1, np.array([[1], [0]]))
    assert np.allclose(cp2, np.array([[2], [0]]))


def test_share_point():
    # Two lines share a point
    p1 = np.array([0, 0])
    p2 = np.array([0, 1])
    p3 = np.array([1, 1])
    d, cp1, cp2 = distances.segment_segment_set(p1, p2, p2, p3)
    assert d == 0
    assert np.allclose(cp1, p2.reshape((-1, 1)))
    assert np.allclose(cp2, p2.reshape((-1, 1)))


def test_intersection_3d():
    p000 = np.array([0, 0, 0])
    p111 = np.array([1, 1, 1])
    p100 = np.array([1, 0, 0])
    p011 = np.array([0, 1, 1])
    d, cp1, cp2 = distances.segment_segment_set(p000, p111, p100, p011)
    assert d == 0
    assert np.allclose(cp1, np.array([0.5, 0.5, 0.5]).reshape((-1, 1)))
    assert np.allclose(cp2, np.array([0.5, 0.5, 0.5]).reshape((-1, 1)))


def test_changed_order_3d():
    # The order of the start and endpoints of the segments should not matter
    p1 = np.random.rand(1, 3)[0]
    p2 = np.random.rand(1, 3)[0]
    p3 = np.random.rand(1, 3)[0]
    p4 = np.random.rand(1, 3)[0]
    d1, cp11, cp12 = distances.segment_segment_set(p1, p2, p3, p4)
    d2, cp21, cp22 = distances.segment_segment_set(p2, p1, p3, p4)
    d3, cp31, cp32 = distances.segment_segment_set(p1, p2, p4, p3)
    d4, cp41, cp42 = distances.segment_segment_set(p2, p1, p4, p3)
    d5, cp51, cp52 = distances.segment_segment_set(p4, p3, p2, p1)
    assert np.allclose(d1, d2)
    assert np.allclose(d1, d3)
    assert np.allclose(d1, d4)
    assert np.allclose(d1, d5)

    assert np.allclose(cp11, cp21)
    assert np.allclose(cp31, cp11)
    assert np.allclose(cp41, cp11)
    assert np.allclose(cp52, cp11)
    assert np.allclose(cp12, cp22)
    assert np.allclose(cp12, cp32)
    assert np.allclose(cp12, cp42)
    assert np.allclose(cp12, cp51)


# Test pointset


def test_unit_square():
    p = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
    d = distances.pointset(p)
    s2 = np.sqrt(2)
    known = np.array([[0, 1, s2, 1], [1, 0, 1, s2], [s2, 1, 0, 1], [1, s2, 1, 0]])
    assert np.allclose(d, known)


def test_3d():
    p = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]).T
    d = distances.pointset(p)
    known = np.array([[0, 1, 1], [1, 0, np.sqrt(2)], [1, np.sqrt(2), 0]])
    assert np.allclose(d, known)


def test_zero_diagonal():
    sz = 5
    p = np.random.rand(3, sz)
    d = distances.pointset(p)
    assert np.allclose(np.diagonal(d), np.zeros(sz))


def test_symmetry():
    p = np.random.rand(3, 7)
    d = distances.pointset(p)
    assert np.allclose(d, d.T)


def test_single_point():
    p = np.random.rand(2)
    d = distances.pointset(p)
    assert d.shape == (1, 1)
    assert d[0, 0] == 0


# Test points_segments


def test_single_point_and_segment():
    p = np.array([0, 0])
    start = np.array([1, 0])
    end = np.array([1, 1])

    d, cp = distances.points_segments(p, start, end)

    assert d[0, 0] == 1
    assert np.allclose(cp[0, 0, :], np.array([1, 0]))


def test_many_points_single_segment():
    p = np.array([[0, 1], [1, 1], [1.5, 1], [2, 1], [3, 1]]).T
    start = np.array([1, 0])
    end = np.array([2, 0])

    d, cp = distances.points_segments(p, start, end)

    assert d.shape[0] == 5
    assert d.shape[1] == 1

    known_d = np.array([np.sqrt(2), 1, 1, 1, np.sqrt(2)]).reshape((-1, 1))
    assert np.allclose(d, known_d)

    known_cp = np.array([[[1, 0]], [[1, 0]], [[1.5, 0]], [[2, 0]], [[2, 0]]])
    assert np.allclose(cp, known_cp)


def test_single_point_many_segments():
    p = np.array([1, 1])
    start = np.array([[0, 0], [0, 1], [0, 2]]).T
    end = np.array([[2, 0], [2, 1], [2, 2]]).T

    d, cp = distances.points_segments(p, start, end)

    assert d.shape[0] == 1
    assert d.shape[1] == 3

    known_d = np.array([1, 0, 1])
    assert np.allclose(d[0], known_d)

    known_cp = np.array([[[1, 0], [1, 1], [1, 2]]])
    assert np.allclose(cp, known_cp)


def test_many_points_and_segments():
    p = np.array([[0, 0, 0], [1, 0, 0]]).T

    start = np.array([[0, 0, -1], [0, 1, 1]]).T
    end = np.array([[0, 0, 1], [1, 1, 1]]).T

    d, cp = distances.points_segments(p, start, end)

    assert d.shape[0] == 2
    assert d.shape[1] == 2

    known_d = np.array([[0, np.sqrt(2)], [1, np.sqrt(2)]])
    assert np.allclose(d, known_d)

    known_cp = np.array([[[0, 0, 0], [0, 1, 1]], [[0, 0, 0], [1, 1, 1]]])
    assert np.allclose(cp, known_cp)


def test_point_closest_segment_end():
    p = np.array([0, 0])
    start = np.array([[1, 0], [1, 1]]).T
    end = np.array([[2, 0], [2, 1]]).T

    d, cp = distances.points_segments(p, start, end)

    known_d = np.array([1, np.sqrt(2)])
    assert np.allclose(d[0], known_d)

    known_cp = np.array([[[1, 0], [1, 1]]])
    assert np.allclose(cp, known_cp)


def test_flipped_lines():
    p = np.array([0, 0])
    start = np.array([[1, 0], [1, 1]]).T
    end = np.array([[1, 1], [1, 0]])

    d, cp = distances.points_segments(p, start, end)
    known_d = np.array([1, 1])
    assert np.allclose(d[0], known_d)

    known_cp = np.array([[[1, 0], [1, 0]]])
    assert np.allclose(cp, known_cp)


# Test points_polygon


def test_norot_poly():
    poly = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
    p = np.array(
        [
            [0.5, 0.5, 0],
            [0.5, 0, 0],
            [0.5, 0.5, 1],
            [0, 0, 1],
            [0.5, 0, 1],
            [2, 0.5, 0],
            [2, 0, 1],
        ]
    ).T

    d, cp, in_poly = distances.points_polygon(p, poly)

    known_d = np.array([0, 0, 1, 1, 1, 1, np.sqrt(2)])
    known_cp = np.array(
        [
            [0.5, 0.5, 0],
            [0.5, 0, 0],
            [0.5, 0.5, 0],
            [0, 0, 0],
            [0.5, 0, 0],
            [1, 0.5, 0],
            [1, 0, 0],
        ]
    ).T

    assert np.allclose(d, known_d)
    assert np.allclose(cp, known_cp)


def test_rot_poly():
    poly = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]).T

    p = np.array(
        [
            [0, 0, 0],
            [0, 0.5, 0.5],
            [2, 0.5, 0.5],
            [0, 0, 0.5],
            [0, -1, 0.5],
            [1, 0, 0],
            [1, 0.5, 0.5],
        ]
    ).T

    d, cp, in_poly = distances.points_polygon(p, poly)

    known_d = np.array([1, 1, 1, 1, np.sqrt(2), 0, 0])
    known_cp = np.array(
        [
            [1, 0, 0],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0, 0.5],
            [1, 0, 0.5],
            [1, 0, 0],
            [1, 0.5, 0.5],
        ]
    ).T

    assert np.allclose(d, known_d)
    assert np.allclose(cp, known_cp)


# Test segments_polygon


def test_segment_intersects_no_rot():
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T

    start = np.array([0.5, 0.5, -1])
    end = np.array([0.5, 0.5, 1])

    d, cp = distances.segments_polygon(start, end, p)

    known_d = np.array([0])
    known_cp = np.array([0.5, 0.5, 0]).reshape((-1, 1))

    assert np.allclose(d, known_d)
    assert np.allclose(cp, known_cp)


def test_segment_no_intersection_intersect_extrusion__no_rot():
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T

    start = np.array([0.5, 0.5, 1])
    end = np.array([0.5, 0.5, 2])

    d, cp = distances.segments_polygon(start, end, p)

    known_d = np.array([1])
    known_cp = np.array([0.5, 0.5, 0]).reshape((-1, 1))

    assert np.allclose(d, known_d)
    assert np.allclose(cp, known_cp)


def test_segment_intersects_in_endpoint_no_rot():
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T

    start = np.array([0.5, 0.5, 0])
    end = np.array([0.5, 0.5, 1])

    d, cp = distances.segments_polygon(start, end, p)

    known_d = np.array([0])
    known_cp = np.array([0.5, 0.5, 0]).reshape((-1, 1))

    assert np.allclose(d, known_d)
    assert np.allclose(cp, known_cp)


def test_segment_no_intersection_no_rot():
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T

    start = np.array([1.5, 0.5, -1])
    end = np.array([1.5, 0.5, 1])

    d, cp = distances.segments_polygon(start, end, p)

    known_d = np.array([0.5])
    known_cp = np.array([1.5, 0.5, 0]).reshape((-1, 1))

    assert np.allclose(d, known_d)
    assert np.allclose(cp, known_cp)


def test_segment_in_polygon():
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T

    start = np.array([0.5, 0.5, 0])
    end = np.array([0.7, 0.7, 0])

    d, cp = distances.segments_polygon(start, end, p)

    known_d = np.array([0])

    assert np.allclose(d, known_d)
    assert cp[2] == 0  # x and y coordinate of closest point is not clear in this case.


def test_segment_parallel_polygon():
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T

    start = np.array([0.5, 0.5, 1])
    end = np.array([0.7, 0.7, 1])

    d, cp = distances.segments_polygon(start, end, p)

    known_d = np.array([1])

    assert np.allclose(d, known_d)
    assert cp[2] == 0  # x and y coordinate of closest point is not clear in this case.


def test_segment_parallel_polygon_extends_outside_both_sides():
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T

    start = np.array([-0.5, -0.5, 0])
    end = np.array([1.7, 1.7, 0])

    d, cp = distances.segments_polygon(start, end, p)

    known_d = np.array([0])

    assert np.allclose(d, known_d)
    assert cp[2] == 0  # x and y coordinate of closest point is not clear in this case.


def test_segment_parallel_polygon_extends_outside_intersection():
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T

    start = np.array([0.5, 0.5, 0])
    end = np.array([1.7, 1.7, 0])

    d, cp = distances.segments_polygon(start, end, p)

    known_d = np.array([0])

    assert np.allclose(d, known_d)
    assert cp[2] == 0  # x and y coordinate of closest point is not clear in this case.


def test_segment_parallel_polygon_extends_outside_no_intersection():
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T

    start = np.array([0.5, 0.5, 1])
    end = np.array([1.7, 1.7, 1])

    d, cp = distances.segments_polygon(start, end, p)

    known_d = np.array([1])

    assert np.allclose(d, known_d)
    assert cp[2] == 0  # x and y coordinate of closest point is not clear in this case.
