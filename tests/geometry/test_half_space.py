"""Tests of functions in pp.geometry.half_space."""

import numpy as np
import pytest

from porepy.applications.test_utils.arrays import compare_arrays
from porepy.geometry import half_space


def test_one_half_space():
    # Test point on the right side of a single half space
    n = np.array([[-1], [0], [0]])
    x0 = np.array([[0], [0], [0]])
    pts = np.array([[1, -1], [0, 0], [0, 0]])
    out = half_space.point_inside_half_space_intersection(n, x0, pts)
    assert np.all(out == np.array([True, False]))


def test_two_half_spaces():
    # Test points on the right side of two half spaces
    n = np.array([[-1, 0], [0, -1], [0, 0]])
    x0 = np.array([[0, 0], [0, 1], [0, 0]])
    pts = np.array([[1, -1, 1, 0], [2, 0, 2, 0], [0, 0, 0, 0]])
    out = half_space.point_inside_half_space_intersection(n, x0, pts)
    assert np.all(out == np.array([True, False, True, False]))


def test_half_space_interior_point_convex_2d():
    # Find interior point of convex half space
    n = np.array([[0, 0, 0, 0, 1, -1], [1, 1, -1, -1, 0, 0], [0, 0, 0, 0, 0, 0]])
    x0 = np.array(
        [
            [0, 0, 0, 0, 0, 2.0 / 3.0],
            [0, 0, 1.0 / 3.0, 1.0 / 3.0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    known_pt = np.array([1.0 / 6.0, 1.0 / 6.0, 0.0]).reshape((-1, 1))
    pts = np.array([[0, 2.0 / 3.0], [0, 1.0 / 3.0], [0, 0]])
    pt = half_space.half_space_interior_point(n, x0, pts).reshape((-1, 1))
    assert np.all(
        np.sign(np.sum(n * pt, axis=0)) == np.sign(np.sum(n * known_pt, axis=0))
    )


def test_half_space_interior_point_star_shaped_2d():
    # Find interior point in star-shaped, but not convex domain
    n = np.array(
        [[0, 0, 1, 0, -1, 0, 1], [1, 1, 0, 1, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0]]
    )
    x0 = np.array(
        [
            [0, 0, 2.0 / 3.0, 0, 1, 0, 0],
            [1.0 / 3.0, 1.0 / 3.0, 0, 0, 0, 2.0 / 3.0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    pts = np.array([[0, 1], [0, 2.0 / 3.0], [0, 0]])
    pt = half_space.half_space_interior_point(n, x0, pts).reshape((-1, 1))

    # Verify that the computed point is on the same side of all the normal
    # vectors as a point known to be in the interior.
    known_pt = np.array([5.0 / 6.0, 1.0 / 2.0, 0.0]).reshape((-1, 1))
    assert np.all(
        np.sign(np.sum(n * (pt - x0), axis=0))
        == np.sign(np.sum(n * (known_pt - x0), axis=0))
    )


def test_half_space_interior_point_convex_3d():
    # Find interior point in convex domain made of half spaces in 3d.
    n = np.array([[1, -1, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 1, -1]])
    x0 = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1]])
    pts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    pt = half_space.half_space_interior_point(n, x0, pts).reshape((-1, 1))

    # Verify that the computed point is on the same side of all the normal
    # vectors as a point known to be in the interior.
    known_pt = np.array([1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0]).reshape((-1, 1))
    assert np.all(
        np.sign(np.sum(n * (pt - x0), axis=0))
        == np.sign(np.sum(n * (known_pt - x0), axis=0))
    )


def test_half_space_interior_point_star_shaped_3d():
    n = np.array(
        [
            [0, 0, 1, 0, -1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, -1],
        ]
    )
    x0 = np.array(
        [
            [0, 0, 2.0 / 3.0, 0, 1, 0, 0, 0, 0],
            [1.0 / 3.0, 1.0 / 3.0, 0, 0, 0, 2.0 / 3.0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    known_pt = np.array([5.0 / 6.0, 1.0 / 2.0, 1.0 / 6.0]).reshape((-1, 1))
    pts = np.array([[0, 1], [0, 2.0 / 3.0], [0, 1]])
    pt = half_space.half_space_interior_point(n, x0, pts).reshape((-1, 1))
    assert np.all(
        np.sign(np.sum(n * (pt - x0), axis=0))
        == np.sign(np.sum(n * (known_pt - x0), axis=0))
    )


def test_half_space_interior_point_test0():
    # Extra test of half spaces, presumably this is a difficult configuration.
    n = np.array(
        [
            [
                -0.4346322145903923,
                -0.5837212981640826,
                -0.5545602917404856,
                0.8874304062782964,
                0.9432862408780104,
                -0.5285915818615673,
            ],
            [
                -0.9006080379611606,
                -0.8119540911096134,
                0.8321435470065893,
                -0.4609417251808932,
                0.3319805231790443,
                0.8488762804938574,
            ],
            [0.0, -0.0, 0.0, -0.0, 0.0, 0.0],
        ]
    )
    pts = np.array(
        [
            [
                0.6264867087233955,
                0.6264867087233955,
                0.6292849480051679,
                0.6767304875123992,
                0.6767304875123992,
                0.6622642106131997,
                0.6598935979971375,
                0.6012244303376967,
                0.6012244303376967,
                0.6598935979971375,
                0.6622642106131997,
                0.6292849480051679,
            ],
            [
                0.8042920815320497,
                0.8042920815320497,
                0.8411535645991798,
                0.8205852695643441,
                0.8205852695643441,
                0.8616896085330092,
                0.7881699621908844,
                0.8224533668748337,
                0.8224533668748337,
                0.7881699621908844,
                0.8616896085330092,
                0.8411535645991798,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    x0 = np.array(
        [
            [
                0.6264867087233955,
                0.6264867087233955,
                0.6292849480051679,
                0.6767304875123992,
                0.6767304875123992,
                0.6622642106131997,
            ],
            [
                0.8042920815320497,
                0.8042920815320497,
                0.8411535645991798,
                0.8205852695643441,
                0.8205852695643441,
                0.8616896085330092,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    x1 = pts[:, 6:12]
    x0 = (x1 + x0) / 2

    pt = half_space.half_space_interior_point(n, x0, pts)
    pt_known = np.array([0.6484723969922208, 0.8225131504113889, 0.0])
    assert np.allclose(pt, pt_known)


def test_half_space_interior_point_test1():
    # Test of half space interior point, difficult configuration.
    n = np.array(
        [
            [
                1.0,
                0.6773171037156254,
                -0.0965964317843146,
                0.5615536850805473,
                -0.8227124789179447,
                -0.8663868741056542,
                -0.3297308258566596,
            ],
            [
                -0.0,
                -0.7356911994949217,
                0.9953236304672657,
                0.8274403052622333,
                0.5684577178935039,
                -0.4993733917395212,
                -0.9440749877419088,
            ],
            [0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0],
        ]
    )
    pts = np.array(
        [
            [
                1.0,
                1.0,
                1.0,
                0.9456767570553427,
                0.9002821557415752,
                0.9236181291669759,
                0.9236181291669759,
                1.0,
                0.9689475893045396,
                0.9456767570553427,
                0.9188491193586552,
                0.9188491193586552,
                0.9002821557415752,
                0.9689475893045396,
            ],
            [
                0.3333333333328948,
                0.3333333333328948,
                0.3749999999996713,
                0.3697279143475474,
                0.3610634261905756,
                0.3205767254478346,
                0.3205767254478346,
                0.3749999999996713,
                0.3047448047626512,
                0.3697279143475474,
                0.3879348577545479,
                0.3879348577545479,
                0.3610634261905756,
                0.3047448047626512,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    x0 = np.array(
        [
            [
                1.0,
                0.9844737946522698,
                0.9728383785276713,
                0.9322629382069989,
                0.9095656375501152,
                0.9119501424542755,
                0.9462828592357577,
            ],
            [
                0.354166666666283,
                0.319039069047773,
                0.3723639571736094,
                0.3788313860510477,
                0.3744991419725617,
                0.3408200758192051,
                0.3126607651052429,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    pt = half_space.half_space_interior_point(n, x0, pts)

    pt_known = np.array([0.9411337621203867, 0.3417142038105351, 0])
    assert np.allclose(pt, pt_known)


@pytest.mark.parametrize(
    "normals,x,known_vertexes",
    [
        (
            # Unit triangle
            np.array([[-1, 0, 1], [0, -1, 1]]).T,
            np.array([0, 0, -1]),
            np.array([[0, 1, 0], [0, 0, 1]]),
        ),
        (
            # Truncated unit triangle
            np.array([[-1, 0, 1, 0], [0, -1, 1, 1]]).T,
            np.array([0, 0, -1, -0.5]),
            np.array([[0, 1, 0, 0.5], [0, 0, 0.5, 0.5]]),
        ),
        (
            # Unit tetrahedral
            np.array([[-1, 0, 0, 1], [0, -1, 0, 1], [0, 0, -1, 1]]).T,
            np.array([0, 0, 0, -1]),
            np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        ),
        (
            # Truncated unit tetrahedral
            np.array([[-1, 0, 0, 1, 0], [0, -1, 0, 1, 0], [0, 0, -1, 1, 1]]).T,
            np.array([0, 0, 0, -1, -0.5]),
            np.array(
                [[0, 1, 0, 0.5, 0, 0], [0, 0, 1, 0, 0.5, 0], [0, 0, 0, 0.5, 0.5, 0.5]]
            ),
        ),
    ],
)
def test_vertexes_convex_domain(normals, x, known_vertexes):
    # Test of function to get vertexes of convex polygons.
    # Test cases (parametrized) cover 2d and 3d domains.
    vertexes = half_space.vertexes_of_convex_domain(normals, x)
    assert compare_arrays(vertexes, known_vertexes)
