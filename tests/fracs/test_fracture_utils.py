"""Testing functionality related to fracture utils module.

Created on Mon Mar 26 10:12:47 2018

@author: eke001
"""

import numpy as np
import pytest

import porepy as pp
from porepy import frac_utils


def arrays_equal(a, b, tol=1e-5):
    # Utility function
    def nrm(x, y):
        if x.ndim == 1:
            x = np.reshape(x, (-1, 1))
        return np.sqrt(np.sum(np.power(x - y, 2), axis=0))

    is_true = True
    for i in range(a.shape[1]):
        is_true *= np.min(nrm(a[:, i], b)) < tol

    for i in range(b.shape[1]):
        is_true *= np.min(nrm(b[:, i], a)) < tol
    return is_true


@pytest.mark.parametrize(
    "points_edges_expected",
    [
        # Single fracture
        ([[0, 1], [0, 0]], [[0], [1]], 1),
        # Single fracture not aligned
        ([[0, 1], [0, 1]], [[0], [1]], np.sqrt(2)),
        # Two fractures separate points
        ([[0, 1, 0, 0], [0, 1, 0, 1]], [[0, 2], [1, 3]], [np.sqrt(2), 1]),
        # Common points reverse order
        ([[0, 1, 0], [0, 1, 1]], [[1, 0], [0, 2]], [np.sqrt(2), 1]),
    ],
)
def test_fracture_length_2d(points_edges_expected):
    points = np.array(points_edges_expected[0])
    edges = np.array(points_edges_expected[1])
    expected = np.array(points_edges_expected[2])
    fl = frac_utils.fracture_length_2d(points, edges)

    assert np.allclose(fl, expected)


@pytest.mark.parametrize(
    "points_edges_tol_expected",
    [
        # No change
        (
            [[0, 1], [0, 0]],  # points
            [[0], [1]],  # edges
            1e-4,  # tol
            None,  # expected_points
            None,  # expected_edges
        ),
        # Merge one point
        (
            [[0, 1, 0, 0], [0, 1, 0, 1]],  # points
            [[0, 2], [1, 3]],  # edges
            1e-4,  # tol
            [[0, 1, 0], [0, 1, 1]],  # expected_points
            [[0, 0], [1, 2]],  # expected_edges
        ),
        # Merge one point variable tolerance
        (
            [[0, 1, 0, 0], [0, 1, 1e-3, 1]],  # points
            [[0, 2], [1, 3]],  # edges
            1e-2,  # tol
            [[0, 1, 0], [0, 1, 1]],  # expected_points
            [[0, 0], [1, 2]],  # expected_edges
        ),
        # There should be no merge
        (
            [[0, 1, 0, 0], [0, 1, 1e-3, 1]],  # points
            [[0, 2], [1, 3]],  # edges
            1e-4,  # tol
            None,  # expected_points
            None,  # expected_edges
        ),
    ],
)
def test_uniquify_points(points_edges_tol_expected):
    points = np.array(points_edges_tol_expected[0])
    edges = np.array(points_edges_tol_expected[1])
    tol = points_edges_tol_expected[2]
    if points_edges_tol_expected[3] is not None:
        expected_points = np.array(points_edges_tol_expected[3])
    else:
        expected_points = points.copy()
    if points_edges_tol_expected[4] is not None:
        expected_edges = np.array(points_edges_tol_expected[4])
    else:
        expected_edges = edges.copy()

    up, ue, deleted = frac_utils.uniquify_points(points, edges, tol=tol)
    assert arrays_equal(up, expected_points)
    assert arrays_equal(ue, expected_edges)
    assert deleted.size == 0


def test_uniquify_points_delete_point_edge():
    p = np.array([[0, 1, 1, 2], [0, 0, 0, 0]])
    # Edge with tags
    e = np.array([[0, 1, 2], [1, 2, 3], [0, 1, 2]])

    up, ue, deleted = frac_utils.uniquify_points(p, e, tol=1e-2)

    p_known = np.array([[0, 1, 2], [0, 0, 0]])
    # Edge with tags
    e_known = np.array([[0, 1], [1, 2], [0, 2]])
    assert arrays_equal(p_known, up)
    assert arrays_equal(e_known, ue)
    assert deleted.size == 1
    assert deleted[0] == 1


class Test_pts_edges_to_linefractures:
    """This class is a collection of tests of the function pts_edges_to_linefractures."""

    @pytest.mark.parametrize(
        "fracs_edges_points",
        [
            # Test conversion of points and edges into line fractures.
            {
                "fracs": [
                    pp.LineFracture([[0, 2], [1, 3]]),
                    pp.LineFracture([[2, 4], [3, 5]]),
                    pp.LineFracture([[0, 4], [1, 5]]),
                ],
                "edges": [[0, 1, 0], [1, 2, 2]],
                "points": [[0, 2, 4], [1, 3, 5]],
            },
            # Test conversion of points and edges with tags into line fractures.
            # The tags are converted into attributes of the line fractures.
            {
                "fracs": [
                    pp.LineFracture([[0, 2], [1, 3]], tags=[-1, 2, -1]),
                    pp.LineFracture([[2, 4], [3, 5]], tags=[1]),
                    pp.LineFracture([[0, 4], [1, 5]]),
                ],
                "edges": [[0, 1, 0], [1, 2, 2], [-1, 1, -1], [2, -1, -1], [-1, -1, -1]],
                "points": [[0, 2, 4], [1, 3, 5]],
            },
            # Test that edges with 0 entries results in an empty fractures list.
            {
                "fracs": [],
                "edges": [[], []],
                "points": [[0, 2, 4], [1, 3, 5]],
            },
        ],
    )
    def test_pts_edges_to_linefractures(self, fracs_edges_points):
        fracs: list[pp.LineFracture] = fracs_edges_points["fracs"]
        edges = np.array(fracs_edges_points["edges"], dtype=int)
        points = np.array(fracs_edges_points["points"])

        converted_fracs = frac_utils.pts_edges_to_linefractures(points, edges)
        assert len(converted_fracs) == len(fracs)
        for frac, converted_frac in zip(fracs, converted_fracs):
            for converted_pt, pt in zip(converted_frac.points(), frac.points()):
                assert np.allclose(converted_pt, pt)
            for converted_tag, tag in zip(converted_frac.tags, frac.tags):
                assert np.all(converted_tag == tag)


@pytest.mark.parametrize(
    "fracs_edges_points",
    [
        # Test conversion of line fractures with tags into pts and edges. The tags are
        # converted into additional rows in the ``edges`` array.
        {
            # Create line fractures with different tag structures. Empty tags first,
            # then nonempty tag/no tags/only nonempty tags/nonempty tags at the end.
            "fracs": [
                pp.LineFracture([[0, 2], [1, 3]], tags=[-1, -1, 2]),
                pp.LineFracture([[2, 4], [3, 5]]),
                pp.LineFracture([[0, 4], [1, 5]], tags=[1, 1]),
                pp.LineFracture([[0, 4], [1, 5]], tags=[2, 2, 2, -1]),
            ],
            # All edges will have the maximal number of tags (4 in this example). The
            # last row consists of only empty tags. This is wanted behavior, as the
            # conversion does not check the tag values.
            "edges": [
                [0, 1, 0, 0],
                [1, 2, 2, 2],
                [-1, -1, 1, 2],
                [-1, -1, 1, 2],
                [2, -1, -1, 2],
                [-1, -1, -1, -1],
            ],
            "points": [[0, 2, 4], [1, 3, 5]],
        },
        # Test conversion of line fractures into points and edges.
        {
            "fracs": [
                pp.LineFracture([[0, 2], [1, 3]]),
                pp.LineFracture([[2, 4], [3, 5]]),
                pp.LineFracture([[0, 4], [1, 5]]),
            ],
            "edges": [[0, 1, 0], [1, 2, 2]],
            "points": [[0, 2, 4], [1, 3, 5]],
        },
        # Test that an empty fractures list results in edges and pts arrays with 0 size.
        {
            "fracs": [],
            "edges": [[], []],
            "points": [[], []],
        },
        # Next 3 cases:
        # Test that points within the tolerance are reduced to a single point.
        # Default tolerance:
        {
            "fracs": [
                pp.LineFracture([[0, 1], [0, 3]]),
                pp.LineFracture([[1, 1], [1, 3 + 1e-9]]),
                pp.LineFracture([[2, 1], [2, 3 + 1e-10]]),
            ],
            "edges": [[0, 2, 3], [1, 1, 1]],
            "points": [[0, 1, 1, 2], [0, 3, 1, 2]],
        },
        # Default tolerance, not within tolerance.
        {
            "fracs": [
                pp.LineFracture([[0, 1], [1, 3]]),
                pp.LineFracture([[2, 3], [1, 3 + 1e-5]]),
                pp.LineFracture([[4, 5], [1, 3 + 1e-5]]),
            ],
            "edges": [[0, 2, 4], [1, 3, 5]],
            "points": [
                [0, 1, 2, 3, 4, 5],
                [1, 3, 1, 3 + 1e-5, 1, 3 + 1e-5],
            ],
        },
        # Custom tolerance
        {
            "fracs": [
                pp.LineFracture([[0, 1], [0, 3]]),
                pp.LineFracture([[1, 1], [1, 3 + 1e-6]]),
                pp.LineFracture([[2, 1], [2, 3 + 1e-7]]),
            ],
            "edges": [[0, 2, 3], [1, 1, 1]],
            "points": [[0.0, 1.0, 1.0, 2.0], [0.0, 3.0, 1.0, 2.0]],
            "tol": 1e-5,
        },
    ],
)
def test_linefractures_to_pts_edges(fracs_edges_points):
    fracs: list[pp.LineFracture] = fracs_edges_points["fracs"]
    expected_edges = np.array(fracs_edges_points["edges"], dtype=int)
    expected_points = np.array(fracs_edges_points["points"])
    if tol := fracs_edges_points.get("tol", None):
        converted_pts, converted_edges = frac_utils.linefractures_to_pts_edges(
            fracs, tol=tol
        )
    else:
        converted_pts, converted_edges = frac_utils.linefractures_to_pts_edges(fracs)
    assert np.allclose(converted_pts, expected_points, atol=1e-20)
    assert np.allclose(converted_edges, expected_edges, atol=1e-20)
