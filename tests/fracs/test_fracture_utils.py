"""Testing functionality related to fracture utils module.

Created on Mon Mar 26 10:12:47 2018

@author: eke001
"""

import numpy as np
import pytest

import porepy as pp
from porepy import frac_utils


class Test_pts_edges_to_linefractures:
    """This class is a collection of tests of the function
    pts_edges_to_linefractures."""

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
