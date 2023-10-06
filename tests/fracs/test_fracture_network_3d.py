"""Testing functionality related to FractureNetwork3d."""
import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.arrays import compare_arrays
from porepy.fracs.plane_fracture import PlaneFracture
from porepy.grids.standard_grids.utils import unit_domain


@pytest.mark.parametrize(
    "points_expected",
    [
        # Completely outside lower.
        {
            "points": [
                [-2.0, -1.0, -1.0, -2.0],
                [0.5, 0.5, 0.5, 0.5],
                [0.0, 0.0, 1.0, 1.0],
            ],
            "expected_num_fractures": 0,
        },
        # Outside west bottom.
        {
            "points": [
                [-0.5, 0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5, 0.5],
                [-1.5, -1.5, -0.5, -0.5],
            ],
            "expected_num_fractures": 0,
        },
        # Intersect one.
        {
            "points": [
                [-0.5, 0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5, 0.5],
                [0.2, 0.2, 0.8, 0.8],
            ],
            "expected_points": [
                [0.0, 0.5, 0.5, 0],
                [0.5, 0.5, 0.5, 0.5],
                [0.2, 0.2, 0.8, 0.8],
            ],
            "expected_num_fractures": 1,
        },
        # Full incline.
        {
            "points": [
                [-0.5, 0.5, 0.5, -0.5],
                [0.5, 0.5, 1.5, 1.5],
                [-0.5, -0.5, 1, 1],
            ],
            "expected_points": [
                [0.0, 0.5, 0.5, 0],
                [5.0 / 6, 5.0 / 6, 1, 1],
                [0.0, 0.0, 0.25, 0.25],
            ],
            "expected_num_fractures": 1,
        },
        # Incline in plane.
        {
            "points": [
                [-0.5, 0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5, 0.5],
                [0.0, -0.5, 0.5, 1.0],
            ],
            "expected_points": [
                [0.0, 0.5, 0.5, 0],
                [0.5, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.5, 0.75],
            ],
            "expected_num_fractures": 1,
        },
        # Intersect two same.
        {
            "points": [
                [-0.5, 1.5, 1.5, -0.5],
                [0.5, 0.5, 0.5, 0.5],
                [0.2, 0.2, 0.8, 0.8],
            ],
            "expected_points": [
                [0.0, 1, 1, 0],
                [0.5, 0.5, 0.5, 0.5],
                [0.2, 0.2, 0.8, 0.8],
            ],
            "expected_num_fractures": 1,
        },
    ],
)
def test_impose_external_boundary(points_expected):
    """Test of algorithm for constraining a fracture a bounding box.

    Since that algorithm uses fracture intersection methods, the tests functions as
    partial test for the wider fracture intersection framework as well. Full tests
    of the latter are too time consuming to fit into a unit test.

    Now the boundary is defined as set of "fake" fractures, all fracture network
    have 2*dim additional fractures (hence the + 6 in the assertions)

    """

    points = np.array(points_expected["points"])
    expected_num_fractures = points_expected["expected_num_fractures"]
    fracture = pp.PlaneFracture(points, check_convexity=False)
    network = pp.create_fracture_network([fracture])
    network.impose_external_boundary(unit_domain(3))
    assert len(network.fractures) == (6 + expected_num_fractures)
    p_comp = network.fractures[0].pts
    if expected_points := points_expected.get("expected_points", None):
        assert compare_arrays(p_comp, np.array(expected_points), sort=True, tol=1e-5)


def test_impose_external_boundary_bounding_box():
    # Test of method FractureNetwork.bounding_box() when an external
    # boundary is added. Thus double as test of this adding.
    fracture = PlaneFracture(
        np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]]),
        check_convexity=False,
    )
    network = pp.create_fracture_network([fracture])

    external_box = {"xmin": -1, "xmax": 2, "ymin": -1, "ymax": 2, "zmin": -1, "zmax": 2}
    domain_to_impose = pp.Domain(bounding_box=external_box.copy())
    network.impose_external_boundary(domain=domain_to_impose)

    assert network.domain.bounding_box == external_box


@pytest.mark.parametrize(
    "fracs_expected",
    [
        {
            "fracs": [
                PlaneFracture(
                    np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1]]),
                    check_convexity=False,
                )
            ],
            "expected": dict(xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1),
        },
        {
            "fracs": [
                PlaneFracture(
                    np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]),
                    check_convexity=False,
                )
            ],
            "expected": dict(xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=0),
        },
        # Test two fractures
        {
            "fracs": [
                PlaneFracture(
                    np.array([[0, 2, 2, 0], [0, 0, 1, 1], [0, 0, 1, 1]]),
                    check_convexity=False,
                ),
                PlaneFracture(
                    np.array([[0, 1, 1, 0], [0, 0, 1, 1], [-1, -1, 1, 1]]),
                    check_convexity=False,
                ),
            ],
            "expected": dict(xmin=0, xmax=2, ymin=0, ymax=1, zmin=-1, zmax=1),
        },
    ],
)
def test_bounding_box(fracs_expected):
    # Test of method FractureNetwork.bounding_box() to inquire about network extent.
    fracs = fracs_expected["fracs"]
    expected = fracs_expected["expected"]
    network = pp.create_fracture_network(fracs)
    d = network.bounding_box()
    assert d == expected
