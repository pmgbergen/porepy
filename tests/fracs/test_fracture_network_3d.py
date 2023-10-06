"""Testing functionality related to FractureNetwork3d.

Content:
    - Tests of the methods to impose an external boundary to the domain.
    - Test of the functionality to determine the mesh size.

"""
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


def test_mesh_size_determination():
    """3d domain. One fracture which extends to the boundary.

    Check that the mesh size functionality in FractureNetwork3d determines the
    expected mesh sizes by comparing against hard-coded values.

    """

    f_1 = np.array([[1, 5, 5, 1], [1, 1, 1, 1], [1, 1, 3, 3]])
    f_set = [pp.PlaneFracture(f_1)]
    domain = pp.Domain(
        {"xmin": 0, "ymin": 0, "zmin": 0, "xmax": 5, "ymax": 5, "zmax": 5}
    )
    on_boundary = np.array(
        [False, False, False, False, True, True, True, True, True, True, True, True]
    )
    # Mesh size arguments
    mesh_size_min = 0.1
    mesh_size_frac = 0.1
    mesh_size_bound = 2
    # Define a fracture network, impose the boundary, find and split intersections.
    # These operations mirror those performed in FractureNetwork3d.mesh() up to the
    # point where the mesh size is determined and auxiliary points used to enforce the
    # mesh size are inserted.
    network = pp.create_fracture_network(f_set)
    network.impose_external_boundary(domain)
    network.find_intersections()
    network.split_intersections()
    network._insert_auxiliary_points(
        mesh_size_frac=mesh_size_frac,
        mesh_size_min=mesh_size_min,
        mesh_size_bound=mesh_size_bound,
    )
    mesh_size = network._determine_mesh_size(point_tags=on_boundary)

    # To get the mesh size, we need to access the decomposition of the domain
    decomp = network.decomposition

    # Many of the points should have mesh size of 2, adjust the exceptions below
    mesh_size_known = np.full(decomp["points"].shape[1], mesh_size_bound, dtype=float)

    # Find the points on the fracture
    fracture_poly = np.where(np.logical_not(network.tags["boundary"]))[0]
    # There is only one fracture. This is a sanity check.
    assert fracture_poly.size == 1

    # Points on the fracture should have been assigned fracture mesh size.
    fracture_points = decomp["polygons"][fracture_poly[0]][0]
    mesh_size_known[fracture_points] = mesh_size_frac

    # Two of the domain corners are close enough to the fracture to have their mesh size
    # modified from the default boundary value.
    origin = np.zeros((3, 1))
    _, ind = pp.utils.setmembership.ismember_rows(origin, decomp["points"])
    mesh_size_known[ind] = np.sqrt(3)

    corner = np.array([5, 0, 0]).reshape((3, 1))
    _, ind = pp.utils.setmembership.ismember_rows(corner, decomp["points"], sort=False)
    mesh_size_known[ind] = np.sqrt(2)

    assert np.all(np.isclose(mesh_size, mesh_size_known))
