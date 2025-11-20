"""Testing functionality related to FractureNetwork3d.

Content:
    - Tests of the methods to impose an external boundary to the domain.
    - Test of the functionality to determine the mesh size.
    - Test of meshing.

"""

from collections import namedtuple

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.domains import unit_cube_domain as unit_domain
from porepy.applications.test_utils.arrays import compare_arrays
from porepy.fracs.plane_fracture import PlaneFracture


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
    fractures_kept, fractures_deleted = network.impose_external_boundary(unit_domain(3))
    assert len(fractures_kept) == expected_num_fractures
    assert len(fractures_deleted) == 1 - expected_num_fractures
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

    Check that the mesh size functionality in FractureNetwork3d determines the expected
    mesh sizes by comparing against hard-coded values.

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

    # To get the mesh size, we need to access the decomposition of the domain.
    decomp = network.decomposition

    # Many of the points should have mesh size of 2, adjust the exceptions below.
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
    _, ind = pp.array_operations.ismember_columns(origin, decomp["points"])
    mesh_size_known[ind] = np.sqrt(3)

    corner = np.array([5, 0, 0]).reshape((3, 1))
    _, ind = pp.array_operations.ismember_columns(corner, decomp["points"], sort=False)
    mesh_size_known[ind] = np.sqrt(2)

    assert np.all(np.isclose(mesh_size, mesh_size_known))


# Named tuple used to identify intersections of fractures by their parent fractures
# and their coordinates. The coordinates should describe the full intersection line,
# independent of how many 1d grids are created from it.
IntersectionInfo = namedtuple("IntersectionInfo", ["parent_0", "parent_1", "coord"])
IntersectionInfo3Frac = namedtuple(
    "IntersectionInfo3Frac", ["parent_0", "parent_1", "parent_2", "coord"]
)


def _standard_domain(modify: bool = False) -> dict | pp.Domain:
    """Create a standard domain for testing purposes."""
    bbox = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
    if modify:
        return bbox
    else:
        domain = pp.Domain(bbox)
        return domain


def _create_mdg(
    fractures, domain=None, mesh_args: dict | None = None, constraints=None
) -> pp.MixedDimensionalGrid:
    """Create a mixed-dimensional grid from a list of fractures."""
    if mesh_args is None:
        mesh_args = {"mesh_size_bound": 1, "mesh_size_frac": 1, "mesh_size_min": 0.1}
    network = pp.create_fracture_network(fractures, domain=domain)
    if constraints is None:
        mdg = network.mesh(mesh_args)
    else:
        mdg = network.mesh(mesh_args, constraints=constraints)
    return mdg


def check_mdg(
    mdg: pp.MixedDimensionalGrid,
    domain: pp.Domain,
    fractures=None,
    isect_line=None,
    isect_pt=None,
    expected_num_1d_grids=0,
    expected_num_0d_grids=0,
):
    """Validate the generated mixed-dimensional grid based on the expected grid
    properties.

    Parameters:
        mdg: Grid to be validated
        domain: Domain of the grid
        fractures: List of fractures in the domain
        isect_line: List of expected intersection lines
        isect_pt: List of expected intersection points
        expected_num_1d_grids: Expected number of 1d grids
        expected_num_0d_grids: Expected number of 0d grids

    """
    if fractures is None:
        fractures = []
    if isect_line is None:
        isect_line = []
    if isect_pt is None:
        isect_pt = []

    def compare_bounding_boxes(box_1, box_2):
        # Helper method to compare two bounding boxes
        for k, v in box_1.items():
            if np.abs(box_2[k] - v) > 1e-10:
                return False
        return True

    # Get the bounding box of the nodes of the 3d grid.
    bb = pp.domain.bounding_box_of_point_cloud(mdg.subdomains(dim=3)[0].nodes)
    assert compare_bounding_boxes(bb, domain.bounding_box)

    # Check that the number of subdomain grids are as expected
    assert len(fractures) == len(mdg.subdomains(dim=2))
    assert expected_num_1d_grids == len(mdg.subdomains(dim=1))
    assert expected_num_0d_grids == len(mdg.subdomains(dim=0))

    # Loop over all fractures, find the grid with the corresponding frac_num. Check
    # that their bounding boxes are the same.
    for fi, f in enumerate(fractures):
        for sd in mdg.subdomains(dim=2):
            if sd.frac_num == fi and not isinstance(f, pp.EllipticFracture):
                assert compare_bounding_boxes(
                    pp.domain.bounding_box_of_point_cloud(f.pts),
                    pp.domain.bounding_box_of_point_cloud(sd.nodes),
                )

    # The bounding boxes of the constructed 1d grids will be compared to the expected
    # values. To construct the grid bounding boxes, we will loop over all 1d grids and
    # update the bounding box of the corresponding intersection grid.

    # Bounding box for the computed intersection grids. The key is a tuple of the
    # parent fractures, the value is a dictionary with the bounding box and the
    # intersection line which should result from this combination of fractures.
    ii_computed_box: dict[tuple[int, int] | tuple[int, int, int], dict] = {}

    for isect in isect_line:
        # The initial assumption is that the bounding box is empty, signified by min and
        # max values being inf and -inf, respectively (meaningful values will be
        # inserted once we start updating the boxes).
        inital_box = {
            "xmin": np.inf,
            "xmax": -np.inf,
            "ymin": np.inf,
            "ymax": -np.inf,
            "zmin": np.inf,
            "zmax": -np.inf,
        }
        if isinstance(isect, IntersectionInfo):
            # This is an intersection of two fractures. Sort the parents to get a
            # unique key.
            p_0, p_1 = sorted([isect.parent_0, isect.parent_1])

            ii_computed_box[(p_0, p_1)] = {
                "isect": isect,
                "coord": inital_box,
            }
        elif isinstance(isect, IntersectionInfo3Frac):
            # This is an intersection of three fractures.
            p_0, p_1, p_2 = sorted([isect.parent_0, isect.parent_1, isect.parent_2])
            ii_computed_box[(p_0, p_1, p_2)] = {
                "coord": inital_box,
                "isect": isect,
            }

    def update_box(box, update):
        # Helper method to update a bounding box.
        for k in ["xmin", "ymin", "zmin"]:
            box["coord"][k] = min(box["coord"][k], update[k])
        for k in ["xmax", "ymax", "zmax"]:
            box["coord"][k] = max(box["coord"][k], update[k])

    # Loop over the 1d domains, and update the bounding boxes of the intersection grid.
    # Since a fracture intersection may be shared by several 1d grids (exactly how this
    # is handled is not clear to EK at the moment, but it surely cannot be wrong to
    # allow for more than one 1d grid to share an intersection), we need to update the
    # bounding box of the intersection grid for each 1d grid that shares the
    # intersection.
    for sd in mdg.subdomains(dim=1):
        # Get the parents (fractures) of the 1d grid
        neighs = mdg.neighboring_subdomains(sd, only_higher=True)
        # bounding box of the 1d grid
        box = pp.domain.bounding_box_of_point_cloud(sd.nodes)

        # Update the bounding box of the intersection grid (identified by the parent
        # fractures).
        if len(neighs) == 2:
            f_0, f_1 = sorted([neighs[0].frac_num, neighs[1].frac_num])
            update_box(ii_computed_box[(f_0, f_1)], box)

        elif len(neighs) == 3:
            f_0, f_1, f_2 = sorted(
                [neighs[0].frac_num, neighs[1].frac_num, neighs[2].frac_num]
            )
            update_box(ii_computed_box[(f_0, f_1, f_2)], box)

    # Check that the bounding boxes of the intersection grids correspond to the
    # expected values.
    for val in ii_computed_box.values():
        coord = val["coord"]
        isect = val["isect"]
        assert compare_bounding_boxes(
            coord, pp.domain.bounding_box_of_point_cloud(isect.coord)
        )

    # For each 0d grid, check that it is present as an expected intersection point.
    for sd in mdg.subdomains(dim=0):
        found = False
        for p in isect_pt:
            if np.allclose(p, sd.cell_centers):
                found = True
                break
        assert found
    # For each intersection point, check that it is present as a 0d grid.
    for p in isect_pt:
        found = False
        for sd in mdg.subdomains(dim=0):
            if np.allclose(p, sd.cell_centers):
                found = True
                break
        assert found


@pytest.fixture(scope="module")
def unit_box() -> pp.Domain:
    """Create a unit box domain for testing purposes."""
    bbox = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
    domain = pp.Domain(bbox)
    return domain


def _verify_points_in_fracture(points: np.ndarray, fracture: pp.PlaneFracture):
    """Verify that points lie in the plane of the fracture.

    Parameters:
        points: Points to verify.
        fracture: Fracture defining the plane.

    """
    dist, *_ = pp.distances.points_polygon(points, fracture.pts)
    assert np.allclose(dist, 0.0, atol=1e-6)


def _verify_points_in_line(points: np.ndarray, start: np.ndarray, end: np.ndarray):
    """Verify that points lie on the line defined by line_point and line_dir.

    Parameters:
        points: Points to verify.
        start: Start point of the line.
        end: End point of the line.

    """
    d, *_ = pp.distances.points_segments(
        points, start.reshape((3, 1)), end.reshape((3, 1))
    )
    assert np.allclose(d, 0.0, atol=1e-6)


@pytest.mark.parametrize(
    "x_coord, is_constraint",
    [
        # No fractures.
        ([], []),
        # Fracture fully inside the domain, not a constraint.
        ([0.2], [False]),
        # Fracture fully inside the domain, is a constraint.
        ([0.2], [True]),
        # Fracture on the boundary, not a constraint.
        ([0.0], [False]),
        # Fracture on the boundary, is a constraint.
        ([0.0], [True]),
        # Two fracutres fully inside the domain.
        ([0.2, 0.5], [False, False]),
        # Fracture outside the domain, not a constraint.
        ([-0.5], [False]),
        # Fracture outside the domain, is a constraint.
        ([-0.5], [True]),
        # One fracture inside, one outside, none a constraint.
        ([0.2, -0.5], [False, False]),
        # One fracture inside, one outside. Outside fracture first on the list.
        ([-0.5, 0.2], [False, False]),
        # One fracture inside, one outside, both constraints.
        ([0.2, -0.5], [True, True]),
        # One fracture inside, one outside. Constraint first on the list.
        ([-0.5, 0.2], [True, False]),
    ],
)
def test_meshing_no_intersections(
    x_coord: list[float], is_constraint: list[bool], unit_box: pp.Domain
):
    """Test meshing of a single fracture without intersections.

    Parameters:
        x_coord: x-coordinate of the vertical fracture.
        is_constraint: Whether the fracture is a constraint.
        unit_box: Unit box domain.

    """
    is_fracture = len(x_coord) * [True]

    fractures = []

    for i, x in enumerate(x_coord):
        if is_constraint[i] or x >= 1.0 or x <= 0.0:
            is_fracture[i] = False

    for i, x in enumerate(x_coord):
        fractures.append(
            pp.PlaneFracture(
                np.array([[x, x, x, x], [0.2, 0.8, 0.8, 0.2], [0.2, 0.2, 0.8, 0.8]])
            )
        )

    network = pp.create_fracture_network(fractures, unit_box)
    constraints = np.where(is_constraint)[0]
    # Generate a mixed-dimensional grid with a grid as coarse as possible.
    mdg = network.mesh(
        mesh_args={"mesh_size_bound": 1, "mesh_size_frac": 1, "mesh_size_min": 0.5},
        constraints=constraints,
    )

    assert len(mdg.subdomains(dim=2)) == sum(is_fracture)
    assert len(mdg.subdomains(dim=1)) == 0
    assert len(mdg.subdomains(dim=0)) == 0

    # Verify that the 2d grids lie in the fracture planes, but only for the fractures
    # that should be represented by actual grids.
    internal_fractures = [f for i, f in enumerate(fractures) if is_fracture[i]]
    for i, f in enumerate(internal_fractures):
        # Find the corresponding 2d grid
        for sd in mdg.subdomains(dim=2):
            if sd.frac_num == i:
                # Check that all nodes of the grid lie in the fracture plane
                _verify_points_in_fracture(sd.nodes, f)


@pytest.mark.parametrize(
    "fracture_present, fracture_constraint, elliptic",
    [
        # Two fractures, zero to two constraints. All are PlanarFracture, no elliptic.
        ([True, True, False], [False, False, False], [False, False, False]),
        ([True, True, False], [True, False, False], [False, False, False]),
        ([True, True, False], [True, True, False], [False, False, False]),
        # Two fractures, zero to two constraints. One or two elliptic fractures.
        ([True, True, False], [False, False, False], [True, False, False]),
        ([True, True, False], [False, False, False], [True, True, False]),
        ([True, True, False], [True, False, False], [True, False, False]),
        ([True, True, False], [True, True, False], [True, True, False]),
        # Three fractures, zero to three constraints. No elliptic fractures.
        ([True, True, True], [False, False, False], [False, False, False]),
        ([True, True, True], [True, False, False], [False, False, False]),
        ([True, True, True], [True, True, False], [False, False, False]),
        ([True, True, True], [True, True, True], [False, False, False]),
        # Three fractures, one is a constraint. One to three elliptic fractures.
        ([True, True, True], [True, False, False], [True, False, False]),
        ([True, True, True], [True, True, False], [True, True, False]),
        ([True, True, True], [True, True, True], [True, True, True]),
    ],
)
def test_cross_intersection(
    fracture_present: list[bool],
    fracture_constraint: list[bool],
    elliptic: list[bool],
    unit_box: pp.Domain,
):
    """Test meshing of a cross intersection of 1-3 fractures"""
    fractures = []
    is_fracture = [a and not b for a, b in zip(fracture_present, fracture_constraint)]

    base_0 = np.array([0.2, 0.2, 0.8, 0.8])
    base_1 = np.array([0.2, 0.8, 0.8, 0.2])
    base_2 = np.array([0.5, 0.5, 0.5, 0.5])

    candidate_planar = [
        pp.PlaneFracture(np.vstack((base_2, base_0, base_1))),
        pp.PlaneFracture(np.vstack((base_1, base_2, base_0))),
        pp.PlaneFracture(np.vstack((base_0, base_1, base_2))),
    ]
    candidate_elliptic = [
        pp.EllipticFracture(
            center=np.array([0.5, 0.5, 0.5]),
            major_axis=0.3,
            minor_axis=0.3,
            major_axis_angle=0,
            strike_angle=np.pi / 2,
            dip_angle=np.pi / 2,
        ),
        pp.EllipticFracture(
            center=np.array([0.5, 0.5, 0.5]),
            major_axis=0.3,
            minor_axis=0.3,
            major_axis_angle=0,
            strike_angle=0,
            dip_angle=np.pi / 2,
        ),
        pp.EllipticFracture(
            center=np.array([0.5, 0.5, 0.5]),
            major_axis=0.3,
            minor_axis=0.3,
            major_axis_angle=0,
            strike_angle=0,
            dip_angle=0,
        ),
    ]

    fractures = []
    for i in range(3):
        if fracture_present[i]:
            if elliptic[i]:
                fractures.append(candidate_elliptic[i])
            else:
                fractures.append(candidate_planar[i])

    network = pp.create_fracture_network(fractures, unit_box)
    constraints = np.where(fracture_constraint)[0]
    # Generate a mixed-dimensional grid with a grid as coarse as possible.
    mdg = network.mesh(
        mesh_args={"mesh_size_bound": 1, "mesh_size_frac": 1, "mesh_size_min": 0.5},
        constraints=constraints,
    )
    num_fractures = sum(is_fracture)
    assert len(mdg.subdomains(dim=2)) == num_fractures
    if num_fractures == 2:
        if sum(fracture_constraint) == 0:
            # No constraints, the two fractures intersect in a line.
            assert len(mdg.subdomains(dim=1)) == 1
            assert len(mdg.subdomains(dim=0)) == 0
        else:
            # The two fractures intersect in a line, but this line will be split by the
            # third fracture (the constraint). The splitting will introduce two 1d grids
            # and a 0d grid.
            assert len(mdg.subdomains(dim=1)) == 2
            assert len(mdg.subdomains(dim=0)) == 1
    elif num_fractures == 3:
        assert len(mdg.subdomains(dim=1)) == 6
        assert len(mdg.subdomains(dim=0)) == 1

    for i, f in enumerate(fractures):
        # Find the corresponding 2d grid
        for sd in mdg.subdomains(dim=2):
            if sd.frac_num == i:
                # Check that all nodes of the grid lie in the fracture plane. Use the
                # planar representation also for elliptic fractures, since the latter
                # has no vertexes on which the comparison can be based. The test that
                # the nodes of an elliptic fracture lie on the ellipse, and not only on
                # a bounding polygon (which is whath the below replacement in effect
                # will do) is done in test_elliptic_fracture.py
                _verify_points_in_fracture(sd.nodes, candidate_planar[i])

    def _check_line_intersection(frac_num_0, frac_num_1):
        # Check the intersection line between fracture frac_num_0 and frac_num_1.

        # Due to the fracture construction, the intersection line is along the
        # coordinate axis not present in either fracture. The other coordinates are 0.5.
        start = np.array([0.5, 0.5, 0.5])
        end = np.array([0.5, 0.5, 0.5])
        dim_not_present = np.setdiff1d([0, 1, 2], [frac_num_0, frac_num_1])[0]
        start[dim_not_present] = 0.2
        end[dim_not_present] = 0.8

        for sd in mdg.subdomains(dim=1):
            neighs = mdg.neighboring_subdomains(sd, only_higher=True)
            if (
                neighs[0].frac_num == frac_num_0 and neighs[1].frac_num == frac_num_1
            ) or (
                neighs[0].frac_num == frac_num_1 and neighs[1].frac_num == frac_num_0
            ):
                _verify_points_in_line(sd.nodes, start, end)

    if is_fracture[0] and is_fracture[1]:
        _check_line_intersection(0, 1)
    if is_fracture[0] and is_fracture[2]:
        _check_line_intersection(0, 2)
    if is_fracture[1] and is_fracture[2]:
        _check_line_intersection(1, 2)

    if num_fractures == 3:
        # Check the intersection line between fracture 0 and 2
        assert np.allclose(
            mdg.subdomains(dim=0)[0].cell_centers, np.array([[0.5], [0.5], [0.5]])
        )


@pytest.mark.parametrize(
    "z_coord",
    [
        ([0.4, 0.6]),  # Full match in the z-range
        ([0.3, 0.5]),  # Partial overlap, the two fractures overlap in z = [0.4, 0.5]
        ([0.3, 0.7]),  # Fracture 1 fully contains fracture 0 in z-direction
    ],
)
@pytest.mark.parametrize("is_t", [True, False])
@pytest.mark.parametrize(
    "is_constraint",
    [
        [False, False],  # No constraints
        [True, False],  # One fracture is a constraint
        [True, True],  # Both fractures are constraints
    ],
)
def test_t_l_intersection(
    z_coord: list[float], is_t: bool, is_constraint: list[bool], unit_box: pp.Domain
):
    fracture_0 = pp.PlaneFracture(
        np.array([[0.2, 0.8, 0.8, 0.2], [0.5, 0.5, 0.5, 0.5], [0.4, 0.4, 0.6, 0.6]])
    )
    x_coord = 0.5 if is_t else 0.2
    fracture_1 = pp.PlaneFracture(
        np.array(
            [
                [x_coord, x_coord, x_coord, x_coord],
                [0.2, 0.5, 0.5, 0.2],
                [z_coord[0], z_coord[0], z_coord[1], z_coord[1]],
            ]
        )
    )
    fractures = [fracture_0, fracture_1]
    network = pp.create_fracture_network(fractures, unit_box)
    constraints = np.where(is_constraint)[0]
    # Generate a mixed-dimensional grid with a grid as coarse as possible.
    mdg = network.mesh(
        mesh_args={"mesh_size_bound": 1, "mesh_size_frac": 1, "mesh_size_min": 0.5},
        constraints=constraints,
    )
    num_fracs = 2 - sum(is_constraint)
    assert len(mdg.subdomains(dim=2)) == num_fracs
    assert len(mdg.subdomains(dim=1)) == (1 if num_fracs == 2 else 0)
    assert len(mdg.subdomains(dim=0)) == 0

    if num_fracs == 2:
        # Check the intersection line between fracture 0 and 1.

        # Due to the fracture construction, the intersection line is along the
        # z-axis if T-intersection, otherwise along the x-axis. The other coordinates
        # are 0.5.
        start = np.array([x_coord, 0.5, z_coord[0]])
        end = np.array([x_coord, 0.5, z_coord[1]])
        sd = mdg.subdomains(dim=1)[0]
        _verify_points_in_line(sd.nodes, start, end)


@pytest.mark.parametrize(
    "is_constraint",
    [
        [False, False, False],  # No constraints. Three intersection grids.
        [True, False, False],  # Tall fracture is constraint. One intersection grid.
        [False, False, True],  # Short fracture is constraint. One intersection grids.
        [True, False, True],  # Both fractures are constraints. No intersection grids.
        [True, True, True],  # All fractures are constraints. No intersection grids.
    ],
)
@pytest.mark.parametrize("dfn", [True, False])
def test_three_fractures_intersecting_along_line(
    is_constraint: list[bool], dfn: bool, unit_box: pp.Domain
):
    """Test meshing of three fractures intersecting along a line.

    Parameters:
        is_constraint: Whether each fracture is a constraint.
        dfn: Whether to use DFN-style meshing.
        unit_box: Unit box domain.

    """
    fracture_0 = pp.PlaneFracture(
        np.array([[0.2, 0.8, 0.8, 0.2], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])
    )
    fracture_1 = pp.PlaneFracture(
        np.array([[0.5, 0.5, 0.5, 0.5], [0.2, 0.8, 0.8, 0.2], [0.2, 0.2, 0.8, 0.8]])
    )
    fracture_2 = pp.PlaneFracture(
        np.array([[0.2, 0.8, 0.8, 0.2], [0.2, 0.8, 0.8, 0.2], [0.3, 0.3, 0.7, 0.7]])
    )
    fractures = [fracture_0, fracture_1, fracture_2]
    network = pp.create_fracture_network(fractures, unit_box)
    constraints = np.where(is_constraint)[0]
    # Generate a mixed-dimensional grid with a grid as coarse as possible.
    mdg = network.mesh(
        mesh_args={"mesh_size_bound": 1, "mesh_size_frac": 1, "mesh_size_min": 0.5},
        constraints=constraints,
        dfn=dfn,
    )
    num_fracs = 3 - sum(is_constraint)
    assert len(mdg.subdomains(dim=2)) == num_fracs

    # All fractures cross along the segment 0.3 < z < 0.7. Fractures 0 and 1 also cross
    # along the two segmentnts 0.2 < z < 0.3 and 0.7 < z < 0.8. Thus, if all fractures
    # are truly fractures (not constraints), there will be three intersection grids and
    # two intersection points. The same applies if only fracture 2 is a constraint, as
    # the presence of the constraint will split the intersection line between fractures
    # 0 and 1, even though the constraint is not present as a fracture (to be clear,
    # yes, this is a bit awkward, but merging the interseciton lines again would be much
    # more cumbersome to implement).
    #
    # In other cases where only two are real fractures, there will be a single
    # intersection grid (along 0.3 < z < 0.7) and no intersection points. If only one or
    # none are real fractures, there will be no intersection grids or points.
    if num_fracs == 3 or (num_fracs == 2 and is_constraint[2]):
        expected_1d_grids = 3
        expected_0d_grids = 2
    elif num_fracs == 2:
        expected_1d_grids = 1
        expected_0d_grids = 0
    else:
        expected_1d_grids = 0
        expected_0d_grids = 0
    assert len(mdg.subdomains(dim=1)) == expected_1d_grids
    assert len(mdg.subdomains(dim=0)) == expected_0d_grids

    if num_fracs >= 2:
        # Check the intersection line between fracture 0 and 1.

        # Due to the fracture construction, the intersection line is along the
        # z-axis . The other coordinates are 0.5
        if is_constraint[0] or is_constraint[1]:
            start = np.array([0.5, 0.5, 0.3])
            end = np.array([0.5, 0.5, 0.7])
        else:
            start = np.array([0.5, 0.5, 0.2])
            end = np.array([0.5, 0.5, 0.8])
        sd = mdg.subdomains(dim=1)[0]
        _verify_points_in_line(sd.nodes, start, end)


@pytest.mark.parametrize(
    "x_min, x_max, is_constraint",
    [
        (-0.5, 0.5, False),  # Fracture hits the boundary, not a constraint.
        (-0.5, 0.5, True),  # Fracture hits the boundary, is a constraint.
        (-0.5, 1.5, False),  # Fracture extends beyond the boundary, not a constraint.
        (-0.5, 1.5, True),  # Fracture extends beyond the boundary, is a constraint.
        (0.0, 1.0, False),  # Fracture exactly on the boundary, not a constraint.
        (0.0, 1.0, True),  # Fracture exactly on the boundary, is a constraint.
    ],
)
def test_fracture_hits_boundary(x_min, x_max, is_constraint, unit_box: pp.Domain):
    """Test meshing of a fracture hitting the domain boundary.

    Parameters:
        x_min: Minimum x-coordinate of the fracture.
        x_max: Maximum x-coordinate of the fracture.
        is_constraint: Whether the fracture is a constraint.
        unit_box: Unit box domain.

    """
    fracture = pp.PlaneFracture(
        np.array(
            [[x_min, x_max, x_max, x_min], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]]
        )
    )
    fractures = [fracture]
    network = pp.create_fracture_network(fractures, unit_box)
    constraints = np.array([0]) if is_constraint else None
    # Generate a mixed-dimensional grid with a grid as coarse as possible.
    mdg = network.mesh(
        mesh_args={"mesh_size_bound": 1, "mesh_size_frac": 1, "mesh_size_min": 0.5},
        constraints=constraints,
    )
    num_fracs = 1 - (1 if is_constraint else 0)
    assert len(mdg.subdomains(dim=2)) == num_fracs
    assert len(mdg.subdomains(dim=1)) == 0
    assert len(mdg.subdomains(dim=0)) == 0

    truncated_x_min = max(x_min, 0.0)
    truncated_x_max = min(x_max, 1.0)
    truncated_fracture = pp.PlaneFracture(
        np.array(
            [
                [truncated_x_min, truncated_x_max, truncated_x_max, truncated_x_min],
                [0.5, 0.5, 0.5, 0.5],
                [0.2, 0.2, 0.8, 0.8],
            ]
        )
    )

    if num_fracs == 1:
        # Check that all nodes of the grid lie in the fracture plane
        sd = mdg.subdomains(dim=2)[0]
        _verify_points_in_fracture(sd.nodes, truncated_fracture)


@pytest.mark.parametrize(
    "extend_beyond",
    [False, True],  # Fracture hits corner line or extends beyond it.
)
@pytest.mark.parametrize(
    "is_constraint",
    [False, True],  # Fracture is a constraint or not.
)
def test_fracture_hits_domain_corner_line(
    extend_beyond: bool, is_constraint: bool, unit_box: pp.Domain
):
    """Test meshing of a fracture hitting the domain corner line.

    Parameters:
        extend_beyond: Whether the fracture extends beyond the domain corner line.
        is_constraint: Whether the fracture is a constraint.
        unit_box: Unit box domain.

    """
    x_min = 0.5
    y_min = 0.5
    x_max = 1.5 if extend_beyond else 1.0
    y_max = 1.5 if extend_beyond else 1.0
    z_min = 0.2
    z_max = 0.8
    fracture = pp.PlaneFracture(
        np.array(
            [
                [x_min, x_max, x_max, x_min],
                [y_min, y_max, y_max, y_min],
                [z_min, z_min, z_max, z_max],
            ]
        )
    )
    fractures = [fracture]
    network = pp.create_fracture_network(fractures, unit_box)
    constraints = np.array([0]) if is_constraint else None
    # Generate a mixed-dimensional grid with a grid as coarse as possible.
    mdg = network.mesh(
        mesh_args={"mesh_size_bound": 1, "mesh_size_frac": 1, "mesh_size_min": 0.5},
        constraints=constraints,
    )
    num_fracs = 1 - (1 if is_constraint else 0)
    assert len(mdg.subdomains(dim=2)) == num_fracs
    assert len(mdg.subdomains(dim=1)) == 0
    assert len(mdg.subdomains(dim=0)) == 0

    truncated_x_max = min(x_max, 1.0)
    truncated_y_max = min(y_max, 1.0)
    truncated_fracture = pp.PlaneFracture(
        np.array(
            [
                [x_min, truncated_x_max, truncated_x_max, x_min],
                [y_min, truncated_y_max, truncated_y_max, y_min],
                [z_min, z_min, z_max, z_max],
            ]
        )
    )
    if num_fracs == 1:
        # Check that all nodes of the grid lie in the fracture plane
        sd = mdg.subdomains(dim=2)[0]
        _verify_points_in_fracture(sd.nodes, truncated_fracture)
    else:
        assert len(mdg.subdomains(dim=2)) == 0


class TestDFMMeshGeneration:
    """Test meshing of fracture networks in 3d. No fracture hits the domain boundary.

    TODO: We could possibly delete the remaining tests, EK is not sure.
    """

    def test_one_fracture_intersected_by_two(self):
        """One fracture, intersected by two other (but no point intersections)."""

        f_1 = pp.PlaneFracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.PlaneFracture(
            np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])
        )
        f_3 = pp.PlaneFracture(f_2.pts + np.array([0.5, 0, 0]).reshape((-1, 1)))

        # Add some parameters for grid size
        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2, f_3], domain)

        isect_0_coord = np.array([[0, 0], [0, 0], [-0.7, 0.8]])

        isect_1_coord = np.array([[0.5, 0.5], [0, 0], [-0.7, 0.8]])

        isect_0 = IntersectionInfo(0, 1, isect_0_coord)
        isect_1 = IntersectionInfo(0, 2, isect_1_coord)
        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2, f_3],
            isect_line=[isect_0, isect_1],
            expected_num_1d_grids=2,
        )

    def test_partial_rubics_cube(self):
        """This is a part of a rubics-cube style fracture network."""
        f_0 = pp.PlaneFracture(
            np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0], [0, 0, 1, 1]])
        )
        f_1 = pp.PlaneFracture(
            np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])
        )
        f_2 = pp.PlaneFracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        )
        f_3 = pp.PlaneFracture(
            np.array([[0.5, 1, 1, 0.5], [0.5, 0.5, 1, 1], [0.75, 0.75, 0.75, 0.75]])
        )
        f_4 = pp.PlaneFracture(
            np.array([[0.75, 0.75, 0.75, 0.75], [0.5, 1, 1, 0.5], [0.5, 0.5, 1, 1]])
        )

        # This test does not use the standard domain or mesh size arguments, thus we
        # do meshing by hand.
        bbox = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
        domain = pp.Domain(bbox)

        mesh_args = {
            "mesh_size_frac": 0.4,
            "mesh_size_bound": 1,
            "mesh_size_min": 0.2,
            "return_expected": True,
        }
        network = pp.create_fracture_network([f_0, f_1, f_2, f_3, f_4], domain)
        mdg = network.mesh(mesh_args)

        # Known intersection lines.
        isect_lines = [
            # Fracture 0 and 1
            IntersectionInfo(0, 1, np.array([[0.5, 0.5], [0.5, 0.5], [0, 1]])),
            # Fracture 0 and 2
            IntersectionInfo(0, 2, np.array([[0.5, 0.5], [0, 1], [0.5, 0.5]])),
            # Fracture 1 and 2
            IntersectionInfo(1, 2, np.array([[0, 1], [0.5, 0.5], [0.5, 0.5]])),
            # Fracture 0 and 3
            IntersectionInfo(0, 3, np.array([[0.5, 0.5], [0.5, 1], [0.75, 0.75]])),
            # Fracture 1 and 3
            IntersectionInfo(1, 3, np.array([[0.5, 1], [0.5, 0.5], [0.75, 0.75]])),
            # Fracture 1 and 4
            IntersectionInfo(1, 4, np.array([[0.75, 0.75], [0.5, 0.5], [0.5, 1]])),
            # Fracture 2 and 4
            IntersectionInfo(2, 4, np.array([[0.75, 0.75], [0.5, 1], [0.5, 0.5]])),
            # Fracture 3 and 4
            IntersectionInfo(3, 4, np.array([[0.75, 0.75], [0.5, 1], [0.75, 0.75]])),
        ]

        # Known intersection points.
        isect_pt = [
            np.array([0.5, 0.5, 0.75]).reshape((-1, 1)),
            np.array([0.5, 0.5, 0.5]).reshape((-1, 1)),
            np.array([0.75, 0.5, 0.75]).reshape((-1, 1)),
            np.array([0.75, 0.5, 0.5]).reshape((-1, 1)),
        ]

        check_mdg(
            mdg,
            domain,
            fractures=[f_0, f_1, f_2, f_3, f_4],
            expected_num_1d_grids=15,
            expected_num_0d_grids=4,
            isect_line=isect_lines,
            isect_pt=isect_pt,
        )


class TestDFMPolytopeDomain:
    """Test fracture meshing on polytope (non-box) domains.

    This is a rather minimal test suite. There are surely cases that are not covered
    here, and in all likelihood, adding such tests will uncover bugs and shortcomings in
    the implementation. However, considering the limited use of true polytopal domains,
    the current coverage will have to do for now.

    """

    def domain(self):
        """Set up a polytope domain."""
        west = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        east = np.array([[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
        south_w = np.array([[0, 0.5, 0.5, 0], [0, 0, 0, 0], [0, -0.5, 1, 1]])
        south_e = np.array([[0.5, 1, 1, 0.5], [0, 0, 0, 0], [-0.5, 0, 1, 1]])
        north_w = np.array([[0, 0.5, 0.5, 0], [1, 1, 1, 1], [0, -0.5, 1, 1]])
        north_e = np.array([[0.5, 1, 1, 0.5], [1, 1, 1, 1], [-0.5, 0, 1, 1]])
        bottom_w = np.array([[0, 0.5, 0.5, 0], [0, 0, 1, 1], [0, -0.5, -0.5, 0]])
        bottom_e = np.array([[0.5, 1, 1, 0.5], [0, 0, 1, 1], [-0.5, 0.0, 0, -0.5]])
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

    def _generate_mesh(self, fractures):
        domain = pp.Domain(polytope=self.domain())
        network = pp.create_fracture_network(fractures, domain)
        mesh_args = {"mesh_size_bound": 1, "mesh_size_frac": 1, "mesh_size_min": 0.1}
        mdg = network.mesh(mesh_args)
        return mdg

    def test_fracture_split_by_domain(self):
        """The fracture should be split into subfractures because of the non-convexity
        of the domain.
        """
        f_1 = pp.PlaneFracture(
            np.array([[-1, 2, 2, -1], [0.5, 0.5, 0.5, 0.5], [-1, -1, 0.3, 0.3]])
        )
        mdg = self._generate_mesh([f_1])
        assert len(mdg.subdomains(dim=2)) == 1

    def test_fracture_not_split_by_domain(self):
        f_1 = pp.PlaneFracture(
            np.array([[-1, 2, 2, -1], [0.5, 0.5, 0.5, 0.5], [0, 1, 0.7, 0.7]])
        )
        mdg = self._generate_mesh([f_1])
        assert len(mdg.subdomains(dim=2)) == 1
