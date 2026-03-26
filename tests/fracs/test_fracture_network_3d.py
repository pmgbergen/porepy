"""Testing functionality related to FractureNetwork3d."""

from collections import namedtuple

import gmsh  # Needed to finalize gmsh in fixture.
import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.domains import unit_cube_domain as unit_domain
from porepy.applications.test_utils.arrays import compare_arrays
from porepy.fracs.plane_fracture import PlaneFracture


@pytest.fixture(scope="module")
def unit_box() -> pp.Domain:
    """Create a unit box domain for testing purposes."""
    bbox = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
    domain = pp.Domain(bbox)
    return domain


@pytest.fixture(scope="module")
def mesh_args() -> dict:
    """Create standard mesh arguments for testing purposes."""
    return {
        "mesh_size_boundary": 1,
        "mesh_size_fracture": 1,
        "mesh_size_min": 0.1,
        # Set a very low refinement proximity multiplier to avoid adaptive refinement.
        "refinement_proximity_multiplier": 1e-6,
        # Set values for mesh coarsening that will have minimal impact on the mesh.
        "refinement_size_multiplier": 1.0,
        "background_transition_multiplier": 1.01,
    }


@pytest.fixture(autouse=True)
def finalize_gmsh():
    """Fixture to ensure gmsh is finalized after each test.

    This is to avoid tests failing because gmsh was not cleared after a previously
    breaking test.
    """
    yield  # This is where the test runs
    try:
        # Try to clear and finalize gmsh after each test. This will raise an error
        # if gmsh was not initialized in the test, but we can ignore that.
        gmsh.clear()
        gmsh.finalize()
    except Exception:
        pass


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


def _find_intersection_line(mdg, frac_num_0, frac_num_1):
    """Find the set of 1d subdomain corresponding to the intersection of two fractures.

    Parameters:
        mdg: Mixed-dimensional grid. frac_num_0: Fracture number of the first fracture.
        frac_num_1: Fracture number of the second fracture.

    Returns:
        A list of 1d subdomains corresponding to the intersection line. The list may
        contain multiple subdomains if the intersection line is split by other
        fractures, or by mesh size control points.

    """
    subdomains = []
    for sd in mdg.subdomains(dim=1):
        neighs = mdg.neighboring_subdomains(sd, only_higher=True)
        if (neighs[0].frac_num == frac_num_0 and neighs[1].frac_num == frac_num_1) or (
            neighs[0].frac_num == frac_num_1 and neighs[1].frac_num == frac_num_0
        ):
            subdomains.append(sd)
    if subdomains:
        return subdomains
    assert False, "Intersection line not found."


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
    x_coord: list[float],
    is_constraint: list[bool],
    unit_box: pp.Domain,
    mesh_args: dict,
):
    """Test meshing of a single fracture without intersections.

    Parameters:
        x_coord: x-coordinate of the vertical fracture.
        is_constraint: Whether the fracture is a constraint.
        unit_box: Unit box domain.
        mesh_args: Mesh arguments for meshing.

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
    mdg = network.mesh(mesh_args=mesh_args, constraints=constraints)

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
    mesh_args: dict,
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
    mdg = network.mesh(mesh_args=mesh_args, constraints=constraints)
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
        for sd in _find_intersection_line(mdg, frac_num_0, frac_num_1):
            _verify_points_in_line(sd.nodes, start, end)

    if is_fracture[0] and is_fracture[1]:
        _check_line_intersection(0, 1)
    if is_fracture[0] and is_fracture[2]:
        _check_line_intersection(0, 2)
    if is_fracture[1] and is_fracture[2]:
        _check_line_intersection(1, 2)

    if num_fractures == 3:
        # Verify that there is exactly one 0d grid located at the intersection point of
        # the three fractures. There may be other points related to mesh size control,
        # but these we ignore.
        assert (
            np.sum(
                [
                    np.allclose(sd.cell_centers, np.array([[0.5], [0.5], [0.5]]))
                    for sd in mdg.subdomains(dim=0)
                ]
            )
            == 1
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
    z_coord: list[float],
    is_t: bool,
    is_constraint: list[bool],
    unit_box: pp.Domain,
    mesh_args: dict,
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
    mdg = network.mesh(mesh_args=mesh_args, constraints=constraints)
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
    is_constraint: list[bool], dfn: bool, unit_box: pp.Domain, mesh_args: dict
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
        mesh_args=mesh_args,
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


def test_meshing_with_mesh_size_control_points(unit_box: pp.Domain, mesh_args: dict):
    """Use mesh size parameters that will trigger insertion of mesh size control points.

    Parameters:
        unit_box: Unit box domain.

    """
    # Set the mesh size parameters to values that will trigger insertion of mesh size
    # control points.
    mesh_args["refinement_threshold"] = 0.1
    fracture_0 = pp.PlaneFracture(
        np.array([[0.4, 0.91, 0.91, 0.4], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])
    )
    fracture_1 = pp.PlaneFracture(
        np.array([[0.3, 0.8, 0.8, 0.3], [0.3, 0.3, 0.7, 0.7], [0.5, 0.5, 0.5, 0.5]])
    )
    fracture_2 = pp.PlaneFracture(
        np.array([[0.5, 0.5, 0.5, 0.5], [0.3, 0.3, 0.7, 0.7], [0.2, 0.8, 0.8, 0.2]])
    )

    fractures = [fracture_0, fracture_1, fracture_2]
    network = pp.create_fracture_network(fractures, unit_box)
    mdg = network.mesh(mesh_args=mesh_args)
    assert len(mdg.subdomains(dim=2)) == 3
    assert len(mdg.subdomains(dim=1)) == 6
    assert len(mdg.subdomains(dim=0)) == 1


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
def test_fracture_hits_boundary(
    x_min, x_max, is_constraint, unit_box: pp.Domain, mesh_args: dict
):
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
    mdg = network.mesh(mesh_args=mesh_args, constraints=constraints)
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
    extend_beyond: bool, is_constraint: bool, unit_box: pp.Domain, mesh_args: dict
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
    mdg = network.mesh(mesh_args=mesh_args, constraints=constraints)
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


@pytest.mark.parametrize("num_fracs", [1, 2, 3])
def test_domain_split_by_fractures(
    num_fracs: int, unit_box: pp.Domain, mesh_args: dict
):
    """Test meshing when fractures split the domain into multiple subdomains.

    This is known to be a weak point in the meshing algorithm, since Gmsh has a tendency
    to treat the domain as multiple subdomains, generating edge cases that must be
    handled in a robust implementation.

    Parameters:
        num_fracs: Number of fractures to include in the network. unit_square: Unit
        square domain fixture. mesh_args: Meshing arguments.

    """
    fractures = [
        pp.PlaneFracture(
            np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]])
        ),
        pp.PlaneFracture(
            np.array([[0.0, 1.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0]])
        ),
        pp.PlaneFracture(
            np.array([[0.5, 0.5, 0.5, 0.5], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        ),
    ][:num_fracs]

    network = pp.create_fracture_network(fractures, unit_box)
    # Generate a mixed-dimensional grid with a grid as coarse as possible.
    mdg = network.mesh(mesh_args)

    # There should still be a single 3d grid as far as PorePy is concerned.
    assert len(mdg.subdomains(dim=3)) == 1
    # There should be num_fracs 1d grids.
    assert len(mdg.subdomains(dim=2)) == num_fracs
    # There should be 0, 1, or 3 1d grids depending on num_fracs.
    expected_1d_grids = 0
    if num_fracs == 2:
        expected_1d_grids = 1
    elif num_fracs == 3:
        expected_1d_grids = 6
    assert len(mdg.subdomains(dim=1)) == expected_1d_grids

    # There should be a single 0d grid if there are three fractures.
    num_0d_grids = 1 if num_fracs == 3 else 0
    assert len(mdg.subdomains(dim=0)) == num_0d_grids


@pytest.mark.parametrize("num_constraints", [0, 1, 2])
def test_fractures_intersect_at_boundary(
    num_constraints: int, unit_box: pp.Domain, mesh_args: dict
):
    """Test meshing when fractures intersect at the domain boundary.

    Parameters:
        num_constraints: Number of fractures to include as constraints.
        unit_square: Unit square domain fixture.
        mesh_args: Meshing arguments.

    """
    fractures = [
        pp.PlaneFracture(
            np.array([[0.2, 0.8, 0.8, 0.2], [0.0, 0.0, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5]])
        ),
        pp.PlaneFracture(
            np.array([[0.2, 0.5, 0.5, 0.2], [0.2, 0.0, 0.0, 0.2], [0.2, 0.2, 0.8, 0.8]])
        ),
        pp.PlaneFracture(
            np.array([[0.8, 0.5, 0.5, 0.8], [0.2, 0.0, 0.0, 0.2], [0.2, 0.2, 0.8, 0.8]])
        ),
    ]

    network = pp.create_fracture_network(fractures, unit_box)
    constraints = np.arange(num_constraints)
    # Generate a mixed-dimensional grid with a grid as coarse as possible.
    mdg = network.mesh(mesh_args, constraints=constraints)
    # TODO: The number of intersection lines and points are critical here, but we cannot
    # fix it until we have cleaned up the meshing code so that mesh control points do
    # not introduce 1d and 0d grids.
    num_fracs = 3 - num_constraints
    assert len(mdg.subdomains(dim=2)) == num_fracs
    if num_fracs == 3:
        # There should be three intersection lines and one intersection point.
        assert len(mdg.subdomains(dim=1)) == 2
        assert len(mdg.subdomains(dim=0)) == 0
    else:
        # There should be a single intersection line.
        assert len(mdg.subdomains(dim=1)) == 0
        assert len(mdg.subdomains(dim=0)) == 0


class TestDFMMeshGeneration:
    """Legacy tests for meshing. These cover aspects not covered by the more
    parametrized tests above, and are therefore kept for completeness.
    """

    def test_one_fracture_intersected_by_two(
        self, unit_box: pp.Domain, mesh_args: dict
    ):
        """One fracture, intersected by two other (but no point intersections)."""

        f_0 = pp.PlaneFracture(
            np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])
        )
        f_1 = pp.PlaneFracture(
            np.array([[0.2, 0.2, 0.2, 0.2], [0, 1, 1, 0], [0, 0, 0.8, 0.8]])
        )
        f_2 = pp.PlaneFracture(f_1.pts + np.array([0.5, 0, 0]).reshape((-1, 1)))
        fractures = [f_0, f_1, f_2]
        network = pp.create_fracture_network(fractures, unit_box)
        mdg = network.mesh(mesh_args=mesh_args)

        for sd in mdg.subdomains(dim=2):
            _verify_points_in_fracture(sd.nodes, fractures[sd.frac_num])

        for sd in _find_intersection_line(mdg, 0, 1):
            _verify_points_in_line(
                sd.nodes, np.array([0.2, 0.5, 0]), np.array([0.2, 0.5, 0.8])
            )
        for sd in _find_intersection_line(mdg, 0, 2):
            _verify_points_in_line(
                sd.nodes, np.array([0.7, 0.5, 0]), np.array([0.7, 0.5, 0.8])
            )

    def test_partial_rubics_cube(self, unit_box: pp.Domain, mesh_args: dict):
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
        fractures = [f_0, f_1, f_2, f_3, f_4]
        network = pp.create_fracture_network(fractures, unit_box)
        mdg = network.mesh(mesh_args=mesh_args)

        for sd in mdg.subdomains(dim=2):
            _verify_points_in_fracture(sd.nodes, fractures[sd.frac_num])

        for sd in _find_intersection_line(mdg, 0, 1):
            _verify_points_in_line(
                sd.nodes, np.array([0.5, 0.5, 0]), np.array([0.5, 0.5, 1])
            )
        for sd in _find_intersection_line(mdg, 0, 2):
            _verify_points_in_line(
                sd.nodes, np.array([0.5, 0, 0.5]), np.array([0.5, 1, 0.5])
            )
        for sd in _find_intersection_line(mdg, 1, 2):
            _verify_points_in_line(
                sd.nodes, np.array([0, 0.5, 0.5]), np.array([1, 0.5, 0.5])
            )
        for sd in _find_intersection_line(mdg, 0, 3):
            _verify_points_in_line(
                sd.nodes, np.array([0.5, 0.5, 0.75]), np.array([0.5, 1, 0.75])
            )
        for sd in _find_intersection_line(mdg, 1, 3):
            _verify_points_in_line(
                sd.nodes, np.array([0.5, 0.5, 0.75]), np.array([1, 0.5, 0.75])
            )
        for sd in _find_intersection_line(mdg, 1, 4):
            _verify_points_in_line(
                sd.nodes, np.array([0.75, 0.5, 0.5]), np.array([0.75, 0.5, 1])
            )
        for sd in _find_intersection_line(mdg, 2, 4):
            _verify_points_in_line(
                sd.nodes, np.array([0.75, 0.5, 0.5]), np.array([0.75, 1, 0.5])
            )
        for sd in _find_intersection_line(mdg, 3, 4):
            _verify_points_in_line(
                sd.nodes, np.array([0.75, 1, 0.75]), np.array([0.75, 0.5, 0.75])
            )

        # Known intersection points.
        isect_pt = [
            np.array([0.5, 0.5, 0.75]).reshape((-1, 1)),
            np.array([0.5, 0.5, 0.5]).reshape((-1, 1)),
            np.array([0.75, 0.5, 0.75]).reshape((-1, 1)),
            np.array([0.75, 0.5, 0.5]).reshape((-1, 1)),
        ]

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

    def _generate_mesh(self, fractures, mesh_args: dict):
        domain = pp.Domain(polytope=self.domain())
        network = pp.create_fracture_network(fractures, domain)
        mdg = network.mesh(mesh_args=mesh_args)
        return mdg

    def test_fracture_split_by_domain(self, mesh_args: dict):
        """The fracture should be split into subfractures because of the non-convexity
        of the domain.
        """
        f_1 = pp.PlaneFracture(
            np.array([[-1, 2, 2, -1], [0.5, 0.5, 0.5, 0.5], [-1, -1, 0.3, 0.3]])
        )
        mdg = self._generate_mesh([f_1], mesh_args)
        assert len(mdg.subdomains(dim=2)) == 1

    def test_fracture_not_split_by_domain(self, mesh_args: dict):
        f_1 = pp.PlaneFracture(
            np.array([[-1, 2, 2, -1], [0.5, 0.5, 0.5, 0.5], [0, 1, 0.7, 0.7]])
        )
        mdg = self._generate_mesh([f_1], mesh_args)
        assert len(mdg.subdomains(dim=2)) == 1
