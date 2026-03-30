"""Testing functionality related to FractureNetwork2d."""

from pathlib import Path

import gmsh
import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.domains import unit_cube_domain as unit_domain
from porepy.applications.test_utils.arrays import compare_arrays
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.line_fracture import LineFracture
from porepy.fracs.utils import pts_edges_to_linefractures
from porepy.geometry.domain import Domain


@pytest.fixture(scope="module")
def unit_square() -> pp.Domain:
    return pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})


@pytest.fixture(scope="module")
def mesh_args() -> dict:
    """Create standard mesh arguments for testing purposes."""
    return {
        "mesh_size_boundary": 1.0,
        "mesh_size_fracture": 1.0,
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


def _verify_1d_grid_geometry(sd: pp.Grid, frac: pp.LineFracture) -> None:
    """Helper method to verify that a 1d grid corresponds to a given fracture.

    We check that the grid nodes lie on the fracture line segment (the distance
    is zero) and that those fracture points that are tagged as boundary or tip nodes
    correspond to the fracture endpoints.

    Parameters:
        sd: 1d grid.
        frac: Line fracture.

    """
    # Check that all nodes are on the fracture line.
    dist, _ = pp.geometry.distances.points_segments(
        sd.nodes[:2],
        frac.pts[:, 0].reshape((-1, 1)),
        frac.pts[:, 1].reshape((-1, 1)),
    )
    assert np.allclose(dist, 0)


@pytest.mark.parametrize(
    "x_coord, is_constraint",
    [
        # No fractures.
        ([], None),
        # Fracture fully inside the domain, not a constraint.
        ([0.2], [False]),
        # Fracture fully inside the domain, is a constraint.
        ([0.2], [True]),
        # Two fracutres fully inside the domain.
        ([0.2, 0.5], [False, False]),
        # Fracture outside the domain, not a constraint.
        ([-0.5], [False]),
        # Fracture outside the domain, is a constraint.
        ([-0.5], [True]),
        # Fracture on the domain boundary.
        ([0.0], [False]),
        # Constraint on the domain boundary.
        ([0.0], [True]),
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
    is_constraint: list[bool] | None,
    unit_square: pp.Domain,
    mesh_args: dict,
):
    """Test meshing of a single fracture without intersections.

    We vary the x-coordinate of the fracture and whether it is constrained or not.

    Parameters:
        x_coord: x-coordinate of the vertical fracture.
        is_constraint: Whether the fracture is a constraint.
        unit_square: Unit square domain.
        mesh_args: Meshing arguments.

    """
    if is_constraint is None:
        is_constraint = len(x_coord) * [False]

    is_fracture = len(x_coord) * [True]

    fractures = []

    for i, x in enumerate(x_coord):
        frac = pp.LineFracture(np.array([[x, x], [0.2, 0.8]]))
        fractures.append(frac)
        if is_constraint[i] or x >= 1.0 or x <= 0.0:
            is_fracture[i] = False

    network = pp.create_fracture_network(fractures, unit_square)
    constraints = np.where(is_constraint)[0]
    # Generate a mixed-dimensional grid with a grid as coarse as possible.
    mdg = network.mesh(mesh_args, constraints=constraints)

    assert len(mdg.subdomains(dim=1)) == sum(is_fracture)
    assert len(mdg.subdomains(dim=0)) == 0
    sd_1d = mdg.subdomains(dim=1)
    counter = 0
    for frac in fractures:
        if is_fracture[counter]:
            _verify_1d_grid_geometry(sd_1d[counter], frac)
            counter += 1


@pytest.mark.parametrize(
    "x_coord",
    [
        0.2,  # Will give an X-type intersection
        0.5,  # Will give a T-type intersection
    ],
)
@pytest.mark.parametrize(
    "is_constraint", [[False, False], [True, False], [False, True], [True, True]]
)
@pytest.mark.parametrize("dfn", [False, True])
def test_meshing_two_intersecting_fractures(
    x_coord: float,
    is_constraint: list[bool],
    dfn: bool,
    unit_square: pp.Domain,
    mesh_args: dict,
):
    """Test meshing of two intersecting fractures.

    We vary whether each fracture is a constraint or not.

    Parameters:
        x_coord: x-coordinate of the vertical fracture.
        is_constraint: Whether each fracture is a constraint.
        dfn: Whether to use DFN meshing or not.
        unit_square: Unit square domain.
        mesh_args: Meshing arguments.

    """
    fractures = [
        pp.LineFracture(np.array([[x_coord, 0.8], [0.5, 0.5]])),
        pp.LineFracture(np.array([[0.5, 0.5], [0.2, 0.8]])),
    ]

    network = pp.create_fracture_network(fractures, unit_square)
    constraints = np.where(is_constraint)[0]
    # Generate a mixed-dimensional grid with a grid as coarse as possible.
    mdg = network.mesh(mesh_args, constraints=constraints, dfn=dfn)

    assert len(mdg.subdomains(dim=1)) == 2 - sum(is_constraint)
    num_0d_grids = 0 if any(is_constraint) else 1
    assert len(mdg.subdomains(dim=0)) == num_0d_grids

    counter = 0
    sd_1d = mdg.subdomains(dim=1)
    for i, frac in enumerate(fractures):
        if not is_constraint[i]:
            _verify_1d_grid_geometry(sd_1d[counter], frac)
            counter += 1

    if num_0d_grids == 1:
        sd_0d = mdg.subdomains(dim=0)[0]
        intersection_point = np.array([[0.5], [0.5], [0.0]])
        assert np.allclose(sd_0d.cell_centers, intersection_point)


@pytest.mark.parametrize(
    "x_coord",
    [
        -0.5,  # Both endpoints outside the domain
        0.5,  # One endpoint inside the domain
    ],
)
@pytest.mark.parametrize("is_constraint", [False, True])
def test_meshing_fracture_crosses_boundary(
    x_coord: float, is_constraint: bool, unit_square: pp.Domain, mesh_args: dict
):
    """Test meshing of a fracture crossing the domain boundary.

    We vary whether the fracture is a constraint or not.

    Parameters:
        x_coord: x-coordinate of the left endpoint of the fracture. The right
            endpoint is at (1.5, 0.5), hence on the right of the domain.
        is_constraint: Whether the fracture is a constraint.
        unit_square: Unit square domain fixture.
        mesh_args: Meshing arguments.

    """
    fracture = pp.LineFracture(np.array([[x_coord, 1.5], [0.5, 0.5]]))

    network = pp.create_fracture_network([fracture], unit_square)
    constraints = np.array([0]) if is_constraint else np.array([])
    # Generate a mixed-dimensional grid with a grid as coarse as possible.
    mdg = network.mesh(mesh_args, constraints=constraints)
    left_lim = max(0.0, x_coord)

    # The constrained fracture will be shorter than the original one.
    constrained_fracture = pp.LineFracture(np.array([[left_lim, 1.0], [0.5, 0.5]]))

    if is_constraint:
        assert len(mdg.subdomains(dim=1)) == 0
    else:
        assert len(mdg.subdomains(dim=1)) == 1
        sd_1d = mdg.subdomains(dim=1)[0]
        _verify_1d_grid_geometry(sd_1d, constrained_fracture)


@pytest.mark.parametrize("num_fracs", [1, 2])
def test_domain_split_by_fractures(
    num_fracs: int, unit_square: pp.Domain, mesh_args: dict
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
        pp.LineFracture(np.array([[0.5, 0.5], [0.0, 1.0]])),
        pp.LineFracture(np.array([[0.0, 1.0], [0.5, 0.5]])),
    ][:num_fracs]

    network = pp.create_fracture_network(fractures, unit_square)
    # Generate a mixed-dimensional grid with a grid as coarse as possible.
    mdg = network.mesh(mesh_args)

    # There should still be a single 2d grid as far as PorePy is concerned.
    assert len(mdg.subdomains(dim=2)) == 1
    # There should be num_fracs 1d grids.
    assert len(mdg.subdomains(dim=1)) == num_fracs
    # There should be a single 0d grid if there are two fractures.
    num_0d_grids = 1 if num_fracs == 2 else 0
    assert len(mdg.subdomains(dim=0)) == num_0d_grids


@pytest.mark.parametrize("num_constraints", [0, 1, 2])
def test_fractures_intersect_at_boundary(
    num_constraints: int, unit_square: pp.Domain, mesh_args: dict
):
    """Test meshing when fractures intersect at the domain boundary.

    This is known to be a weak point in the meshing algorithm, since Gmsh has a tendency
    to treat the domain as multiple subdomains, generating edge cases that must be
    handled in a robust implementation.

    Parameters:
        unit_square: Unit square domain fixture.
        mesh_args: Meshing arguments.

    """
    fractures = [
        pp.LineFracture(np.array([[0.2, 0.5], [0.2, 0.0]])),
        pp.LineFracture(np.array([[0.8, 0.5], [0.2, 0.0]])),
    ]
    constraints = np.arange(num_constraints)

    network = pp.create_fracture_network(fractures, unit_square)
    # Generate a mixed-dimensional grid with a grid as coarse as possible.
    mdg = network.mesh(mesh_args, constraints=constraints)

    # There should still be a single 2d grid as far as PorePy is concerned.
    assert len(mdg.subdomains(dim=2)) == 1
    # There should be two 1d grids.
    assert len(mdg.subdomains(dim=1)) == 2 - num_constraints
    # There should be a single 0d grid.
    assert len(mdg.subdomains(dim=0)) == 0


def test_meshing_with_mesh_size_control_points(unit_square: pp.Domain, mesh_args: dict):
    """Test meshing of a fracture network with mesh size control points.

    We test that the mesh generation process is successful when the insertion of mesh
    size control points is triggered.

    Parameters:
        unit_square: Unit square domain fixture.
        mesh_args: Meshing arguments.

    """
    fractures = [
        pp.LineFracture(np.array([[0.2, 0.8], [0.5, 0.5]])),
        pp.LineFracture(np.array([[0.5, 0.5], [0.52, 0.6]])),
        pp.LineFracture(np.array([[0.3, 0.3], [0.8, 0.2]])),
    ]

    network = pp.create_fracture_network(fractures, unit_square)
    # Increase the refinement threshold to trigger the insertion of mesh size control
    # points.
    mesh_args["refinement_threshold"] = 0.1

    # Generate a mixed-dimensional grid with a grid as coarse as possible.
    mdg = network.mesh(mesh_args)

    # There should be two 1d grids.
    assert len(mdg.subdomains(dim=1)) == 3
    # There should be a single 0d grid.
    assert len(mdg.subdomains(dim=0)) == 1
