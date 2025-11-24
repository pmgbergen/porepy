"""Testing functionality related to FractureNetwork2d."""

from pathlib import Path

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.domains import unit_cube_domain as unit_domain
from porepy.applications.test_utils.arrays import compare_arrays
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.line_fracture import LineFracture
from porepy.fracs.utils import pts_edges_to_linefractures
from porepy.geometry.domain import Domain
from pathlib import Path
import gmsh


def check_mdg_from_polytopal_2d_domain(
    domain: pp.Domain, network: FractureNetwork2d, num_dom: list[int]
) -> None:
    """Helper method to check that meshing a 2d domain with a polytopal (i.e., not
    rectangular) domain results works as expected. Three tests are performed:
    1. That imposing the external boundary does not change the domain
    2. That the generated mesh has the expected number of subdomains
    3. That the generated mesh respects the domain.polytope lines and that
       boundary nodes and faces are correctly tagged.

    """
    # check if the domain in a fracture network is preserved after calling
    # impose_external_boundary
    network_0 = network.copy()
    network.impose_external_boundary(domain)
    # A failure here means that the domain in the network was not properly set.
    assert network_0.domain == network.domain

    # construct the gmsh mesh and check some of its properties
    mdg = pp.create_mdg(
        grid_type="simplex",
        meshing_args={"cell_size": 0.5},
        fracture_network=network,
    )

    # check the number of subdomains
    for dim, n in enumerate(num_dom):
        # A failure here (most likely related to a fracture subdomain) means that a
        # fracture was not properly treated (possibly not split) when the domain
        # boundary was imposed.
        assert n == len(mdg.subdomains(dim=dim))

    # check if boundary faces and nodes respect the domain.polytope lines
    for sd in mdg.subdomains():
        # check if the boundary faces belong to the domain.polytope
        faces = sd.tags["domain_boundary_faces"]
        bf = sd.face_centers[:, faces]

        # check if the boundary nodes belong to the domain.polytope
        nodes = sd.tags["domain_boundary_nodes"]
        bn = sd.nodes[:, nodes]

        face_on_boundary = np.zeros(faces.sum(), dtype=bool)
        node_on_boundary = np.zeros(nodes.sum(), dtype=bool)
        # Loop over all lines in the domain.polytope. Compute the distance from the
        # tagged boundary faces and nodes, mark those that are (almost) zero as
        # being on a boundary.
        for line in domain.polytope:
            dist, _ = pp.geometry.distances.points_segments(
                bf[:2, :], line[:, 0], line[:, 1]
            )
            face_on_boundary[np.isclose(dist, 0).ravel()] = True

            dist, _ = pp.geometry.distances.points_segments(
                bn[:2, :], line[:, 0], line[:, 1]
            )
            node_on_boundary[np.isclose(dist, 0).ravel()] = True

        # All faces and nodes on the boundary should have been found by the above
        # loop.
        assert np.all(face_on_boundary)
        assert np.all(node_on_boundary)


@pytest.fixture
def points() -> np.ndarray:
    return np.array([[0, 2, 1, 1], [0, 0, 0, 1]])


@pytest.fixture
def edges() -> np.ndarray:
    return np.array([[0, 2], [1, 3]])


@pytest.fixture
def fracs(points: np.ndarray, edges: np.ndarray) -> list[pp.LineFracture]:
    return pts_edges_to_linefractures(points, edges)


@pytest.fixture
def domain() -> pp.Domain:
    return pp.Domain({"xmin": 0, "xmax": 5, "ymin": -1, "ymax": 5})


@pytest.fixture
def small_domain() -> pp.Domain:
    return pp.Domain({"xmin": -1, "xmax": 1.5, "ymin": -1, "ymax": 5})


@pytest.fixture(scope="module")
def unit_square() -> pp.Domain:
    return pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})


@pytest.fixture(scope="module")
def mesh_args() -> dict:
    """Create standard mesh arguments for testing purposes."""
    return {"mesh_size_bound": 1.0, "mesh_size_frac": 1.0, "mesh_size_min": 1e-5}


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


def test_snap_fractures():
    p = np.array([[0, 2, 1, 1], [0, 0, 1e-3, 1]])
    e = np.array([[0, 2], [1, 3]])
    fracs = pts_edges_to_linefractures(p, e)
    network = pp.create_fracture_network(fracs)
    snapped = network.snapped_copy(tol=1e-2)

    known_points = np.array([[0, 2, 1, 1], [0, 0, 0, 1]])
    assert compare_arrays(known_points, snapped._pts)

    snapped_2 = network.snapped_copy(tol=1e-4)
    assert compare_arrays(p, snapped_2._pts)


def test_split_intersections(fracs: list[LineFracture], points: np.ndarray):
    network = pp.create_fracture_network(fracs)

    split_network = network.copy_with_split_intersections()
    assert compare_arrays(split_network._pts, points)
    assert split_network._edges.shape[1] == 3


def test_constrain_to_domain(
    fracs: list[LineFracture], domain: Domain, points: np.ndarray, small_domain: Domain
):
    network = pp.create_fracture_network(fracs, domain)
    new_network = network.constrain_to_domain()
    assert compare_arrays(points, new_network._pts)

    small_network = network.constrain_to_domain(small_domain)
    known_points = np.array([[0, 1.5, 1, 1], [0, 0, 0, 1]])
    assert compare_arrays(known_points, small_network._pts)


# Below are tests with polytopal domains


@pytest.mark.parametrize(
    "arg",
    [
        # Test a grid where the outer domain is a triangle, no fractures.
        {
            "lines": [
                [[0, 1], [0, 0]],
                [[1, 0], [0, 1]],
                [[0, 0], [1, 0]],
            ],
            "expected": [0, 0, 1],
            "fracs": [],
        },
        # Test a grid where the outer domain is a triangle, one fracture immersed in the
        # domain.
        {
            "lines": [
                [[0, 1], [0, 0]],
                [[1, 0], [0, 1]],
                [[0, 0], [1, 0]],
            ],
            "expected": [0, 1, 1],
            "fracs": [pp.LineFracture(np.array([[0, 1], [0, 1]]))],
        },
        # Test a grid where the outer domain is a convex pentagon, one fracture is
        # partially immersed which results in two 1d grids.
        {
            "lines": [
                [[0, 1], [0, 0]],
                [[1, 1], [0, 1]],
                [[1, 0.5], [1, 1.5]],
                [[0.5, 0], [1.5, 1]],
                [[0, 0], [1, 0]],
            ],
            "expected": [0, 1, 1],
            "fracs": [pp.LineFracture(np.array([[0.2, 0.8], [1.3, 1.3]]))],
        },
    ],
)
def test_create_fracture_network(arg):
    lines = np.array(arg["lines"])
    expected = arg["expected"]
    fracs = arg["fracs"]

    domain = pp.Domain(polytope=lines)

    network = pp.create_fracture_network(fractures=fracs, domain=domain)
    check_mdg_from_polytopal_2d_domain(domain, network, expected)


# Test of other methods


def test_copy(fracs: list[LineFracture], points: np.ndarray):
    network_1 = pp.create_fracture_network(fracs)

    copy = network_1.copy()
    num_p = points.shape[1]

    network_1._pts = np.random.rand(2, num_p)
    assert np.allclose(copy._pts, points)


@pytest.mark.parametrize(
    "arg",
    [
        # No snapping.
        {
            "points": [[0, 1, 0, 1], [0, 0, 1, 1]],
            "edges": [[0, 2], [1, 3]],
        },
        # Snap to vertex.
        {
            "points": [[0, 1, 0, 1], [0, 0, 1e-4, 1]],
            "edges": [[0, 2], [1, 3]],
            "points_expected": [[0, 1, 0, 1], [0, 0, 0, 1]],
        },
        # No snapping because the snapping tolerance is small.
        {
            "points": [[0, 1, 0, 1], [0, 0, 1e-4, 1]],
            "edges": [[0, 2], [1, 3]],
            "snap_tol": 1e-5,
        },
        # Snapping to segment.
        {
            "points": [[0, 1, 0.5, 1], [0, 0, 1e-4, 1]],
            "edges": [[0, 2], [1, 3]],
            "points_expected": [[0, 1, 0.5, 1], [0, 0, 0, 1]],
        },
    ],
)
def test_snapping(arg):
    points = np.array(arg["points"])
    edges = np.array(arg["edges"])
    if arg.get("points_expected", None):
        points_expected = np.array(arg["points_expected"])
    else:
        points_expected = points.copy()
    snap_tol = arg.get("snap_tol", 1e-3)
    fracs = pts_edges_to_linefractures(points, edges)
    network = pp.create_fracture_network(fracs)
    pn, conv = network._snap_fracture_set(points, snap_tol=snap_tol)
    assert np.allclose(points_expected, pn)
    assert conv


# Test of meshing


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
def test_meshing_two_intersecting_fractures(
    x_coord: float, is_constraint: list[bool], unit_square: pp.Domain, mesh_args: dict
):
    """Test meshing of two intersecting fractures.

    We vary whether each fracture is a constraint or not.

    Parameters:
        x_coord: x-coordinate of the vertical fracture.
        is_constraint: Whether each fracture is a constraint.
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
    mdg = network.mesh(mesh_args, constraints=constraints)

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
        pp.LineFracture(np.array([[0.5, 0.5], [1.0, 0.0]])),
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
