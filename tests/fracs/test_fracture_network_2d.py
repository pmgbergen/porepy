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
        # Test a grid where the outer domain is a concave pentagon, one fracture is
        # partially immersed which results in two 1d grids.
        {
            "lines": [
                [[0, 1], [0, 0]],
                [[1, 1], [0, 1]],
                [[1, 0.5], [1, 0.5]],
                [[0.5, 0], [0.5, 1]],
                [[0, 0], [1, 0]],
            ],
            "expected": [0, 2, 1],
            "fracs": [pp.LineFracture(np.array([[0.2, 0.8], [0.6, 0.6]]))],
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


class Test2dDomain:
    """Tests of meshing 2d fractured domains.

    For each fracture configuration, we verify tha the number of 1d and 0d grids
    are as expected.
    """

    def setUp(self):
        self.domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})
        self.mesh_args = {
            "mesh_size_bound": 1,
            "mesh_size_frac": 1,
            "mesh_size_min": 0.1,
        }

        self.p1 = np.array([[0.2, 0.8], [0.2, 0.8]])
        self.e1 = np.array([[0], [1]])
        # Two intersecting fractures. These extend to the boundary of the domain; hence
        # the domain will be split into smaller units by these fractures.
        self.p2 = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
        self.e2 = np.array([[0, 2], [1, 3]])

    def _verify_num_grids(self, mdg: pp.MixedDimensionalGrid, num_grids: list[int]):
        """Helper method to verify that the number of grids of each dimension is as
        expected.

        Parameters:
            mdg: Mixed-dimensional grid.
            num_grids: List of expected number of grids of each dimension, starting from
                0d grids.

        """
        for dim, n in enumerate(num_grids):
            assert n == len(mdg.subdomains(dim=dim))

    def _generate_mesh(self, pts, edges, constraints=None):
        if pts is None and edges is None:
            line_fractures = None
        else:
            line_fractures = pts_edges_to_linefractures(pts, edges)
        network = pp.create_fracture_network(line_fractures, self.domain)
        mdg = network.mesh(self.mesh_args, constraints=constraints)
        return mdg

    def test_no_fractures(self):
        self.setUp()
        mdg = self._generate_mesh(None, None)
        self._verify_num_grids(mdg, [0, 0, 1])

    def test_one_fracture(self):
        self.setUp()
        mdg = self._generate_mesh(self.p1, self.e1)
        self._verify_num_grids(mdg, [0, 1, 1])

    def test_two_intersecting_fractures(self):
        self.setUp()
        mdg = self._generate_mesh(self.p2, self.e2)
        self._verify_num_grids(mdg, [1, 2, 1])

    def test_one_constraint(self):
        self.setUp()
        mdg = self._generate_mesh(self.p1, self.e1, constraints=np.array([0]))
        self._verify_num_grids(mdg, [0, 0, 1])

    def test_two_constraints(self):
        self.setUp()
        mdg = self._generate_mesh(self.p2, self.e2, constraints=np.arange(2))
        self._verify_num_grids(mdg, [0, 0, 1])

    def test_one_fracture_one_constraint(self):
        self.setUp()
        mdg = self._generate_mesh(self.p2, self.e2, constraints=np.array(1))
        self._verify_num_grids(mdg, [0, 1, 1])
