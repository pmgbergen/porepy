"""Testing functionality related to FractureNetwork2d."""
import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.arrays import compare_arrays
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.line_fracture import LineFracture
from porepy.fracs.utils import pts_edges_to_linefractures
from porepy.geometry.domain import Domain
from porepy.applications.md_grids.domains import unit_cube_domain as unit_domain


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


def test_get_points(points: np.ndarray, edges: np.ndarray, fracs: list[LineFracture]):
    network = pp.create_fracture_network(fracs)

    start = network.start_points()
    assert compare_arrays(start, points[:, edges[0]])
    end = network.end_points()
    assert compare_arrays(end, points[:, edges[1]])

    # Then index based
    start = network.start_points(frac_index=0)
    assert compare_arrays(start, points[:, 0].reshape((-1, 1)))
    end = network.end_points(frac_index=0)
    assert compare_arrays(end, points[:, 1].reshape((-1, 1)))


def test_length(fracs: list[LineFracture]):
    network = pp.create_fracture_network(fracs)
    length = network.length()
    known_length = np.array([2, 1])
    assert np.allclose(length, known_length)


def test_angle_0_pi():
    num_frac = 10
    start = np.zeros((2, num_frac))

    end = 2 * np.random.rand(2, num_frac) - 1
    p = np.hstack((start, end))
    e = np.vstack((np.arange(num_frac), num_frac + np.arange(num_frac)))
    fracs = pts_edges_to_linefractures(p, e)
    network = pp.create_fracture_network(fracs)
    angle = network.orientation()

    assert np.all(angle >= 0)
    assert np.all(angle < np.pi)


def test_angle(fracs: list[LineFracture]):
    network = pp.create_fracture_network(fracs)
    angle = network.orientation()
    known_orientation = np.array([0, np.pi / 2])
    assert np.logical_or(
        angle == known_orientation, angle - np.pi == known_orientation
    ).all()


def test_add_networks_no_domain(fracs: list[LineFracture], points: np.ndarray):
    network_1 = pp.create_fracture_network(fracs)
    p2 = np.array([[0, 2, 1, 1], [0, 0, 0, 1]]) + 2
    e2 = np.array([[0, 2], [1, 3]])
    fracs2 = pts_edges_to_linefractures(p2, e2)
    network_2 = pp.create_fracture_network(fracs2)

    together = network_1.add_network(network_2)

    p_known = np.hstack((points, p2))
    assert compare_arrays(p_known, together._pts)
    # The known edges has 2 rows, thus by testing for equality, we implicitly
    # verify there are no tags in the joint network
    e_known = np.array([[0, 2, 4, 6], [1, 3, 5, 7]])
    assert compare_arrays(together._edges, e_known)


def test_add_networks_domains(
    fracs: list[LineFracture], domain: Domain, small_domain: Domain
):
    network_1 = pp.create_fracture_network(fracs, domain)
    p2 = np.array([[0, 2, 1, 1], [0, 0, 0, 1]]) + 2
    e2 = np.array([[0, 2], [1, 3]])  #
    fracs2 = pts_edges_to_linefractures(p2, e2)
    # A network with no domain
    network_2 = pp.create_fracture_network(fracs2)

    # Add first to second, check domain
    together = network_1.add_network(network_2)
    assert domain.bounding_box == together.domain.bounding_box

    # Add second to first, check domain
    together = network_2.add_network(network_1)
    assert domain.bounding_box == together.domain.bounding_box

    # Assign domain, then add
    network_2.domain = domain
    together = network_1.add_network(network_2)
    assert domain.bounding_box == together.domain.bounding_box

    # Assign different domain, check that the sum has a combination of the
    # domain dicts
    network_2.domain = small_domain
    together = network_1.add_network(network_2)
    combined_domain = pp.Domain({"xmin": -1, "xmax": 5.0, "ymin": -1, "ymax": 5})
    assert combined_domain.bounding_box == together.domain.bounding_box


def test_add_networks_preserve_tags(fracs: list[LineFracture]):
    network_1 = pp.create_fracture_network(fracs)
    p2 = np.array([[0, 2, 1, 1], [0, 0, 0, 1]]) + 2
    # Network 2 has tags
    tag2 = 1
    e2 = np.array([[0, 2], [1, 3], [1, 1]])
    fracs2 = pts_edges_to_linefractures(p2, e2)
    network_2 = pp.create_fracture_network(fracs2)

    # Add networks, check tags
    together = network_1.add_network(network_2)
    assert np.all(together._edges[2, 2:4] == tag2)

    # Add networks reverse order
    together = network_2.add_network(network_1)
    assert np.all(together._edges[2, :2] == tag2)

    # Assign tags to network1
    tag1 = 2
    # Had to change this to ensure that edges still has ``dtype np.int8``
    network_1._edges = np.vstack((network_1._edges, tag1 * np.ones(2, dtype=np.int8)))
    together = network_1.add_network(network_2)
    known_tags = np.array([tag1, tag1, tag2, tag2])
    assert np.all(together._edges[2] == known_tags)


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
        self.p2 = np.array([[0.2, 0.8, 0.2, 0.8], [0.2, 0.8, 0.8, 0.2]])
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


class TestGmshTags:
    """Test that the tags of the faces of the grids returned from gmsh are correct."""

    def check_boundary_faces(self, g: pp.Grid, offset: int):
        """Helper method to check that the boundary faces are correctly tagged.

        The domain is assumed to be the unit square, and the ordering of the parts of
        the boundary (east, south, etc.) is hardcoded in the FractureNetwork2d class. We
        therefore check that faces tagged as parts of the boundary have the expected
        coordinates.

        NOTE: The domain boundary tags are of the form
        'domain_boundary_line_{offset}_faces', where offset reflects the number of
        fractures and constraints (auxiliary lines) in the network (again, this is just
        how the FractureNetwork2d class is implemented).
        """
        tag = np.where(g.tags[f"domain_boundary_line_{offset}_faces"])[0]
        assert np.allclose(g.face_centers[1, tag], 0)

        tag = np.where(g.tags[f"domain_boundary_line_{offset + 1}_faces"])[0]
        assert np.allclose(g.face_centers[0, tag], 1)

        tag = np.where(g.tags[f"domain_boundary_line_{offset + 2}_faces"])[0]
        assert np.allclose(g.face_centers[1, tag], 1)

        tag = np.where(g.tags[f"domain_boundary_line_{offset + 3}_faces"])[0]
        assert np.allclose(g.face_centers[0, tag], 0)

    def check_auxiliary_fracture_faces(
        self,
        g: pp.Grid,
        key: str,
        start: list[np.ndarray],
        end: list[np.ndarray],
        offset: int = 0,
    ):
        """Helper method to check that the faces of auxiliary (constraint) and fracture
        lines are correctly tagged.
        """
        num_lines = len(start)
        for i in range(num_lines):
            # In cases where both fractures and constraints are present, we need to
            # offset the tag index (since we cannot check fractures and constraints
            # at the same time, this was not implemented).
            tag = g.tags[f"{key}_{i + offset}_faces"]
            dist, _ = pp.distances.points_segments(g.face_centers[:2], start[i], end[i])

            assert np.allclose(dist[tag, 0], 0)
            assert np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))

    def test_face_tags_from_gmsh_before_grid_split(self):
        """Check that the faces of the grids returned from gmsh has correct tags
        before the grid is split. The domain is the unit square with two fractures
        that intersect.
        """
        p = np.array([[0, 1, 0.5, 0.5], [0.5, 0.5, 0, 1]])
        e = np.array([[0, 2], [1, 3]])
        fractures = pts_edges_to_linefractures(p, e)
        network = pp.create_fracture_network(fractures, unit_domain(2))
        mesh_args = {
            "mesh_size_frac": 0.1,
            "mesh_size_bound": 0.1,
        }

        file_name = "mesh_simplex.msh"
        gmsh_data = network.prepare_for_gmsh(mesh_args, None, True, None, False)

        # Consider the dimension of the problem
        ndim = 2
        gmsh_writer = pp.fracs.gmsh_interface.GmshWriter(gmsh_data)
        gmsh_writer.generate(file_name, ndim)

        grid_list = pp.fracs.simplex.triangle_grid_from_gmsh(
            file_name,
        )
        pp.meshing._tag_faces(grid_list, False)
        for grid_of_dim in grid_list:
            for g in grid_of_dim:
                g.compute_geometry()
                if g.dim == 2:
                    f_cc0 = g.face_centers[:, g.tags["fracture_0_faces"]]
                    f_cc1 = g.face_centers[:, g.tags["fracture_1_faces"]]

                    # Test position of the fracture faces. Fracture 1 is aligned with
                    # the y-axis, fracture 2 is aligned with the x-axis
                    assert np.allclose(f_cc0[1], 0.5)
                    assert np.allclose(f_cc1[0], 0.5)

                if g.dim > 0:
                    db_cc = g.face_centers[:, g.tags["domain_boundary_faces"]]
                    # Test that domain boundary faces are on the boundary
                    assert np.all(
                        (np.abs(db_cc[0]) < 1e-10)
                        | (np.abs(db_cc[1]) < 1e-10)
                        | (np.abs(db_cc[0] - 1) < 1e-10)
                        | (np.abs(db_cc[1] - 1) < 1e-10)
                    )
                # No tip faces in this test case
                assert np.all(g.tags["tip_faces"] == 0)

    def test_boundary(self):
        """No fractures, test that the boundary faces are correct."""
        network = pp.create_fracture_network(None, unit_domain(2))
        mesh_args = {"mesh_size_frac": 1}
        mdg = network.mesh(mesh_args)
        g = mdg.subdomains(dim=2)[0]
        self.check_boundary_faces(g, 0)

    @pytest.mark.parametrize(
        "mesh_args", [{"mesh_size_frac": 1}, {"mesh_size_frac": 0.33}]
    )
    def test_single_constraint(self, mesh_args):
        """A single auxiliary line (constraints)"""
        subdomain_start = np.array([[0.25], [0.5]])
        subdomain_end = np.array([[0.5], [0.5]])
        p = np.hstack((subdomain_start, subdomain_end))
        e = np.array([[0], [1]])
        fractures = pts_edges_to_linefractures(p, e)
        network = pp.create_fracture_network(fractures, unit_domain(2))

        mesh_args = {"mesh_size_frac": 1}
        mdg = network.mesh(mesh_args, constraints=np.array([0]))
        g = mdg.subdomains(dim=2)[0]

        self.check_boundary_faces(g, 1)
        self.check_auxiliary_fracture_faces(
            g, "auxiliary_line", [subdomain_start], [subdomain_end]
        )

    @pytest.mark.parametrize(
        "mesh_args", [{"mesh_size_frac": 1}, {"mesh_size_frac": 0.25}]
    )
    def test_two_constraints(self, mesh_args):
        constraint_start_0 = np.array([[0.0], [0.5]])
        constraint_end_0 = np.array([[0.75], [0.5]])
        constraint_start_1 = np.array([[0.5], [0.25]])
        constraint_end_1 = np.array([[0.5], [0.75]])
        p = np.hstack(
            (constraint_start_0, constraint_end_0, constraint_start_1, constraint_end_1)
        )
        e = np.array([[0, 2], [1, 3]])
        fractures = pts_edges_to_linefractures(p, e)
        network = pp.create_fracture_network(fractures, unit_domain(2))

        mdg = network.mesh(mesh_args, constraints=np.arange(2))
        g = mdg.subdomains(dim=2)[0]

        self.check_boundary_faces(g, 2)

        self.check_auxiliary_fracture_faces(
            g,
            "auxiliary_line",
            [constraint_start_0, constraint_start_1],
            [constraint_end_0, constraint_end_1],
        )

    @pytest.mark.parametrize("cell_size", [1.0, 0.25])
    def test_single_fracture(self, cell_size):
        frac_start = np.array([[0.25], [0.5]])
        frac_end = np.array([[0.75], [0.5]])
        mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "simplex",
            meshing_args={"cell_size": cell_size},
            fracture_indices=[1],
            fracture_endpoints=[np.array([0.25, 0.75])],
        )
        g = mdg.subdomains(dim=2)[0]

        self.check_boundary_faces(g, 1)
        self.check_auxiliary_fracture_faces(g, "fracture", [frac_start], [frac_end])

    def test_fracture_and_constraints(self):
        frac_start = np.array([[0.5], [0.25]])
        frac_end = np.array([[0.5], [0.75]])
        constraint_start = np.array([[0.25], [0.5]])
        constraint_end = np.array([[0.5], [0.5]])
        p = np.hstack((frac_start, frac_end, constraint_start, constraint_end))
        e = np.array([[0, 2], [1, 3]])
        fractures = pts_edges_to_linefractures(p, e)
        network = pp.create_fracture_network(fractures, unit_domain(2))

        constraints = np.array([1])
        mesh_args = {"mesh_size_frac": 1}
        mdg = network.mesh(mesh_args, constraints=constraints)
        g = mdg.subdomains(dim=2)[0]

        self.check_boundary_faces(g, 2)

        self.check_auxiliary_fracture_faces(g, "fracture", [frac_start], [frac_end])
        self.check_auxiliary_fracture_faces(
            g, "auxiliary_line", [constraint_start], [constraint_end], 1
        )
