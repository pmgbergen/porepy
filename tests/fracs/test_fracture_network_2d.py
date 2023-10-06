"""Testing functionality related to FractureNetwork2d."""
import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.arrays import compare_arrays
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.line_fracture import LineFracture
from porepy.fracs.utils import pts_edges_to_linefractures
from porepy.geometry.domain import Domain


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
