import numpy as np
import porepy as pp
from porepy.applications.test_utils.arrays import compare_arrays
from porepy.fracs.utils import pts_edges_to_linefractures


class TestFractureNetwork2d:
    def check_mdg_from_polytopal_2d_domain(self, domain, network, num_dom):
        # Helper method to check that meshing a 2d domain with a polytopal (i.e., not
        # rectangular) domain results works as expected. Three tests are performed:
        # 1. That imposing the external boundary does not change the domain
        # 2. That the generated mesh has the expected number of subdomains
        # 3. That the generated mesh respects the domain.polytope lines and that
        #    boundary nodes and faces are correctly tagged.

        # check if the domain in a fracture network is preserved after
        # calling impose_external_boundary
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

    def compare_dictionaries(self, a: dict[str, pp.number], b: dict[str, pp.number]):
        ka = list(a.keys())
        kb = list(b.keys())

        if len(ka) != len(kb):
            return False

        common = set(a) - (set(a) - set(b))
        if len(ka) != len(common):
            return False
        for k in common:
            if a[k] != b[k]:
                return False
        return True

    def setup_class(self):
        self.pts = np.array([[0, 2, 1, 1], [0, 0, 0, 1]])
        self.e = np.array([[0, 2], [1, 3]])
        self.fracs = pts_edges_to_linefractures(self.pts, self.e)
        self.domain = pp.Domain({"xmin": 0, "xmax": 5, "ymin": -1, "ymax": 5})
        self.small_domain = pp.Domain({"xmin": -1, "xmax": 1.5, "ymin": -1, "ymax": 5})

    def test_snap_fractures(self):
        p = np.array([[0, 2, 1, 1], [0, 0, 1e-3, 1]])
        e = np.array([[0, 2], [1, 3]])
        fracs = pts_edges_to_linefractures(p, e)
        network = pp.create_fracture_network(fracs)
        snapped = network.snapped_copy(tol=1e-2)

        known_points = np.array([[0, 2, 1, 1], [0, 0, 0, 1]])
        assert compare_arrays(known_points, snapped._pts)

        snapped_2 = network.snapped_copy(tol=1e-4)
        assert compare_arrays(p, snapped_2._pts)

    def test_split_intersections(self):
        network = pp.create_fracture_network(self.fracs)

        split_network = network.copy_with_split_intersections()
        assert compare_arrays(split_network._pts, self.pts)
        assert split_network._edges.shape[1] == 3

    def test_constrain_to_domain(self):
        network = pp.create_fracture_network(self.fracs, self.domain)
        new_network = network.constrain_to_domain()
        assert compare_arrays(self.pts, new_network._pts)

        small_network = network.constrain_to_domain(self.small_domain)
        known_points = np.array([[0, 1.5, 1, 1], [0, 0, 0, 1]])
        assert compare_arrays(known_points, small_network._pts)

    def test_get_points(self):
        p = self.pts
        e = self.e
        fracs = self.fracs

        network = pp.create_fracture_network(fracs)

        start = network.start_points()
        assert compare_arrays(start, p[:, e[0]])
        end = network.end_points()
        assert compare_arrays(end, p[:, e[1]])

        # Then index based
        start = network.start_points(frac_index=0)
        assert compare_arrays(start, p[:, 0].reshape((-1, 1)))
        end = network.end_points(frac_index=0)
        assert compare_arrays(end, p[:, 1].reshape((-1, 1)))

    def test_length(self):
        network = pp.create_fracture_network(self.fracs)
        length = network.length()
        known_length = np.array([2, 1])
        assert np.allclose(length, known_length)

    def test_angle_0_pi(self):
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

    def test_angle(self):
        network = pp.create_fracture_network(self.fracs)
        angle = network.orientation()
        known_orientation = np.array([0, np.pi / 2])
        assert np.logical_or(
            angle == known_orientation, angle - np.pi == known_orientation
        ).all()

    def test_add_networks_no_domain(self):
        network_1 = pp.create_fracture_network(self.fracs)
        p2 = np.array([[0, 2, 1, 1], [0, 0, 0, 1]]) + 2
        e2 = np.array([[0, 2], [1, 3]])
        fracs2 = pts_edges_to_linefractures(p2, e2)
        network_2 = pp.create_fracture_network(fracs2)

        together = network_1.add_network(network_2)

        p_known = np.hstack((self.pts, p2))
        assert compare_arrays(p_known, together._pts)
        # The known edges has 2 rows, thus by testing for equality, we implicitly
        # verify there are no tags in the joint network
        e_known = np.array([[0, 2, 4, 6], [1, 3, 5, 7]])
        assert compare_arrays(together._edges, e_known)

    def test_add_networks_domains(self):
        network_1 = pp.create_fracture_network(self.fracs, self.domain)
        p2 = np.array([[0, 2, 1, 1], [0, 0, 0, 1]]) + 2
        e2 = np.array([[0, 2], [1, 3]])  #
        fracs2 = pts_edges_to_linefractures(p2, e2)
        # A network with no domain
        network_2 = pp.create_fracture_network(fracs2)

        # Add first to second, check domain
        together = network_1.add_network(network_2)
        assert self.compare_dictionaries(
            self.domain.bounding_box, together.domain.bounding_box
        )

        # Add second to first, check domain
        together = network_2.add_network(network_1)
        assert self.compare_dictionaries(
            self.domain.bounding_box, together.domain.bounding_box
        )

        # Assign domain, then add
        network_2.domain = self.domain
        together = network_1.add_network(network_2)
        assert self.compare_dictionaries(
            self.domain.bounding_box, together.domain.bounding_box
        )

        # Assign different domain, check that the sum has a combination of the
        # domain dicts
        network_2.domain = self.small_domain
        together = network_1.add_network(network_2)
        combined_domain = pp.Domain({"xmin": -1, "xmax": 5.0, "ymin": -1, "ymax": 5})
        assert self.compare_dictionaries(
            combined_domain.bounding_box, together.domain.bounding_box
        )

    def test_add_networks_preserve_tags(self):
        network_1 = pp.create_fracture_network(self.fracs)
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
        network_1._edges = np.vstack(
            (network_1._edges, tag1 * np.ones(2, dtype=np.int8))
        )
        together = network_1.add_network(network_2)
        known_tags = np.array([tag1, tag1, tag2, tag2])
        assert np.all(together._edges[2] == known_tags)

    # Below are tests with polytopal domains

    def test_triangle_domain_no_fracs(self):
        # test a grid where the outer domain is a triangle, no fractures
        line_1 = np.array([[0, 1], [0, 0]])
        line_2 = np.array([[1, 0], [0, 1]])
        line_3 = np.array([[0, 0], [1, 0]])
        lines = [line_1, line_2, line_3]
        domain = pp.Domain(polytope=lines)

        network = pp.create_fracture_network(domain=domain)
        self.check_mdg_from_polytopal_2d_domain(domain, network, [0, 0, 1])

    def test_triangle_domain_with_fracs(self):
        # test a grid where the outer domain is a triangle, one fracture immersed in the
        # domain
        line_1 = np.array([[0, 1], [0, 0]])
        line_2 = np.array([[1, 0], [0, 1]])
        line_3 = np.array([[0, 0], [1, 0]])
        lines = [line_1, line_2, line_3]
        domain = pp.Domain(polytope=lines)

        frac = pp.LineFracture(np.array([[0, 1], [0, 1]]))
        network = pp.create_fracture_network([frac], domain=domain)

        self.check_mdg_from_polytopal_2d_domain(domain, network, [0, 1, 1])

    def test_pentagon_domain_with_fracs(self):
        # test a grid where the outer domain is a concave pentagon, one fracture is
        # partially immersed which results in two 1d grids
        line_1 = np.array([[0, 1], [0, 0]])
        line_2 = np.array([[1, 1], [0, 1]])
        line_3 = np.array([[1, 0.5], [1, 0.5]])
        line_4 = np.array([[0.5, 0], [0.5, 1]])
        line_5 = np.array([[0, 0], [1, 0]])
        lines = [line_1, line_2, line_3, line_4, line_5]
        domain = pp.Domain(polytope=lines)

        frac = pp.LineFracture(np.array([[0.2, 0.8], [0.6, 0.6]]))
        network = pp.create_fracture_network([frac], domain=domain)

        self.check_mdg_from_polytopal_2d_domain(domain, network, [0, 2, 1])

    # Test of other methods

    def test_copy(self):
        network_1 = pp.create_fracture_network(self.fracs)

        copy = network_1.copy()
        num_p = self.pts.shape[1]

        network_1._pts = np.random.rand(2, num_p)
        assert np.allclose(copy._pts, self.pts)

    def test_no_snapping(self):
        p = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
        e = np.array([[0, 2], [1, 3]])
        fracs = pts_edges_to_linefractures(p, e)
        network = pp.create_fracture_network(fracs)
        pn, conv = network._snap_fracture_set(p, snap_tol=1e-3)
        assert np.allclose(p, pn)
        assert conv

    def test_snap_to_vertex(self):
        p = np.array([[0, 1, 0, 1], [0, 0, 1e-4, 1]])
        e = np.array([[0, 2], [1, 3]])
        fracs = pts_edges_to_linefractures(p, e)
        network = pp.create_fracture_network(fracs)
        pn, conv = network._snap_fracture_set(p, snap_tol=1e-3)

        p_known = np.array([[0, 1, 0, 1], [0, 0, 0, 1]])
        assert np.allclose(p_known, pn)
        assert conv

    def test_no_snap_to_vertex_small_tol(self):
        # No snapping because the snapping tolerance is small
        p = np.array([[0, 1, 0, 1], [0, 0, 1e-4, 1]])
        e = np.array([[0, 2], [1, 3]])
        fracs = pts_edges_to_linefractures(p, e)
        network = pp.create_fracture_network(fracs)
        pn, conv = network._snap_fracture_set(p, snap_tol=1e-5)

        assert np.allclose(p, pn)
        assert conv

    def test_snap_to_segment(self):
        p = np.array([[0, 1, 0.5, 1], [0, 0, 1e-4, 1]])
        e = np.array([[0, 2], [1, 3]])
        fracs = pts_edges_to_linefractures(p, e)
        network = pp.create_fracture_network(fracs)
        pn, conv = network._snap_fracture_set(p, snap_tol=1e-3)
        p_known = np.array([[0, 1, 0.5, 1], [0, 0, 0, 1]])
        assert np.allclose(p_known, pn)
        assert conv
