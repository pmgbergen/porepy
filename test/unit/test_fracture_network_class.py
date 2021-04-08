"""
Various checks of the FractureNetwork2d and 3d class

Also test unitily function for generation of defalut domains.
"""

import unittest
from test import test_utils

import numpy as np

import porepy as pp


class TestFractureNetwork2d(unittest.TestCase):
    def compare_dictionaries(self, a, b):
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

    def setUp(self):
        self.p = np.array([[0, 2, 1, 1], [0, 0, 0, 1]])
        self.e = np.array([[0, 2], [1, 3]])
        self.domain = {"xmin": 0, "xmax": 5, "ymin": -1, "ymax": 5}
        self.small_domain = {"xmin": -1, "xmax": 1.5, "ymin": -1, "ymax": 5}

    def test_snap_fractures(self):

        p = np.array([[0, 2, 1, 1], [0, 0, 1e-3, 1]])
        e = np.array([[0, 2], [1, 3]])

        network = pp.FractureNetwork2d(p, e)
        snapped = network.snapped_copy(tol=1e-2)

        known_points = np.array([[0, 2, 1, 1], [0, 0, 0, 1]])
        self.assertTrue(test_utils.compare_arrays(known_points, snapped.pts))

        snapped_2 = network.snapped_copy(tol=1e-4)
        self.assertTrue(test_utils.compare_arrays(p, snapped_2.pts))

    def test_split_intersections(self):
        network = pp.FractureNetwork2d(self.p, self.e)

        split_network = network.copy_with_split_intersections()
        self.assertTrue(test_utils.compare_arrays(split_network.pts, self.p))
        self.assertTrue(split_network.edges.shape[1] == 3)

    def test_constrain_to_domain(self):
        network = pp.FractureNetwork2d(self.p, self.e, self.domain)
        new_network = network.constrain_to_domain()
        self.assertTrue(test_utils.compare_arrays(self.p, new_network.pts))

        small_network = network.constrain_to_domain(self.small_domain)
        known_points = np.array([[0, 1.5, 1, 1], [0, 0, 0, 1]])
        self.assertTrue(test_utils.compare_arrays(known_points, small_network.pts))

    def test_get_points(self):
        p = self.p
        e = self.e

        network = pp.FractureNetwork2d(p, e)

        start = network.start_points()
        self.assertTrue(test_utils.compare_arrays(start, p[:, e[0]]))
        end = network.end_points()
        self.assertTrue(test_utils.compare_arrays(end, p[:, e[1]]))

        # Then index based
        start = network.start_points(fi=0)
        self.assertTrue(test_utils.compare_arrays(start, p[:, 0].reshape((-1, 1))))
        end = network.end_points(fi=0)
        self.assertTrue(test_utils.compare_arrays(end, p[:, 1].reshape((-1, 1))))

    def test_length(self):
        network = pp.FractureNetwork2d(self.p, self.e)
        length = network.length()
        known_length = np.array([2, 1])
        self.assertTrue(np.allclose(length, known_length))

    def test_angle_0_pi(self):
        num_frac = 10
        start = np.zeros((2, num_frac))

        end = 2 * np.random.rand(2, num_frac) - 1
        p = np.hstack((start, end))
        e = np.vstack((np.arange(num_frac), num_frac + np.arange(num_frac)))
        network = pp.FractureNetwork2d(p, e)
        angle = network.orientation()

        self.assertTrue(np.all(angle >= 0))
        self.assertTrue(np.all(angle < np.pi))

    def test_angle(self):
        network = pp.FractureNetwork2d(self.p, self.e)
        angle = network.orientation()
        known_orientation = np.array([0, np.pi / 2])
        self.assertTrue(
            np.logical_or(
                angle == known_orientation, angle - np.pi == known_orientation
            ).all()
        )

    def test_add_networks_no_domain(self):
        network_1 = pp.FractureNetwork2d(self.p, self.e)
        p2 = np.array([[0, 2, 1, 1], [0, 0, 0, 1]]) + 2
        e2 = np.array([[0, 2], [1, 3]])
        network_2 = pp.FractureNetwork2d(p2, e2)

        together = network_1.add_network(network_2)

        p_known = np.hstack((self.p, p2))
        self.assertTrue(test_utils.compare_arrays(p_known, together.pts))
        # The known edges has 2 rows, thus by testing for equality, we implicitly
        # verify there are no tags in the joint network
        e_known = np.array([[0, 2, 4, 6], [1, 3, 5, 7]])
        self.assertTrue(test_utils.compare_arrays(together.edges, e_known))

    def test_add_networks_domains(self):
        network_1 = pp.FractureNetwork2d(self.p, self.e, self.domain)
        p2 = np.array([[0, 2, 1, 1], [0, 0, 0, 1]]) + 2
        e2 = np.array([[0, 2], [1, 3]])
        # A network with no domain
        network_2 = pp.FractureNetwork2d(p2, e2)

        # Add first to second, check domain
        together = network_1.add_network(network_2)
        self.assertTrue(self.compare_dictionaries(self.domain, together.domain))

        # Add second to first, check domain
        together = network_2.add_network(network_1)
        self.assertTrue(self.compare_dictionaries(self.domain, together.domain))

        # Assign domain, then add
        network_2.domain = self.domain
        together = network_1.add_network(network_2)
        self.assertTrue(self.compare_dictionaries(self.domain, together.domain))

        # Assign different domain, check that the sum has a combination of the
        # domain dicts
        network_2.domain = self.small_domain
        together = network_1.add_network(network_2)
        combined_domain = {"xmin": -1, "xmax": 5.0, "ymin": -1, "ymax": 5}
        self.assertTrue(self.compare_dictionaries(combined_domain, together.domain))

    def test_add_networks_preserve_tags(self):
        network_1 = pp.FractureNetwork2d(self.p, self.e)
        p2 = np.array([[0, 2, 1, 1], [0, 0, 0, 1]]) + 2
        # Network 2 has tags
        tag2 = 1
        e2 = np.array([[0, 2], [1, 3], [1, 1]])
        network_2 = pp.FractureNetwork2d(p2, e2)

        # Add networks, check tags
        together = network_1.add_network(network_2)
        self.assertTrue(np.all(together.edges[2, 2:4] == tag2))

        # Add networks reverse order
        together = network_2.add_network(network_1)
        self.assertTrue(np.all(together.edges[2, :2] == tag2))

        # Assign tags to network1
        tag1 = 2
        network_1.edges = np.vstack((network_1.edges, tag1 * np.ones(2)))
        together = network_1.add_network(network_2)
        known_tags = np.array([tag1, tag1, tag2, tag2])
        self.assertTrue(np.all(together.edges[2] == known_tags))

    def test_copy(self):
        network_1 = pp.FractureNetwork2d(self.p, self.e)

        copy = network_1.copy()
        num_p = self.p.shape[1]

        network_1.pts = np.random.rand(2, num_p)
        self.assertTrue(np.allclose(copy.pts, self.p))

    def test_no_snapping(self):
        p = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
        e = np.array([[0, 2], [1, 3]])

        network = pp.FractureNetwork2d(p, e)
        pn, conv = network._snap_fracture_set(p, snap_tol=1e-3)
        self.assertTrue(np.allclose(p, pn))
        self.assertTrue(conv)

    def test_snap_to_vertex(self):
        p = np.array([[0, 1, 0, 1], [0, 0, 1e-4, 1]])
        e = np.array([[0, 2], [1, 3]])
        network = pp.FractureNetwork2d(p, e)
        pn, conv = network._snap_fracture_set(p, snap_tol=1e-3)

        p_known = np.array([[0, 1, 0, 1], [0, 0, 0, 1]])
        self.assertTrue(np.allclose(p_known, pn))
        self.assertTrue(conv)

    def test_no_snap_to_vertex_small_tol(self):
        # No snapping because the snapping tolerance is small
        p = np.array([[0, 1, 0, 1], [0, 0, 1e-4, 1]])
        e = np.array([[0, 2], [1, 3]])
        network = pp.FractureNetwork2d(p, e)
        pn, conv = network._snap_fracture_set(p, snap_tol=1e-5)

        self.assertTrue(np.allclose(p, pn))
        self.assertTrue(conv)

    def test_snap_to_segment(self):
        p = np.array([[0, 1, 0.5, 1], [0, 0, 1e-4, 1]])
        e = np.array([[0, 2], [1, 3]])

        network = pp.FractureNetwork2d(p, e)
        pn, conv = network._snap_fracture_set(p, snap_tol=1e-3)
        p_known = np.array([[0, 1, 0.5, 1], [0, 0, 0, 1]])
        self.assertTrue(np.allclose(p_known, pn))
        self.assertTrue(conv)



class TestFractureNetwork3dBoundingBox(unittest.TestCase):
    def test_single_fracture(self):
        # Test of method FractureNetwork.bounding_box() to inquire about
        # network extent
        f1 = pp.Fracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1]]), check_convexity=False
        )

        network = pp.FractureNetwork3d([f1])
        d = network.bounding_box()

        self.assertTrue(d["xmin"] == 0)
        self.assertTrue(d["xmax"] == 1)
        self.assertTrue(d["ymin"] == 0)
        self.assertTrue(d["ymax"] == 1)
        self.assertTrue(d["zmin"] == 0)
        self.assertTrue(d["zmax"] == 1)

    def test_sinle_fracture_aligned_with_axis(self):
        # Test of method FractureNetwork.bounding_box() to inquire about
        # network extent
        f1 = pp.Fracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]), check_convexity=False
        )

        network = pp.FractureNetwork3d([f1])
        d = network.bounding_box()

        self.assertTrue(d["xmin"] == 0)
        self.assertTrue(d["xmax"] == 1)
        self.assertTrue(d["ymin"] == 0)
        self.assertTrue(d["ymax"] == 1)
        self.assertTrue(d["zmin"] == 0)
        self.assertTrue(d["zmax"] == 0)

    def test_two_fractures(self):
        # Test of method FractureNetwork.bounding_box() to inquire about
        # network extent
        f1 = pp.Fracture(
            np.array([[0, 2, 2, 0], [0, 0, 1, 1], [0, 0, 1, 1]]), check_convexity=False
        )

        f2 = pp.Fracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [-1, -1, 1, 1]]),
            check_convexity=False,
        )

        network = pp.FractureNetwork3d([f1, f2])
        d = network.bounding_box()

        self.assertTrue(d["xmin"] == 0)
        self.assertTrue(d["xmax"] == 2)
        self.assertTrue(d["ymin"] == 0)
        self.assertTrue(d["ymax"] == 1)
        self.assertTrue(d["zmin"] == -1)
        self.assertTrue(d["zmax"] == 1)

    def test_external_boundary_added(self):
        # Test of method FractureNetwork.bounding_box() when an external
        # boundary is added. Thus double as test of this adding.
        f1 = pp.Fracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1]]), check_convexity=False
        )

        network = pp.FractureNetwork3d([f1])

        external_boundary = {
            "xmin": -1,
            "xmax": 2,
            "ymin": -1,
            "ymax": 2,
            "zmin": -1,
            "zmax": 2,
        }
        network.impose_external_boundary(domain=external_boundary)
        d = network.bounding_box()

        self.assertTrue(d["xmin"] == external_boundary["xmin"])
        self.assertTrue(d["xmax"] == external_boundary["xmax"])
        self.assertTrue(d["ymin"] == external_boundary["ymin"])
        self.assertTrue(d["ymax"] == external_boundary["ymax"])
        self.assertTrue(d["zmin"] == external_boundary["zmin"])
        self.assertTrue(d["zmax"] == external_boundary["zmax"])


class TestDomain(unittest.TestCase):
    def check_key_value(self, domain, keys, values):
        tol = 1e-12
        for dim, key in enumerate(keys):
            if key in domain:
                self.assertTrue(abs(domain.pop(key) - values[dim]) < tol)
            else:
                self.assertTrue(False)  # Did not find correct key

    def test_unit_cube(self):
        domain = pp.UnitCubeDomain()
        keys = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
        values = [0, 0, 0, 1, 1, 1]
        self.check_key_value(domain, keys, values)
        if len(domain) != 0:
            self.assertTrue(False)  # Domain should be empty now

    def test_unit_square(self):
        domain = pp.UnitSquareDomain()
        keys = ["xmin", "ymin", "xmax", "ymax"]
        values = [0, 0, 1, 1]
        self.check_key_value(domain, keys, values)
        if len(domain) != 0:
            self.assertTrue(False)  # Domain should be empty now

    def test_cube(self):
        physdims = [3.14, 1, 5]
        domain = pp.CubeDomain(physdims)
        keys = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
        values = [0, 0, 0] + physdims

        self.check_key_value(domain, keys, values)
        if len(domain) != 0:
            self.assertTrue(False)  # Domain should be empty now

    def test_squre(self):
        physdims = [2.71, 1e5]
        domain = pp.SquareDomain(physdims)
        keys = ["xmin", "ymin", "xmax", "ymax"]
        values = [0, 0] + physdims

        self.check_key_value(domain, keys, values)
        if len(domain) != 0:
            self.assertTrue(False)  # Domain should be empty now

    def test_negative_domain(self):
        physdims = [1, -1]
        with self.assertRaises(ValueError):
            pp.SquareDomain(physdims)


if __name__ == "__main__":
    unittest.main()
