"""
Various checks for FractureNetwork2d, FractureNetwork3d, and create_fracture_newtork().

Also tests for utility functions for generation of default domains.
"""
from __future__ import annotations

import unittest
import pytest

import numpy as np

import porepy as pp
from tests import test_utils
from porepy.fracs.utils import pts_edges_to_linefractures
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.grids.standard_grids.utils import unit_domain


class TestCreateFractureNetwork:
    """Contain tests that check the correct behaviour of pp.create_fracture_network()"""

    # Set line fracture plane fracture as fixture functions to reuse them later
    @pytest.fixture(scope="class")
    def line_fracture(self):
        return pp.LineFracture(np.array([[0.5, 0.5], [0, 1]]))

    @pytest.fixture(scope="class")
    def plane_fracture(self):
        return pp.PlaneFracture(
            np.array([[0, 2, 2, 0], [0, 2, 2, 0], [-1, -1, 1, 1]])
        )

    # -----> Test #1
    @pytest.mark.parametrize("fractures", [[], None])
    def test_error_raised_if_fractures_and_domain_are_none(self, fractures):
        """Check that an error is raised when both fractures and domain are given as
        None/ empty"""
        with pytest.raises(ValueError) as excinfo:
            msg = "'fractures' and 'domain' cannot both be empty/None."
            pp.create_fracture_network(fractures, None)
        assert msg in str(excinfo.value)

    # -----> Test #2
    @pytest.fixture(scope="function")
    def fractures_lists(self, line_fracture, plane_fracture):
        fractures_1 = [1]
        fractures_2 = [line_fracture, "just_another_fracture"]
        fractures_3 = [line_fracture, plane_fracture]
        fractures_4 = [plane_fracture, line_fracture]
        return [fractures_1, fractures_2, fractures_3, fractures_4]

    @pytest.mark.parametrize("tst_idx", [0, 1, 2, 3])
    def test_error_if_fractures_has_heterogeneous_types(
            self, fractures_lists, tst_idx
    ):
        """Check that a type error is raised if the list of fractures contain
        non-homogeneous types, e.g., it should be all PlaneFractures or
        LineFractures."""
        with pytest.raises(TypeError) as excinfo:
            msg = "All fracture objects must be of the same type."
            pp.create_fracture_network(fractures_lists[tst_idx], None)
        assert msg in str(excinfo.value)

    # -----> Test #3
    @pytest.mark.parametrize("domain", [unit_domain(2), unit_domain(3)])
    def test_empty_list_and_none_create_the_same_network(self, domain):
        """Checks that the same fracture network is generated if an empty fracture
        list OR None is given."""
        fn_none = pp.create_fracture_network(None, domain)
        fn_empty_list = pp.create_fracture_network([], domain)
        assert fn_none.num_frac() == fn_empty_list.num_frac()
        assert fn_none.domain == fn_empty_list.domain

    # -----> Test #4
    def test_error_raised_if_different_dimensions(self, line_fracture):
        """Checks that an error is raised if the dimensions inferred from the set of
        fractures and the domain are different."""
        with pytest.raises(ValueError) as excinfo:
            msg = "The dimensions inferred from 'fractures' and 'domain' do not match."
            pp.create_fracture_network([line_fracture], unit_domain(3))
        assert msg in str(excinfo.value)

    # -----> Test #5
    def test_warning_raised_if_run_checks_true_for_dim_not_3(self):
        """Checks if the warning is properly raised when ``run_checks=True`` for a
        fracture network with a dimensionality different from 3."""
        warn_msg = "'run_checks=True' has no effect if dimension != 3."
        with pytest.warns() as record:
            pp.create_fracture_network(None, domain=unit_domain(2), run_checks=True)
        assert str(record[0].message) == warn_msg

    # -----> Test #6
    @pytest.fixture(scope="function")
    def fractures_list_2d(self, line_fracture):
        list_1 = []
        list_2 = [line_fracture]
        return [list_1, list_2]

    @pytest.mark.parametrize("tst_idx", [0, 1])
    def test_create_fracture_network_2d(self, fractures_list_2d, tst_idx):
        """Test if 2d fracture networks constructed with FractureNetwork2d and with
        create_fracture_network() are the same."""
        fn_with_function = pp.create_fracture_network(
            fractures_list_2d[tst_idx], unit_domain(2)
        )
        fn_with_class = FractureNetwork2d(fractures_list_2d[tst_idx], unit_domain(2))
        assert isinstance(fn_with_function, pp.FractureNetwork2d)
        assert fn_with_function.num_frac() == fn_with_class.num_frac()
        assert fn_with_function.domain == fn_with_class.domain

    # -----> Test #7
    @pytest.fixture(scope="function")
    def fractures_list_3d(self, plane_fracture):
        list_1 = []
        list_2 = [plane_fracture]
        return [list_1, list_2]

    @pytest.mark.parametrize("tst_idx", [0, 1])
    def test_create_fracture_network_3d(self, fractures_list_3d, tst_idx):
        """Test if 3d fracture networks constructed with FractureNetwork3d and with
        create_fracture_network() are the same."""
        fn_with_function = pp.create_fracture_network(
            fractures_list_3d[tst_idx], unit_domain(3)
        )
        fn_with_class = FractureNetwork3d(fractures_list_3d[tst_idx], unit_domain(3))
        assert isinstance(fn_with_function, pp.FractureNetwork3d)
        assert fn_with_function.num_frac() == fn_with_class.num_frac()
        assert fn_with_function.domain == fn_with_class.domain


class TestFractureNetwork2d(unittest.TestCase):

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

    def setUp(self):
        self.pts = np.array([[0, 2, 1, 1], [0, 0, 0, 1]])
        self.e = np.array([[0, 2], [1, 3]])
        self.fracs = pts_edges_to_linefractures(self.pts, self.e)
        self.domain = pp.Domain({"xmin": 0, "xmax": 5, "ymin": -1, "ymax": 5})
        self.small_domain = pp.Domain({"xmin": -1, "xmax": 1.5, "ymin": -1, "ymax": 5})

    def test_snap_fractures(self):

        p = np.array([[0, 2, 1, 1], [0, 0, 1e-3, 1]])
        e = np.array([[0, 2], [1, 3]])
        fracs = pts_edges_to_linefractures(p, e)
        network = pp.FractureNetwork2d(fracs)
        snapped = network.snapped_copy(tol=1e-2)

        known_points = np.array([[0, 2, 1, 1], [0, 0, 0, 1]])
        self.assertTrue(test_utils.compare_arrays(known_points, snapped._pts))

        snapped_2 = network.snapped_copy(tol=1e-4)
        self.assertTrue(test_utils.compare_arrays(p, snapped_2._pts))

    def test_split_intersections(self):
        network = pp.FractureNetwork2d(self.fracs)

        split_network = network.copy_with_split_intersections()
        self.assertTrue(test_utils.compare_arrays(split_network._pts, self.pts))
        self.assertTrue(split_network._edges.shape[1] == 3)

    def test_constrain_to_domain(self):
        network = pp.FractureNetwork2d(self.fracs, self.domain)
        new_network = network.constrain_to_domain()
        self.assertTrue(test_utils.compare_arrays(self.pts, new_network._pts))

        small_network = network.constrain_to_domain(self.small_domain)
        known_points = np.array([[0, 1.5, 1, 1], [0, 0, 0, 1]])
        self.assertTrue(test_utils.compare_arrays(known_points, small_network._pts))

    def test_get_points(self):
        p = self.pts
        e = self.e
        fracs = self.fracs

        network = pp.FractureNetwork2d(fracs)

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
        network = pp.FractureNetwork2d(self.fracs)
        length = network.length()
        known_length = np.array([2, 1])
        self.assertTrue(np.allclose(length, known_length))

    def test_angle_0_pi(self):
        num_frac = 10
        start = np.zeros((2, num_frac))

        end = 2 * np.random.rand(2, num_frac) - 1
        p = np.hstack((start, end))
        e = np.vstack((np.arange(num_frac), num_frac + np.arange(num_frac)))
        fracs = pts_edges_to_linefractures(p, e)
        network = pp.FractureNetwork2d(fracs)
        angle = network.orientation()

        self.assertTrue(np.all(angle >= 0))
        self.assertTrue(np.all(angle < np.pi))

    def test_angle(self):
        network = pp.FractureNetwork2d(self.fracs)
        angle = network.orientation()
        known_orientation = np.array([0, np.pi / 2])
        self.assertTrue(
            np.logical_or(
                angle == known_orientation, angle - np.pi == known_orientation
            ).all()
        )

    def test_add_networks_no_domain(self):
        network_1 = pp.FractureNetwork2d(self.fracs)
        p2 = np.array([[0, 2, 1, 1], [0, 0, 0, 1]]) + 2
        e2 = np.array([[0, 2], [1, 3]])
        fracs2 = pts_edges_to_linefractures(p2, e2)
        network_2 = pp.FractureNetwork2d(fracs2)

        together = network_1.add_network(network_2)

        p_known = np.hstack((self.pts, p2))
        self.assertTrue(test_utils.compare_arrays(p_known, together._pts))
        # The known edges has 2 rows, thus by testing for equality, we implicitly
        # verify there are no tags in the joint network
        e_known = np.array([[0, 2, 4, 6], [1, 3, 5, 7]])
        self.assertTrue(test_utils.compare_arrays(together._edges, e_known))

    def test_add_networks_domains(self):
        network_1 = pp.FractureNetwork2d(self.fracs, self.domain)
        p2 = np.array([[0, 2, 1, 1], [0, 0, 0, 1]]) + 2
        e2 = np.array([[0, 2], [1, 3]])  #
        fracs2 = pts_edges_to_linefractures(p2, e2)
        # A network with no domain
        network_2 = pp.FractureNetwork2d(fracs2)

        # Add first to second, check domain
        together = network_1.add_network(network_2)
        self.assertTrue(self.compare_dictionaries(
            self.domain.bounding_box, together.domain.bounding_box)
        )

        # Add second to first, check domain
        together = network_2.add_network(network_1)
        self.assertTrue(self.compare_dictionaries(
            self.domain.bounding_box, together.domain.bounding_box)
        )

        # Assign domain, then add
        network_2.domain = self.domain
        together = network_1.add_network(network_2)
        self.assertTrue(self.compare_dictionaries(
            self.domain.bounding_box, together.domain.bounding_box)
        )

        # Assign different domain, check that the sum has a combination of the
        # domain dicts
        network_2.domain = self.small_domain
        together = network_1.add_network(network_2)
        combined_domain = pp.Domain({"xmin": -1, "xmax": 5.0, "ymin": -1, "ymax": 5})
        self.assertTrue(
            self.compare_dictionaries(
                combined_domain.bounding_box, together.domain.bounding_box)
        )

    def test_add_networks_preserve_tags(self):
        network_1 = pp.FractureNetwork2d(self.fracs)
        p2 = np.array([[0, 2, 1, 1], [0, 0, 0, 1]]) + 2
        # Network 2 has tags
        tag2 = 1
        e2 = np.array([[0, 2], [1, 3], [1, 1]])
        fracs2 = pts_edges_to_linefractures(p2, e2)
        network_2 = pp.FractureNetwork2d(fracs2)

        # Add networks, check tags
        together = network_1.add_network(network_2)
        self.assertTrue(np.all(together._edges[2, 2:4] == tag2))

        # Add networks reverse order
        together = network_2.add_network(network_1)
        self.assertTrue(np.all(together._edges[2, :2] == tag2))

        # Assign tags to network1
        tag1 = 2
        # Had to change this to ensure that edges still has ``dtype np.int8``
        network_1._edges = np.vstack((network_1._edges, tag1 * np.ones(2, dtype=np.int8)))
        together = network_1.add_network(network_2)
        known_tags = np.array([tag1, tag1, tag2, tag2])
        self.assertTrue(np.all(together._edges[2] == known_tags))

    def test_copy(self):
        network_1 = pp.FractureNetwork2d(self.fracs)

        copy = network_1.copy()
        num_p = self.pts.shape[1]

        network_1._pts = np.random.rand(2, num_p)
        self.assertTrue(np.allclose(copy._pts, self.pts))

    def test_no_snapping(self):
        p = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
        e = np.array([[0, 2], [1, 3]])
        fracs = pts_edges_to_linefractures(p, e)
        network = pp.FractureNetwork2d(fracs)
        pn, conv = network._snap_fracture_set(p, snap_tol=1e-3)
        self.assertTrue(np.allclose(p, pn))
        self.assertTrue(conv)

    def test_snap_to_vertex(self):
        p = np.array([[0, 1, 0, 1], [0, 0, 1e-4, 1]])
        e = np.array([[0, 2], [1, 3]])
        fracs = pts_edges_to_linefractures(p, e)
        network = pp.FractureNetwork2d(fracs)
        pn, conv = network._snap_fracture_set(p, snap_tol=1e-3)

        p_known = np.array([[0, 1, 0, 1], [0, 0, 0, 1]])
        self.assertTrue(np.allclose(p_known, pn))
        self.assertTrue(conv)

    def test_no_snap_to_vertex_small_tol(self):
        # No snapping because the snapping tolerance is small
        p = np.array([[0, 1, 0, 1], [0, 0, 1e-4, 1]])
        e = np.array([[0, 2], [1, 3]])
        fracs = pts_edges_to_linefractures(p, e)
        network = pp.FractureNetwork2d(fracs)
        pn, conv = network._snap_fracture_set(p, snap_tol=1e-5)

        self.assertTrue(np.allclose(p, pn))
        self.assertTrue(conv)

    def test_snap_to_segment(self):
        p = np.array([[0, 1, 0.5, 1], [0, 0, 1e-4, 1]])
        e = np.array([[0, 2], [1, 3]])
        fracs = pts_edges_to_linefractures(p, e)
        network = pp.FractureNetwork2d(fracs)
        pn, conv = network._snap_fracture_set(p, snap_tol=1e-3)
        p_known = np.array([[0, 1, 0.5, 1], [0, 0, 0, 1]])
        self.assertTrue(np.allclose(p_known, pn))
        self.assertTrue(conv)


class TestFractureNetwork3dBoundingBox(unittest.TestCase):
    def test_single_fracture(self):
        # Test of method FractureNetwork.bounding_box() to inquire about
        # network extent
        f1 = pp.PlaneFracture(
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

    def test_single_fracture_aligned_with_axis(self):
        # Test of method FractureNetwork.bounding_box() to inquire about
        # network extent
        f1 = pp.PlaneFracture(
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
        f1 = pp.PlaneFracture(
            np.array([[0, 2, 2, 0], [0, 0, 1, 1], [0, 0, 1, 1]]), check_convexity=False
        )

        f2 = pp.PlaneFracture(
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
        f1 = pp.PlaneFracture(
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
        network.impose_external_boundary(domain=pp.Domain(external_boundary))
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
