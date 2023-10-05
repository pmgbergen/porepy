import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.arrays import compare_arrays
from porepy.fracs.plane_fracture import PlaneFracture
from porepy.geometry.domain import Domain
from porepy.grids.standard_grids.utils import unit_domain


# ---------- Testing impose_external_boundary ----------
# Test of algorithm for constraining a fracture a bounding box.

# Since that algorithm uses fracture intersection methods, the tests functions as
# partial test for the wider fracture intersection framework as well. Full tests
# of the latter are too time consuming to fit into a unit test.

# Now the boundary is defined as set of "fake" fractures, all fracture network
# have 2*dim additional fractures (hence the + 6 in the assertions)


@pytest.fixture
def domain() -> Domain:
    return unit_domain(3)


@pytest.fixture
def fracture() -> PlaneFracture:
    return PlaneFracture(
        np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]]),
        check_convexity=False,
    )


def test_completely_outside_lower(domain: Domain, fracture: PlaneFracture):
    fracture.pts[0] -= 2
    network = pp.create_fracture_network([fracture])
    network.impose_external_boundary(domain)
    assert len(network.fractures) == (0 + 6)


def test_outside_west_bottom(domain: Domain, fracture: PlaneFracture):
    fracture.pts[0] -= 0.5
    fracture.pts[2] -= 1.5
    network = pp.create_fracture_network([fracture])
    network.impose_external_boundary(domain)
    assert len(network.fractures) == (0 + 6)


def test_intersect_one(domain: Domain, fracture: PlaneFracture):
    fracture.pts[0] -= 0.5
    fracture.pts[2, :] = [0.2, 0.2, 0.8, 0.8]
    network = pp.create_fracture_network([fracture])
    network.impose_external_boundary(domain)
    p_known = np.array([[0.0, 0.5, 0.5, 0], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])
    assert len(network.fractures) == (1 + 6)
    p_comp = network.fractures[0].pts
    assert compare_arrays(p_comp, p_known, sort=True, tol=1e-5)


def test_intersect_two_same(domain: Domain, fracture: PlaneFracture):
    f = fracture
    f.pts[0, :] = [-0.5, 1.5, 1.5, -0.5]
    f.pts[2, :] = [0.2, 0.2, 0.8, 0.8]
    network = pp.create_fracture_network([f])
    network.impose_external_boundary(domain)
    p_known = np.array([[0.0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])
    assert len(network.fractures) == (1 + 6)
    p_comp = network.fractures[0].pts
    assert compare_arrays(p_comp, p_known, sort=True, tol=1e-5)


def test_incline_in_plane(domain: Domain, fracture: PlaneFracture):
    f = fracture
    f.pts[0] -= 0.5
    f.pts[2, :] = [0, -0.5, 0.5, 1]
    network = pp.create_fracture_network([f])
    network.impose_external_boundary(domain)
    p_known = np.array(
        [[0.0, 0.5, 0.5, 0], [0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.5, 0.75]]
    )
    assert len(network.fractures) == (1 + 6)
    p_comp = network.fractures[0].pts
    assert compare_arrays(p_comp, p_known, sort=True, tol=1e-5)


def test_full_incline(domain: Domain, fracture: PlaneFracture):
    p = np.array([[-0.5, 0.5, 0.5, -0.5], [0.5, 0.5, 1.5, 1.5], [-0.5, -0.5, 1, 1]])
    f = pp.PlaneFracture(p, check_convexity=False)
    network = pp.create_fracture_network([f])
    network.impose_external_boundary(domain)
    p_known = np.array(
        [[0.0, 0.5, 0.5, 0], [5.0 / 6, 5.0 / 6, 1, 1], [0.0, 0.0, 0.25, 0.25]]
    )
    assert len(network.fractures) == (1 + 6)
    p_comp = network.fractures[0].pts
    assert compare_arrays(p_comp, p_known, sort=True, tol=1e-5)


class TestFractureNetwork3dBoundingBox:
    def test_single_fracture(self):
        # Test of method FractureNetwork.bounding_box() to inquire about
        # network extent
        f1 = PlaneFracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1]]), check_convexity=False
        )

        network = pp.create_fracture_network([f1])
        d = network.bounding_box()

        assert d["xmin"] == 0
        assert d["xmax"] == 1
        assert d["ymin"] == 0
        assert d["ymax"] == 1
        assert d["zmin"] == 0
        assert d["zmax"] == 1

    def test_single_fracture_aligned_with_axis(self):
        # Test of method FractureNetwork.bounding_box() to inquire about
        # network extent
        f1 = PlaneFracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]), check_convexity=False
        )

        network = pp.create_fracture_network([f1])
        d = network.bounding_box()

        assert d["xmin"] == 0
        assert d["xmax"] == 1
        assert d["ymin"] == 0
        assert d["ymax"] == 1
        assert d["zmin"] == 0
        assert d["zmax"] == 0

    def test_two_fractures(self):
        # Test of method FractureNetwork.bounding_box() to inquire about
        # network extent
        f1 = PlaneFracture(
            np.array([[0, 2, 2, 0], [0, 0, 1, 1], [0, 0, 1, 1]]), check_convexity=False
        )
        f2 = PlaneFracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [-1, -1, 1, 1]]),
            check_convexity=False,
        )

        network = pp.create_fracture_network([f1, f2])
        d = network.bounding_box()

        assert d["xmin"] == 0
        assert d["xmax"] == 2
        assert d["ymin"] == 0
        assert d["ymax"] == 1
        assert d["zmin"] == -1
        assert d["zmax"] == 1

    def test_external_boundary_added(self):
        # Test of method FractureNetwork.bounding_box() when an external
        # boundary is added. Thus double as test of this adding.
        fracture_1 = PlaneFracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1]]), check_convexity=False
        )

        network = pp.create_fracture_network([fracture_1])

        external_box = {
            "xmin": -1,
            "xmax": 2,
            "ymin": -1,
            "ymax": 2,
            "zmin": -1,
            "zmax": 2,
        }
        domain_to_impose = pp.Domain(bounding_box=external_box)
        network.impose_external_boundary(domain=domain_to_impose)

        assert network.domain.bounding_box["xmin"] == external_box["xmin"]
        assert network.domain.bounding_box["xmax"] == external_box["xmax"]
        assert network.domain.bounding_box["ymin"] == external_box["ymin"]
        assert network.domain.bounding_box["ymax"] == external_box["ymax"]
        assert network.domain.bounding_box["zmin"] == external_box["zmin"]
        assert network.domain.bounding_box["zmax"] == external_box["zmax"]
