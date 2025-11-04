"""Test functionality related to ellipse_fracture module."""

import numpy as np
import pytest
import gmsh
import porepy as pp
from porepy.fracs import ellipse_fracture
from collections import namedtuple
from tests.fracs.test_fracture_network_3d import check_mdg


@pytest.mark.parametrize(
    "ellipse_fracture_params",
    [
        (np.array([3.0, 4.0, 5.0]), 2.0, 1.0, np.pi / 6.0, np.pi / 4.0, np.pi / 8.0),
        (np.array([8.0, 7.0, 6.0]), 2.5, 0.5, np.pi / 6.0, np.pi / 4.0, np.pi / 8.0),
    ],
)
def test_ellipse_fracture_center(ellipse_fracture_params):
    # parse the parameters
    center, major_axis, minor_axis, major_axis_angle, strike_angle, dip_angle = (
        ellipse_fracture_params
    )

    center_known = ellipse_fracture_params[0]
    fracture = ellipse_fracture.EllipticFracture(
        center, major_axis, minor_axis, major_axis_angle, strike_angle, dip_angle
    )
    assert np.allclose(center_known, fracture.center)


def test_ellipse_fracture_tags():
    gmsh.initialize()
    frac1 = ellipse_fracture.EllipticFracture(
        np.array([3.0, 4.0, 5.0]), 2.0, 1.0, np.pi / 6.0, np.pi / 4.0, np.pi / 8.0
    )
    frac2 = ellipse_fracture.EllipticFracture(
        np.array([8.0, 7.0, 6.0]), 2.5, 0.5, np.pi / 6.0, np.pi / 4.0, np.pi / 8.0
    )
    tag1 = frac1.fracture_to_gmsh_3D()
    tag2 = frac2.fracture_to_gmsh_3D()
    gmsh.finalize()
    assert tag1 != tag2


@pytest.mark.parametrize(
    "ellipse_fracture_params",
    [
        (np.array([3.0, 4.0, 5.0]), 2.0, 1.0, np.pi / 6.0, np.pi / 4.0, np.pi / 8.0),
        (np.array([8.0, 7.0, 6.0]), 2.5, 0.5, np.pi / 6.0, np.pi / 4.0, np.pi / 8.0),
    ],
)
def test_fracture_geometry(ellipse_fracture_params):
    center, major_axis, minor_axis, major_axis_angle, strike_angle, dip_angle = (
        ellipse_fracture_params
    )
    fracture = ellipse_fracture.EllipticFracture(
        center, major_axis, minor_axis, major_axis_angle, strike_angle, dip_angle
    )
    domain = _standard_domain()
    mdg = _create_mdg([fracture], domain)
    frac_nodes = mdg.subdomains(dim=2)[0].nodes
    dis = plane_check(frac_nodes.T, center, strike_angle, dip_angle)
    assert np.abs(dis).max() <= 1e-6


def _standard_domain(modify: bool = False) -> dict | pp.Domain:
    """Create a standard domain for testing purposes."""
    bbox = {"xmin": -10, "xmax": 10, "ymin": -10, "ymax": 10, "zmin": -10, "zmax": 10}
    if modify:
        return bbox
    else:
        domain = pp.Domain(bbox)
        return domain


def _create_mdg(
    fractures, domain=None, mesh_args: dict | None = None, constraints=None
) -> pp.MixedDimensionalGrid:
    """Create a mixed-dimensional grid from a list of fractures."""
    if mesh_args is None:
        mesh_args = {"mesh_size_bound": 1, "mesh_size_frac": 1, "mesh_size_min": 0.1}
    network = pp.create_fracture_network(fractures, domain=domain)
    if constraints is None:
        mdg = network.mesh(mesh_args)
    else:
        mdg = network.mesh(mesh_args, constraints=constraints)
    return mdg


def plane_check(points_xyz, center, strike_angle, dip_angle):
    """
    Check whether the given points are located in the plane defined by strike and dip.
    """
    P = np.asarray(points_xyz)
    c = np.asarray(center).ravel()
    phi = float(strike_angle)
    theta = float(dip_angle)

    n = np.array(
        [np.sin(theta) * np.sin(phi), -np.sin(theta) * np.cos(phi), np.cos(theta)],
    )
    n /= np.linalg.norm(n)

    dis_error = (P - c) @ n
    return dis_error


def test_create_mdg_for_ellipse():
    gmsh.initialize()
    bounding_box = {
        "xmin": 0,
        "xmax": 10,
        "ymin": 0,
        "ymax": 10,
        "zmin": 0,
        "zmax": 10,
    }
    domain = pp.Domain(bounding_box=bounding_box)
    frac1 = pp.EllipticFracture(
        np.array([3.0, 4.0, 5.0]), 2.0, 1.0, np.pi / 6.0, np.pi / 4.0, np.pi / 8.0
    )
    frac2 = pp.EllipticFracture(
        np.array([8.0, 7.0, 6.0]), 2.5, 0.5, np.pi / 6.0, np.pi / 4.0, np.pi / 8.0
    )
    frac3 = pp.PlaneFracture(np.array([[0, 4, 5, 0], [0, 0, 2, 2], [0, 0, 2, 2]]))
    fractures = [frac1, frac2, frac3]
    network = pp.create_fracture_network(fractures, domain)
    gmsh.model.occ.synchronize()
    mesh_args = {
        "cell_size_boundary": 1.0,
        "cell_size_fracture": 0.5,
        "cell_size_min": 0.2,
    }
    mdg = pp.create_mdg("simplex", mesh_args, network)
    assert mdg is not None
    assert mdg.num_subdomains() >= 4


IntersectionInfo = namedtuple("IntersectionInfo", ["parent_0", "parent_1", "coord"])


def test_two_intersecting_fractures():
    """Two fractures intersecting along a line."""

    f_1 = ellipse_fracture.EllipticFracture(
        np.array([3.0, 4.0, 5.0]),
        2.0,
        1.0,
        0.0,
        0.0,
        np.pi / 6.0,
    )
    f_2 = ellipse_fracture.EllipticFracture(
        np.array([3.0, 4.0, 5.0]),
        2.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    domain = _standard_domain()
    mdg = _create_mdg([f_1, f_2], domain)

    isect_coord = np.array([[2, 4], [4, 4], [5, 5]])

    isect = IntersectionInfo(0, 1, isect_coord)

    check_mdg(
        mdg,
        domain,
        fractures=[f_1, f_2],
        isect_line=[isect],
        expected_num_1d_grids=1,
    )
