"""Test functionality related to ellipse_fracture module."""

import gmsh
import numpy as np
import pytest

import porepy as pp
from porepy.fracs import ellipse_fracture


@pytest.mark.parametrize(
    "ellipse_fracture_params",
    [
        (np.array([3.0, 4.0, 5.0]), 2.0, 1.0, np.pi / 6.0, np.pi / 4.0, np.pi / 8.0),
        (np.array([8.0, 7.0, 6.0]), 2.5, 0.5, np.pi / 6.0, np.pi / 4.0, np.pi / 8.0),
    ],
)
def test_fracture_geometry(ellipse_fracture_params):
    """Test that the generated elliptic fractures lie in the correct plane."""
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
    bbox = {"xmin": -15, "xmax": 15, "ymin": -15, "ymax": 15, "zmin": -15, "zmax": 15}
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
        mesh_args = {
            "mesh_size_bound": 10,
            "mesh_size_frac": 10,
            "refinement_threshold": 1e-4,
        }
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
