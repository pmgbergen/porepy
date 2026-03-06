"""Test functionality related to elliptic_fracture module."""

import gmsh
import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.fracture_properties import (
    distance_from_points_to_fracture_plane,
)
from porepy.fracs import elliptic_fracture


@pytest.mark.parametrize(
    "elliptic_fracture_params",
    [
        (np.array([3.0, 4.0, 5.0]), 2.0, 1.0, np.pi / 6.0, np.pi / 4.0, np.pi / 8.0),
        (np.array([8.0, 7.0, 6.0]), 2.5, 0.5, np.pi / 6.0, np.pi / 4.0, np.pi / 8.0),
    ],
)
def test_fracture_geometry(elliptic_fracture_params):
    """Test that the generated elliptic fractures lie in the correct plane."""
    center, major_axis, minor_axis, major_axis_angle, strike_angle, dip_angle = (
        elliptic_fracture_params
    )
    fracture = elliptic_fracture.EllipticFracture(
        center, major_axis, minor_axis, major_axis_angle, strike_angle, dip_angle
    )
    domain = _standard_domain()
    mdg = _create_mdg([fracture], domain)
    frac_nodes = mdg.subdomains(dim=2)[0].nodes
    dis = distance_from_points_to_fracture_plane(
        frac_nodes.T, center, strike_angle, dip_angle
    )
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
            "mesh_size_boundary": 10,
            "mesh_size_fracture": 10,
            "refinement_threshold": 1e-4,
            "refinement_proximity_multiplier": 1,
            "refinement_size_multiplier": 1,
            "background_transition_multiplier": 1.01,
        }
    network = pp.create_fracture_network(fractures, domain=domain)
    if constraints is None:
        mdg = network.mesh(mesh_args)
    else:
        mdg = network.mesh(mesh_args, constraints=constraints)
    return mdg
