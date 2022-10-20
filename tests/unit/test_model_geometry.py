"""Tests of geometry part of a simulation model.

Testing covers:
    Setting of mixed-dimensional grid
    Subdomain and interface list methods:
        subdomains_to_interfaces
        interfaces_to_subdomains
    Utility methods:
        domain_boundary_sides
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp

class SingleFracture2d(pp.Geometry):
    """Single fracture in 2d with unstructured simplex mesh."""

    num_fracs: int = 1
    """Used to compare to the size of the fracture network assigned internally."""
    def mesh_arguments(self) -> dict:
        return {"mesh_size_frac": .5, "mesh_size_min": .5, "mesh_size_bound": .5}

    def set_fracture_network(self):
        pts = np.array([[0, 0.5], [0.5, 0.5]])
        edges = np.array([[0], [1]])
        domain = pp.grids.standard_grids.utils.unit_domain(2)
        self.fracture_network = pp.FractureNetwork2d(pts, edges, domain)

class ThreeFractures3d(SingleFracture2d):
    """Three fractures in 3d with unstructured simplex mesh."""

    ambient_dimension: int = 3
    """Used to compare to the nd attribute assigned internally."""
    num_fracs: int = 3
    """Used to compare to the size of the fracture network assigned internally."""

    def set_fracture_network(self):
        coords = [0, 1]
        pts0 = [coords[0], coords[0], coords[1], coords[1]]
        pts1 = [coords[0], coords[1], coords[1], coords[0]]
        pts2 = [0.5, 0.5, 0.5, 0.5]
        fracs = [pp.PlaneFracture(np.array([pts0, pts1, pts2])),
                 pp.PlaneFracture(np.array([pts2, pts0, pts1])),
                 pp.PlaneFracture(np.array([pts1, pts2, pts0]))]
        domain = pp.grids.standard_grids.utils.unit_domain(3)
        self.fracture_network = pp.FractureNetwork3d(fracs, domain)

geometry_list = [pp.Geometry, SingleFracture2d, ThreeFractures3d]

@pytest.mark.parametrize("geometry_class", geometry_list)
def test_set_fracture_network(geometry_class):
    geometry = geometry_class()
    geometry.set_fracture_network()
    assert getattr(geometry, "num_fracs", 0) == geometry.fracture_network.num_frac()


@pytest.mark.parametrize("geometry_class", geometry_list)
def test_set_geometry(geometry_class):
    geometry = geometry_class()
    geometry.set_geometry()
    for attr in ["mdg", "box", "nd", "fracture_network"]:
        assert hasattr(geometry, attr)
    # For now, the default is not to assign a well network. Assert to remind ourselves to
    # add testing if default is changed.
    assert not hasattr(geometry, "well_network")

    # Checks on attribute values. Default values correspond to the un-modified
    assert geometry.nd == getattr(geometry, "ambient_dimension", 2)


@pytest.mark.parametrize("geometry_class", geometry_list)
def test_boundary_sides(geometry_class):
    geometry = geometry_class()
    geometry.set_geometry()
    for sd in geometry.mdg.subdomains():
        all_bf, east, west, north, south, top, bottom = geometry.domain_boundary_sides(sd)
        all_bool = np.zeros(sd.num_faces, dtype=bool)
        all_bool[all_bf] = 1

        # Check that only valid boundaries are picked
        domain_or_internal_bf = np.where(np.sum(np.abs(sd.cell_faces), axis=1) == 1)
        assert np.all(np.in1d(all_bf, domain_or_internal_bf))
        frac_faces = sd.tags["fracture_faces"].nonzero()[0]
        assert np.all(np.logical_not(np.in1d(all_bf, frac_faces)))
        assert np.all(all_bool == (east + west + north + south + top + bottom))

        # Check coordinates
        for side, dim in zip([east, north, top], [0, 1, 2]):
            assert np.all(np.isclose(sd.face_centers[dim, side], 1))
        for side, dim in zip([west, south, bottom], [0, 1, 2]):
            assert np.all(np.isclose(sd.face_centers[dim, side], 0))


@pytest.mark.parametrize("geometry_class", geometry_list)
def test_subdomain_interface_methods(geometry_class: pp.Geometry) -> None:
    """Test interfaces_to_subdomains and subdomains_to_interfaces.

    Args:
        geometry_class: 

    FIXME: Some of the tests will fail until sorting of subdomains and interfaces has been
        addressed.
    """
    geometry = geometry_class()
    geometry.set_geometry()
    all_subdomains = geometry.mdg.subdomains()
    all_interfaces = geometry.mdg.interfaces()

    returned_subdomains = geometry.interfaces_to_subdomains(all_interfaces)
    returned_interfaces = geometry.subdomains_to_interfaces(all_subdomains)
    assert all_subdomains == returned_subdomains
    assert all_interfaces == returned_interfaces

    # Empty list passed should return empty list for both methods.
    no_subdomains = geometry.interfaces_to_subdomains([])
    no_interfaces = geometry.subdomains_to_interfaces([])
    assert no_subdomains == []
    assert no_interfaces == []
    if geometry.num_fracs > 1:
        # Matrix and two fractures
        three_sds = all_subdomains[:2]
        # This should result in all subdomains, since the method returns all interfaces
        # neighboring any of the domains
        assert all_interfaces == geometry.subdomains_to_interfaces(three_sds)

        two_fractures = all_subdomains[1:3]
        sd_m = geometry.mdg.subdomains(dim=geometry.nd)[0]
        # Only those interfaces involving one of the two fractures are expected.
        interfaces = [geometry.mdg.subdomain_pair_to_interface((sd_m, two_fractures[0])),
                      geometry.mdg.subdomain_pair_to_interface((sd_m, two_fractures[1]))]
        assert interfaces == geometry.subdomains_to_interfaces(two_fractures)

