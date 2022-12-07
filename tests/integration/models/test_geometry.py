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


class SingleFracture2d(pp.ModelGeometry):
    """Single fracture in 2d with unstructured simplex mesh."""

    num_fracs: int = 1
    """Used to compare to the size of the fracture network assigned internally."""

    def mesh_arguments(self) -> dict:
        return {"mesh_size_frac": 0.5, "mesh_size_min": 0.5, "mesh_size_bound": 0.5}

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
        fracs = [
            pp.PlaneFracture(np.array([pts0, pts1, pts2])),
            pp.PlaneFracture(np.array([pts2, pts0, pts1])),
            pp.PlaneFracture(np.array([pts1, pts2, pts0])),
        ]
        domain = pp.grids.standard_grids.utils.unit_domain(3)
        self.fracture_network = pp.FractureNetwork3d(fracs, domain)


class BaseWithUnits(pp.ModelGeometry):
    """ModelGeometry.set_md_geometry requires a units attribute."""

    units: pp.Units = pp.Units()


geometry_list = [BaseWithUnits, SingleFracture2d, ThreeFractures3d]


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
        all_bf, east, west, north, south, top, bottom = geometry.domain_boundary_sides(
            sd
        )
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
def test_subdomain_interface_methods(geometry_class: pp.ModelGeometry) -> None:
    """Test interfaces_to_subdomains and subdomains_to_interfaces.

    Parameters:
        geometry_class:

    """
    geometry = geometry_class()
    geometry.set_geometry()
    all_subdomains = geometry.mdg.subdomains()
    all_interfaces = geometry.mdg.interfaces()

    returned_subdomains = geometry.interfaces_to_subdomains(all_interfaces)
    returned_interfaces = geometry.subdomains_to_interfaces(all_subdomains)
    if all_interfaces == []:
        assert returned_subdomains == []
        assert returned_interfaces == []
    else:
        assert all_subdomains == returned_subdomains
        assert all_interfaces == returned_interfaces

    # Empty list passed should return empty list for both methods.
    no_subdomains = geometry.interfaces_to_subdomains([])
    no_interfaces = geometry.subdomains_to_interfaces([])
    assert no_subdomains == []
    assert no_interfaces == []
    if getattr(geometry, "num_fracs", 0) > 1:
        # Matrix and two fractures. TODO: Use three_sds?
        three_sds = all_subdomains[:2]

        two_fractures = all_subdomains[1:3]
        # Only those interfaces involving one of the two fractures are expected.
        interfaces = []
        for sd in two_fractures:
            interfaces += geometry.mdg.subdomain_to_interfaces(sd)
        sorted_interfaces = geometry.mdg.sort_interfaces(interfaces)
        assert sorted_interfaces == geometry.subdomains_to_interfaces(two_fractures)


@pytest.mark.parametrize("geometry_class", geometry_list)
def test_outwards_normals(geometry_class: pp.ModelGeometry) -> None:
    """Test :meth:`pp.ModelGeometry.outwards_internal_boundary_normals`.

    Parameters:
        geometry_class (pp.ModelGeometry): Class to test.

    """
    geometry: pp.ModelGeometry = geometry_class()
    geometry.set_geometry()
    dim = geometry.nd

    eq_sys = pp.EquationSystem(geometry.mdg)
    interfaces = geometry.mdg.interfaces()
    normal_op = geometry.outwards_internal_boundary_normals(interfaces, unitary=True)
    normals = normal_op.evaluate(eq_sys)
    if len(interfaces) == 0:
        # We have checked that the method can handle empty lists (parsable operator).
        # Check type and entry, then return.
        assert isinstance(normals, sps.spmatrix)
        assert np.allclose(normals.A, 0)
        return
    diag = normals.diagonal()
    # Check that all off-diagonal entries are zero
    assert np.allclose(np.diag(diag) - normals, 0)
    normals_reshaped = np.reshape(diag, (dim, -1), order="F")
    # Check that the normals are unit vectors
    assert np.allclose(np.linalg.norm(normals_reshaped, axis=0), 1)

    # Check that the normals are outward. This is done by checking that the dot product
    # of the normal and the vector from the center of the interface to the center of the
    # neighboring subdomain cell is positive.
    subdomains = geometry.interfaces_to_subdomains(interfaces)
    mortar_projection = pp.ad.MortarProjections(
        geometry.mdg, subdomains, interfaces, dim=dim
    )
    interface_cell_centers = geometry.wrap_grid_attribute(
        interfaces, "cell_centers", dim
    )
    subdomain_cell_centers = geometry.wrap_grid_attribute(
        subdomains, "cell_centers", dim
    )
    trace = pp.ad.Trace(subdomains)
    # Hack, see :meth:`pp.ad.geometry.ModelGeometry.basis`
    face_base = []
    for i in range(dim):
        e_i = np.zeros(dim).reshape(-1, 1)
        e_i[i] = 1
        # expand to cell-wise column vectors.
        num_cells = sum([g.num_faces for g in subdomains])
        mat = sps.kron(sps.eye(num_cells), e_i)
        face_base.append(pp.ad.Matrix(mat))
    cell_base = geometry.basis(subdomains, dim=dim)
    trace_dim = sum([f * trace.trace * e.T for e, f in zip(cell_base, face_base)])
    inv_trace_dim = sum(
        [e * trace.inv_trace * f.T for e, f in zip(cell_base, face_base)]
    )
    vec = (
        interface_cell_centers
        - mortar_projection.primary_to_mortar_avg
        * trace_dim
        * subdomain_cell_centers
        * inv_trace_dim
        * mortar_projection.mortar_to_primary_avg
    ).evaluate(eq_sys)
    # Check that the dot product is positive.
    # Collapse the
    prod = normals.diagonal() * vec.diagonal()
    dot_prod = np.reshape(prod, (dim, -1), order="F").sum(axis=0)
    assert np.all(dot_prod > 0)

    # Left multiply with dim-vector defined on the interface. This should give a vector
    # of length dim*num_intf_cells.
    size = dim * sum([intf.num_cells for intf in interfaces])
    dim_vec = pp.ad.Array(np.ones(size))
    product = (normal_op * dim_vec).evaluate(eq_sys)
    assert product.shape == (size,)
    inner_product = np.sum(product.reshape((dim, -1), order="F"), axis=0)
    assert np.allclose(np.abs(inner_product), 1)
    # The following operation is used in models, and is therefore tested here.
    # TODO: Extract method for inner product using a basis?
    basis = geometry.basis(interfaces, dim)
    nd_to_scalar_sum = sum([e.T for e in basis])
    inner_op = nd_to_scalar_sum * (normal_op * dim_vec)
    assert np.allclose(inner_op.evaluate(eq_sys), inner_product)
