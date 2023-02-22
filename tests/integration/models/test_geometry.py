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

from . import setup_utils

geometry_list = [
    setup_utils.RectangularDomainThreeFractures,
    setup_utils.OrthogonalFractures3d,
]

num_fracs_list = [0, 1, 2, 3]


@pytest.mark.parametrize("geometry_class", geometry_list)
@pytest.mark.parametrize("num_fracs", num_fracs_list)
def test_set_fracture_network(geometry_class, num_fracs):
    """Test the method set_fracture_network."""
    geometry = geometry_class()
    geometry.params = {"num_fracs": num_fracs}
    geometry.units = pp.Units()
    geometry.num_fracs = num_fracs
    geometry.set_fracture_network()
    assert getattr(geometry, "num_fracs", 0) == geometry.fracture_network.num_frac()


@pytest.mark.parametrize("geometry_class", geometry_list)
def test_set_geometry(geometry_class):
    """Test the method set_geometry."""
    geometry = geometry_class()
    # Testing with a single fracture should be sufficient here.
    geometry.params = {"num_fracs": 1}
    geometry.units = pp.Units()
    geometry.set_geometry()
    for attr in ["mdg", "domain", "nd", "fracture_network"]:
        assert hasattr(geometry, attr)
    # For now, the default is not to assign a well network. Assert to remind ourselves
    # to add testing if default is changed.
    assert not hasattr(geometry, "well_network")


@pytest.mark.parametrize("geometry_class", geometry_list)
@pytest.mark.parametrize("num_fracs", num_fracs_list)
def test_boundary_sides(geometry_class, num_fracs):
    geometry = geometry_class()
    geometry.params = {"num_fracs": num_fracs}
    geometry.units = pp.Units()
    geometry.set_geometry()

    # Fetch the bounding box for the domain
    box_min, box_max = pp.domain.mdg_minmax_coordinates(geometry.mdg)

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

        # Check that the coordinates of the
        for side, dim in zip([east, north, top], [0, 1, 2]):
            assert np.all(np.isclose(sd.face_centers[dim, side], box_max[dim]))
        for side, dim in zip([west, south, bottom], [0, 1, 2]):
            assert np.all(np.isclose(sd.face_centers[dim, side], box_min[dim]))


@pytest.mark.parametrize("geometry_class", geometry_list)
# Only test up to two fractures here, that should suffice.
@pytest.mark.parametrize("num_fracs", [0, 1, 2])
def test_wrap_grid_attributes(
    geometry_class: type[pp.ModelGeometry], num_fracs
) -> None:
    """Test that the grid attributes are wrapped correctly.

    The test is based on sending in a list of grids (both subdomains and interfaces)
    wrap a number of attributes, and check that the attributes are wrapped correctly.

    """
    geometry = geometry_class()
    geometry.params = {"num_fracs": num_fracs}
    geometry.units = pp.Units()
    geometry.set_geometry()
    nd: int = geometry.nd

    # Various combinations of single and many subdomains
    all_subdomains = geometry.mdg.subdomains()
    top_subdomain = geometry.mdg.subdomains(dim=geometry.nd)
    some_subdomains = top_subdomain + geometry.mdg.subdomains(dim=geometry.nd - 1)
    # An empty list
    empty_subdomains: list[pp.Grid] = []

    # Various combinations of single and many interfaces
    all_interfaces = geometry.mdg.interfaces()
    top_interfaces = geometry.mdg.interfaces(dim=geometry.nd - 1)
    some_interfaces = top_interfaces + geometry.mdg.interfaces(dim=geometry.nd - 2)

    # Gather all lists of subdomains and all lists of interfaces
    test_subdomains = [all_subdomains, top_subdomain, some_subdomains, empty_subdomains]
    test_interfaces = [all_interfaces, top_interfaces, some_interfaces]

    # Equation system, needed for evaluation.
    eq_system = pp.ad.EquationSystem(geometry.mdg)

    # Test that an error is raised if the grid does not have such an attribute
    with pytest.raises(ValueError):
        geometry.wrap_grid_attribute(
            top_subdomain, "no_such_attribute", dim=1, inverse=False
        )
    # Test that the an error is raised if we try to wrap a field which is not an
    # ndarray.
    with pytest.raises(ValueError):
        # This will return a string
        geometry.wrap_grid_attribute(top_subdomain, "name", dim=1, inverse=False)

    # One loop for both subdomains and interfaces.
    for grids in test_subdomains + test_interfaces:

        # Which attributes to test depends on whether the grids are subdomains or
        # interfaces.
        if len(grids) == 0 or isinstance(grids[0], pp.MortarGrid):
            # Also include the empty list here, one attribute should be sufficient to
            # test that a zero matrix is returned.
            attr_list = ["cell_centers"]
            dim_list = [nd]
        else:
            # All relevant attributes for subdomain grids
            attr_list = [
                "cell_centers",
                "face_centers",
                "face_normals",
                "cell_volumes",
                "face_areas",
            ]
            #  List of dimensions, corresponding to the order in attr_list
            dim_list = [nd, nd, nd, 1, 1]

        # Loop over attributes and corresponding dimensions.
        for attr, dim in zip(attr_list, dim_list):
            # Get hold of the wrapped attribute and the wrapping with inverse=True
            wrapped_value = geometry.wrap_grid_attribute(
                grids, attr, dim=dim, inverse=False
            ).evaluate(eq_system)
            wrapped_value_inverse = geometry.wrap_grid_attribute(
                grids, attr, dim=dim, inverse=True
            ).evaluate(eq_system)

            # Check that the wrapped attribute is a matrix
            assert isinstance(wrapped_value, sps.spmatrix)
            assert isinstance(wrapped_value_inverse, sps.spmatrix)

            # Check that the matrix have the expected size, which depends on the type
            # of attribute wrapped (cell or face) and the dimension of the field.
            size_key = "num_cells" if "cell" in attr else "num_faces"
            tot_size = sum([getattr(sd, size_key) for sd in grids])

            assert wrapped_value.shape == (tot_size * dim, tot_size * dim)
            assert wrapped_value_inverse.shape == (tot_size * dim, tot_size * dim)

            # Get hold of the actual attribute values; we know these reside on the
            # main diagonal.
            values = wrapped_value.diagonal()
            values_inverse = wrapped_value_inverse.diagonal()

            # Counter for the current position in the wrapped attribute
            ind_cc = 0

            # Loop over the grids (be they subdomains or interfaces)
            for grid in grids:
                # Get hold of the actual attribute values straight from the grid
                size = getattr(grid, size_key)
                # Note the use of 2d here, or else the below accessing of [:dim] would
                # not work.
                actual_value = np.atleast_2d(getattr(grid, attr))
                # Compare values with the wrapped attribute, both usual and inverse.
                assert np.allclose(
                    values[ind_cc : ind_cc + size * dim],
                    actual_value[:dim].ravel("F"),
                )
                assert np.allclose(
                    values_inverse[ind_cc : ind_cc + size * dim],
                    1 / actual_value[:dim].ravel("F"),
                )
                # Move to the new position in the wrapped attribute
                ind_cc += size * dim


@pytest.mark.parametrize("geometry_class", geometry_list)
def test_subdomain_interface_methods(geometry_class: type[pp.ModelGeometry]) -> None:
    """Test interfaces_to_subdomains and subdomains_to_interfaces.

    Parameters:
        geometry_class:

    """
    geometry = geometry_class()
    # Use two fractures, that should be enough to test the methods.
    geometry.params = {"num_fracs": 2}
    geometry.units = pp.Units()
    geometry.set_geometry()
    all_subdomains = geometry.mdg.subdomains()
    all_interfaces = geometry.mdg.interfaces()

    returned_subdomains = geometry.interfaces_to_subdomains(all_interfaces)
    returned_interfaces = geometry.subdomains_to_interfaces(all_subdomains, [1])
    if all_interfaces == []:
        assert returned_subdomains == []
        assert returned_interfaces == []
    else:
        assert all_subdomains == returned_subdomains
        assert all_interfaces == returned_interfaces

    # Empty list passed should return empty list for both methods.
    no_subdomains = geometry.interfaces_to_subdomains([])
    no_interfaces = geometry.subdomains_to_interfaces([], [1])
    assert no_subdomains == []
    assert no_interfaces == []
    if getattr(geometry, "num_fracs", 0) > 1:
        # Matrix and two fractures. TODO: Use three_sds?
        two_fractures = all_subdomains[1:3]
        # Only those interfaces involving one of the two fractures are expected.
        interfaces = []
        for sd in two_fractures:
            interfaces += geometry.mdg.subdomain_to_interfaces(sd, [1])
        sorted_interfaces = geometry.mdg.sort_interfaces(interfaces)
        assert sorted_interfaces == geometry.subdomains_to_interfaces(
            two_fractures, [1]
        )


@pytest.mark.parametrize("geometry_class", geometry_list)
@pytest.mark.parametrize("num_fracs", [0, 1, 2, 3])
def test_internal_boundary_normal_to_outwards(
    geometry_class: type[pp.ModelGeometry], num_fracs
) -> None:
    # Define the geometry
    geometry: pp.ModelGeometry = geometry_class()
    geometry.params = {"num_fracs": num_fracs}
    geometry.units = pp.Units()
    geometry.set_geometry()
    dim = geometry.nd

    # Make an equation system, which is needed for parsing of the Ad operator
    # representations of the geometry
    eq_sys = pp.EquationSystem(geometry.mdg)

    # The function to be tested only accepts the top level subdomain(s)
    # NOTE: This test does not cover the case of multiple subdomains on the top level,
    # as could happen if we implement a domain decomposition approach. We could make
    # a partitioning of the top dimensional grid and thereby test the functionality,
    # but this has not been prioritized. Passing in the same subdomain twice will not
    # work, since the function will uniquify the input.
    subdomains = [
        geometry.mdg.interface_to_subdomain_pair(intf)[0]
        for intf in geometry.mdg.interfaces()
    ]

    # Get hold of the matrix to be tested, parse it to numerical format.
    sign_switcher = geometry.internal_boundary_normal_to_outwards(subdomains, dim=dim)
    mat = sign_switcher.evaluate(eq_sys)

    # Check that the wrapped attribute is a matrix
    assert isinstance(mat, sps.spmatrix)
    # Check that the matrix have the expected size.
    expected_size = sum([sd.num_faces for sd in subdomains]) * dim
    assert mat.shape == (expected_size, expected_size)

    # All values are stored on the main diagonal, fetch these.
    mat_vals = mat.diagonal()

    # Offset, needed to deal with the case of several subdomains. It is not relevant
    # for now (see comment above), but we keep it.
    offset = 0

    # Loop over subdomains (at the time of writing, there will only be one) and check
    # that the values are as expected.
    for sd in subdomains:
        # We get the expected values from the cell-face relation of the subdomain:
        # By assumptions in the mesh construction, the normal vector of a boundary
        # face is pointing outwards for those faces that have a positive cell-face
        # item (note that on boundary faces, there is only one non-zero entry in the
        # cell-face for each row, ie., each face).
        cf = sd.cell_faces
        # Summing is a trick to get the sign of the cell-face relation for the boundary
        # faces (we don't care about internal faces).
        cf_sum = np.sum(cf, axis=1)
        # Only compare for fracture faces
        fracture_faces = np.where(sd.tags["fracture_faces"])[0]
        # The matrix constrained to this subdomain
        loc_vals = mat_vals[offset : offset + sd.num_faces * dim]
        loc_size = loc_vals.size
        # The matrix will have one row for each face for each dimension. Loop over the
        # dimensions; the sign should be the same for all dimensions.
        for i in range(dim):
            # Indices belonging to the current dimension
            dim_ind = np.arange(i, loc_size, dim)
            dim_vals = loc_vals[dim_ind]
            assert np.allclose(
                dim_vals[fracture_faces], cf_sum[fracture_faces].A.ravel()
            )
        # Update offset, needed to test for multiple subdomains.
        offset += sd.num_faces * dim


@pytest.mark.parametrize("geometry_class", geometry_list)
@pytest.mark.parametrize("num_fracs", [0, 1, 2, 3])
def test_outwards_normals(geometry_class: type[pp.ModelGeometry], num_fracs) -> None:
    """Test :meth:`pp.ModelGeometry.outwards_internal_boundary_normals`.

    Parameters:
        geometry_class: Class to test.

    """
    # Define the geometry
    geometry: pp.ModelGeometry = geometry_class()
    geometry.params = {"num_fracs": num_fracs}  # type: ignore
    geometry.units = pp.Units()
    geometry.set_geometry()
    dim = geometry.nd
    # Make an equation system, which is needed for parsing of the Ad operator
    # representations of the geometry
    eq_sys = pp.EquationSystem(geometry.mdg)

    # First check the method to compute
    interfaces = geometry.mdg.interfaces()
    normal_op = geometry.outwards_internal_boundary_normals(interfaces, unitary=True)
    normals = normal_op.evaluate(eq_sys)

    # The result should be a sparse matrix
    assert isinstance(normals, sps.spmatrix)

    if len(interfaces) == 0:
        # We have checked that the method can handle empty lists (parsable operator).
        # Check the entry and exit.
        assert np.allclose(normals.A, 0)
        return

    diag = normals.diagonal()
    # Check that all off-diagonal entries are zero
    assert np.allclose(np.diag(diag) - normals, 0)

    # Convert the normals into a nd x num_faces array
    normals_reshaped = np.reshape(diag, (dim, -1), order="F")

    # Check that the normals are unit vectors
    assert np.allclose(np.linalg.norm(normals_reshaped, axis=0), 1)

    # Also construct the normal vectors without normalization, and check that their
    # norms are equal to the volumes of the interface cells.
    normal_op_not_unitary = geometry.outwards_internal_boundary_normals(
        interfaces, unitary=False
    )
    normals_not_unitary = normal_op_not_unitary.evaluate(eq_sys)
    diag_not_unitary = normals_not_unitary.diagonal()
    normals_reshaped_not_unitary = np.reshape(diag_not_unitary, (dim, -1), order="F")

    volumes = np.hstack([intf.cell_volumes for intf in interfaces])
    assert np.allclose(np.linalg.norm(normals_reshaped_not_unitary, axis=0), volumes)

    # Check that the normals are outward. This is done by checking that the dot product
    # of the normal and the vector from the center of the interface to the center of the
    # neighboring subdomain cell is positive.
    offset = 0
    for intf in interfaces:
        sd = geometry.mdg.interface_to_subdomain_pair(intf)[0]

        loc_normals = normals_reshaped[:, offset : offset + intf.num_cells]

        fracture_faces = intf.mortar_to_primary_avg().tocsc().indices
        proj_normals = (intf.mortar_to_primary_avg() * loc_normals.T)[fracture_faces]

        cc = sd.cell_centers
        fc = sd.face_centers

        fracture_cells = sd.cell_faces[fracture_faces].tocsr().indices

        vec = cc[:, fracture_cells] - fc[:, fracture_faces]

        nrm1 = np.linalg.norm(proj_normals, axis=1)

        nrm2 = np.linalg.norm(proj_normals + 1e-3 * vec[:dim].T, axis=1)

        assert np.all(nrm1 > nrm2)
        offset += intf.num_cells

    # Left multiply with dim-vector defined on the interface. This should give a vector
    # of length dim*num_intf_cells.
    size = dim * sum([intf.num_cells for intf in interfaces])
    dim_vec = pp.ad.Array(np.ones(size))

    # Left multiply with the normal operator; in essense this extracts the normal vector
    # (in the geometric sense) as a vector (in the algebraic sense).
    product = (normal_op * dim_vec).evaluate(eq_sys)
    assert product.shape == (size,)

    # Each vector should have unit length, as is checked by by the lines below.
    inner_product = np.linalg.norm(product.reshape((dim, -1), order="F"), axis=0)
    assert np.allclose(np.abs(inner_product), 1)

    # Also sum the components of the normal vector. This is equivalent to taking the dot
    # product between the normal vector and a vector of ones.
    dot_product = np.sum(product.reshape((dim, -1), order="F"), axis=0)

    # The summing can also be done by constructing an nd-to-scalar mapping, using the
    # basis vectors. A similar operation (actually the transpose, scalar-to-nd) is used
    # in the model classes to expand scalars to vectors (e.g., pressure as a potential
    # to pressure as a force).
    basis = geometry.basis(interfaces, dim)
    nd_to_scalar_sum = sum([e.T for e in basis])
    inner_op = nd_to_scalar_sum * (normal_op * dim_vec)

    # The two operations should give the same result
    assert np.allclose(inner_op.evaluate(eq_sys), dot_product)


@pytest.mark.parametrize("geometry_class", geometry_list)
def test_basis_normal_tangential_components(
    geometry_class: type[pp.ModelGeometry],
) -> None:
    """Test that methods to compute basis vectors and extract normal and tangential
    components work as expected.

    Parameters:
        geometry_list: List of classes to test.

    """
    geometry = geometry_class()
    # Use two fractures, that should be sufficient to test the method
    geometry.params = {"num_fracs": 2}
    geometry.units = pp.Units()
    geometry.set_geometry()
    geometry.set_geometry()
    dim = geometry.nd

    # List of subdomains and interfaces. The latter are only needed for one test.
    subdomains = geometry.mdg.subdomains()
    interfaces = geometry.mdg.interfaces()

    # Count the number of cells
    num_subdomain_cells = sum([sd.num_cells for sd in subdomains])
    num_cells_total = num_subdomain_cells + sum([intf.num_cells for intf in interfaces])

    # Make an equation system, which is needed for parsing of the Ad operator
    # representations of the geometry
    eq_sys = pp.EquationSystem(geometry.mdg)

    # First test the method e_i (and thereby also the method basis, since the latter
    # is just a shallow wrapper around the former).
    # Loop over dimension of the basis vectors and of dimensions, construct the basis
    # vectors and check that they have the expected components.
    for basis_dim in range(dim + 1):
        for i in range(basis_dim):
            # Consider both subdomains and interfaces here, since the method allows it.
            e_i = geometry.e_i(subdomains + interfaces, i=i, dim=basis_dim).evaluate(
                eq_sys
            )
            # Expected values
            rows = np.arange(i, num_cells_total * basis_dim, basis_dim)
            cols = np.arange(num_cells_total)
            data = np.ones(num_cells_total)
            mat = sps.coo_matrix(
                (data, (rows, cols)),
                shape=(num_cells_total * basis_dim, num_cells_total),
            )
            assert np.allclose((mat - e_i).data, 0)

            if basis_dim == dim:
                # the dimension of the basis vector space is not specified, the value
                # should be the same as for basis_dim = dim
                e_None = geometry.e_i(subdomains + interfaces, i=i, dim=dim).evaluate(
                    eq_sys
                )
                assert np.allclose((e_None - e_i).data, 0)

    # Next, test the methods to extract normal and tangential components.
    # The normal component is straightforward, the tangential component requires a bit
    # of work to deal with the difference between 2d and 3d.
    normal_component = geometry.normal_component(subdomains).evaluate(eq_sys)

    # The normal component should, for each row, have 1 in the column corresponding to
    # the normal component, so [(0, dim-1), (1, 2*dim-1), ...)] should be non-zero.
    # Note that the number of rows is smaller than the number of columns, since the
    # matrix will multiply a full vector and extract the normal component.
    rows_normal_component = np.arange(num_subdomain_cells)
    cols_normal_component = np.arange(dim - 1, dim * num_subdomain_cells, dim)
    data_normal_component = np.ones(num_subdomain_cells)

    known_normal_component = sps.coo_matrix(
        (data_normal_component, (rows_normal_component, cols_normal_component)),
        shape=(num_subdomain_cells, dim * num_subdomain_cells),
    )

    assert np.allclose((known_normal_component - normal_component).data, 0)

    # For the tangential component, the expected value depends on dimension.
    tangential_component = geometry.tangential_component(subdomains).evaluate(eq_sys)
    if dim == 2:
        # Here we need [(0, 0), (1, 2), (2, 4), ...] to be non-zero
        rows_tangential_component = np.arange(num_subdomain_cells)
        cols_tangential_component = np.arange(0, dim * num_subdomain_cells, dim)
        data_tangential_component = np.ones(num_subdomain_cells)
    elif dim == 3:
        # Here we need [(0, 0), (1, 1), (2, 3), (3, 4), (4, 6), ..] to be non-zero
        rows_tangential_component = np.arange(num_subdomain_cells * (dim - 1))
        cols_tangential_component_ext = np.arange(0, dim * num_subdomain_cells)
        cols_tangential_component = np.setdiff1d(
            cols_tangential_component_ext,
            np.arange(dim - 1, dim * num_subdomain_cells, dim),
        )
        data_tangential_component = np.ones(num_subdomain_cells * (dim - 1))

    known_tangential_component = sps.coo_matrix(
        (
            data_tangential_component,
            (rows_tangential_component, cols_tangential_component),
        ),
        shape=((dim - 1) * num_subdomain_cells, dim * num_subdomain_cells),
    )

    assert np.allclose((known_tangential_component - tangential_component).data, 0)
