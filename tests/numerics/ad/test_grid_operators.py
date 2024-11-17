"""Test collection for Ad representations of grid-related operators.

Checks performed include the following:
    test_subdomain_projections: Operators for restriction and prolongation are checked
        for both faces and cells;
    test_mortar_projections_empty_list: Projections between empty lists of subdomains
        and interfaces;
    test_mortar_projections: Projections between mortar grids and subdomain grids;
    test_boundary_grid_projection:  Tests are conducted on the boundary projection
        operator and its inverse;
    test_trace and test_divergence: Operators for discrete traces and divergences
    test_ad_discretization_class: test for AD discretizations.

"""
import numpy as np
import pytest
import scipy.sparse as sps
import porepy as pp
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data


@pytest.fixture
def mdg():
    """Provide a mixed-dimensional grid for the tests."""
    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    md_grid = pp.meshing.cart_grid(fracs, np.array([2, 2]))
    return md_grid


@pytest.mark.integtest
@pytest.mark.parametrize("scalar", [True, False])
def test_subdomain_projections(mdg, scalar):
    """Test of subdomain projections. Both face and cell restriction and prolongation.

    Test three specific cases:
        1. Projections generated by passing a md-grid and a list of grids are identical
        2. All projections for all grids (individually) in a simple md-grid.
        3. Combined projections for list of grids.

    """
    proj_dim = 1 if scalar else mdg.dim_max()
    n_cells, n_faces, _ = geometry_information(mdg, proj_dim)

    subdomains = mdg.subdomains()
    proj = pp.ad.SubdomainProjections(subdomains=subdomains, dim=proj_dim)

    cell_start = np.cumsum(
        np.hstack((0, np.array([sd.num_cells for sd in subdomains])))
    )
    face_start = np.cumsum(
        np.hstack((0, np.array([sd.num_faces for sd in subdomains])))
    )

    # Helper method to get indices for sparse matrices
    def _mat_inds(nc, nf, grid_ind, dim, cell_start, face_start):
        cell_inds = np.arange(cell_start[grid_ind], cell_start[grid_ind + 1])
        face_inds = np.arange(face_start[grid_ind], face_start[grid_ind + 1])

        data_cell = np.ones(nc * dim)
        row_cell = np.arange(nc * dim)
        data_face = np.ones(nf * dim)
        row_face = np.arange(nf * dim)
        col_cell = pp.fvutils.expand_indices_nd(cell_inds, dim)
        col_face = pp.fvutils.expand_indices_nd(face_inds, dim)
        return row_cell, col_cell, data_cell, row_face, col_face, data_face

    # Test projections to and from an empty list of subdomains
    assert proj.cell_restriction([]).shape == (0, n_cells)
    assert proj.cell_prolongation([]).shape == (n_cells, 0)
    assert proj.face_restriction([]).shape == (0, n_faces)
    assert proj.face_prolongation([]).shape == (n_faces, 0)

    # Test projection of one fracture at a time for the full set of grids
    for sd in subdomains:
        ind = _list_ind_of_grid(subdomains, sd)

        nc, nf = sd.num_cells, sd.num_faces

        num_rows_cell = nc * proj_dim
        num_rows_face = nf * proj_dim

        row_cell, col_cell, data_cell, row_face, col_face, data_face = _mat_inds(
            nc, nf, ind, proj_dim, cell_start, face_start
        )

        known_cell_proj = sps.coo_matrix(
            (data_cell, (row_cell, col_cell)), shape=(num_rows_cell, n_cells)
        ).tocsr()
        known_face_proj = sps.coo_matrix(
            (data_face, (row_face, col_face)), shape=(num_rows_face, n_faces)
        ).tocsr()

        assert _compare_matrices(proj.cell_restriction([sd]), known_cell_proj)
        assert _compare_matrices(proj.cell_prolongation([sd]), known_cell_proj.T)
        assert _compare_matrices(proj.face_restriction([sd]), known_face_proj)
        assert _compare_matrices(proj.face_prolongation([sd]), known_face_proj.T)

    # Project between the full grid and both 1d grids (to combine two grids)
    g1, g2 = mdg.subdomains(dim=1)
    rc1, cc1, dc1, rf1, cf1, df1 = _mat_inds(
        g1.num_cells,
        g1.num_faces,
        _list_ind_of_grid(subdomains, g1),
        proj_dim,
        cell_start,
        face_start,
    )
    rc2, cc2, dc2, rf2, cf2, df2 = _mat_inds(
        g2.num_cells,
        g2.num_faces,
        _list_ind_of_grid(subdomains, g2),
        proj_dim,
        cell_start,
        face_start,
    )

    # Adjust the indices of the second grid, we will stack the matrices.
    rc2 += rc1.size
    rf2 += rf1.size
    num_rows_cell = (g1.num_cells + g2.num_cells) * proj_dim
    num_rows_face = (g1.num_faces + g2.num_faces) * proj_dim

    known_cell_proj = sps.coo_matrix(
        (np.hstack((dc1, dc2)), (np.hstack((rc1, rc2)), np.hstack((cc1, cc2)))),
        shape=(num_rows_cell, n_cells),
    ).tocsr()
    known_face_proj = sps.coo_matrix(
        (np.hstack((df1, df2)), (np.hstack((rf1, rf2)), np.hstack((cf1, cf2)))),
        shape=(num_rows_face, n_faces),
    ).tocsr()

    assert _compare_matrices(proj.cell_restriction([g1, g2]), known_cell_proj)
    assert _compare_matrices(proj.cell_prolongation([g1, g2]), known_cell_proj.T)
    assert _compare_matrices(proj.face_restriction([g1, g2]), known_face_proj)
    assert _compare_matrices(proj.face_prolongation([g1, g2]), known_face_proj.T)


@pytest.mark.integtest
def test_mortar_projections_empty_list(mdg):
    """Test projections between empty lists of subdomains and interfaces.

    This test is located in a separate function from the main test of mortar
    projections, as the parametrization in the main test is not needed here.

    The test sets up projections to various empty lists of subdomains and interfaces.
    The resulting projections should have zero rows or columns (depending on which
    of the lists is empty), but should be non-zero along the other dimension.

    """
    n_cells, n_faces, n_mortar_cells = geometry_information(mdg, 1)
    # Projection operator with empty list of subdomains.
    proj_no_subdomains = pp.ad.MortarProjections(
        subdomains=[], interfaces=mdg.interfaces(), mdg=mdg, dim=1
    )
    # From mortar to subdomains.
    assert proj_no_subdomains.mortar_to_primary_int.shape == (0, n_mortar_cells)
    assert proj_no_subdomains.mortar_to_secondary_int.shape == (0, n_mortar_cells)
    # From subdomains to mortar.
    assert proj_no_subdomains.primary_to_mortar_int.shape == (n_mortar_cells, 0)
    assert proj_no_subdomains.secondary_to_mortar_int.shape == (n_mortar_cells, 0)

    # Projection operator with empty list of interfaces.
    proj_no_interfaces = pp.ad.MortarProjections(
        subdomains=mdg.subdomains(), interfaces=[], mdg=mdg, dim=1
    )
    # From mortar to subdomains.
    assert proj_no_interfaces.mortar_to_primary_int.shape == (n_faces, 0)
    assert proj_no_interfaces.mortar_to_secondary_int.shape == (n_cells, 0)
    # From subdomains to mortar.
    assert proj_no_interfaces.primary_to_mortar_int.shape == (0, n_faces)
    assert proj_no_interfaces.secondary_to_mortar_int.shape == (0, n_cells)

    # Empty list of subdomains and interfaces.
    proj_no_subdomains_interfaces = pp.ad.MortarProjections(
        subdomains=[], interfaces=[], mdg=mdg, dim=1
    )
    # From mortar to subdomains.
    assert proj_no_subdomains_interfaces.mortar_to_primary_int.shape == (0, 0)
    assert proj_no_subdomains_interfaces.mortar_to_secondary_int.shape == (0, 0)
    # From subdomains to mortar.
    assert proj_no_subdomains_interfaces.primary_to_mortar_int.shape == (0, 0)
    assert proj_no_subdomains_interfaces.secondary_to_mortar_int.shape == (0, 0)


@pytest.mark.integtest
@pytest.mark.parametrize("scalar", [True, False])
@pytest.mark.parametrize("non_matching", [True, False])
def test_mortar_projections(mdg, scalar, non_matching):
    """Test of mortar projections between mortar grids and standard subdomain grids.

    Parameters:
        mdg: Mixed-dimensional grid.
        scalar: Boolean indicating whether the field being projected is scalar or
            vector, which will be taken as 2d (since the grid is 2d).
        non_matching: If True, the 1d grids will be refined so that the mappings from
            1d mortars to their secondary subdomains will have non-unitary entries.

    """

    if non_matching:
        # If requested, we will refine the two 1d grids, such that the projection
        # matrices have non-unitary entries.
        for g in mdg.subdomains(dim=1):
            g_new = pp.refinement.refine_grid_1d(g, ratio=2)
            mdg.replace_subdomains_and_interfaces({g: g_new})

    # Define the dimension of the field being projected
    proj_dim = 1 if scalar else mdg.dim_max()

    # Collect geometrical and grid objects.
    n_cells, n_faces, n_mortar_cells = geometry_information(mdg, proj_dim)

    # Define helper functions that for each subdomain and interface compute the offset
    # (start index) in the projection matrix. We will only check projections from mortar
    # to the subdomains (the other way is a transpose, which will be correct if the
    # original is so). Thus for the subdomains, we need the offsets for the faces
    # (relevant for projection to primary) and cells (relevant for projection to
    # secondary). For the interfaces, we need the offset for the cells (used in both
    # directions).
    def _row_offset_primary(sds):
        return proj_dim * np.cumsum(
            np.hstack((0, np.array([sd.num_faces for sd in sds])))
        )

    def _row_offset_secondary(sds):
        return proj_dim * np.cumsum(
            np.hstack((0, np.array([sd.num_cells for sd in sds])))
        )

    def _col_offset(intfs):
        return proj_dim * np.cumsum(
            np.hstack((0, np.array([m.num_cells for m in intfs])))
        )

    # Get the indices of the projection matrices for the mapping to primary and
    # secondary (higher and lower dimensional neighbor) for a given interface. Get the
    # data for both integrating and averaging operators; both are needed to cover
    # non-matching grids.
    def _indices_primary(intf):
        row, col, data_int = sps.find(intf.mortar_to_primary_int(nd=proj_dim))
        _, _, data_avg = sps.find(intf.mortar_to_primary_avg(nd=proj_dim))
        return row, col, data_int, data_avg

    def _indices_secondary(intf: list[pp.MortarGrid]):
        row, col, data_int = sps.find(intf.mortar_to_secondary_int(nd=proj_dim))
        _, _, data_avg = sps.find(intf.mortar_to_secondary_avg(nd=proj_dim))
        return row, col, data_int, data_avg

    # Consider four cases:
    # 1) All subdomains and interfaces. This is the base case.
    # 2) The 2d and 1d subdomains, and the 1d interfaces. This tests that we can
    #    consider a subset of the grids.
    # 3) The 2d subdomain, and the 1d interfaces. This tests that we can have a non-zero
    #    projection in one direction (to the primary) and zero in the other.
    # 4) The 2d subdomain and the 0d interfaces. This tests that we can have projections
    #    with shape dictated by the size of the involved grids, but with only zero
    #    entries.
    subdomain_lists = [
        mdg.subdomains(),
        mdg.subdomains(dim=2) + mdg.subdomains(dim=1),
        mdg.subdomains(dim=2),
        mdg.subdomains(dim=2),
    ]
    interface_lists = [
        mdg.interfaces(),
        mdg.interfaces(dim=1),
        mdg.interfaces(dim=1),
        mdg.interfaces(dim=0),
    ]

    for subdomains, interfaces in zip(subdomain_lists, interface_lists):
        # Compute the offsets for the subdomains and interfaces.
        face_start = _row_offset_primary(subdomains)
        cell_start = _row_offset_secondary(subdomains)
        mortar_start = _col_offset(interfaces)

        # Initialize lists to store the indices and data of the projection matrices
        row_ind_primary, col_ind_primary = [], []
        row_ind_secondary, col_ind_secondary = [], []
        data_primary_int, data_primary_avg = [], []
        data_secondary_int, data_secondary_avg = [], []

        # Loop over the interfaces and subdomains to collect the indices and data of the
        # projection matrices, provided that the subdomain is involved in the
        # projection.
        for intf in interfaces:
            # Neighboring subdomains
            sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)
            # If the subdomain is in the list, collect the indices and data of the
            # projection matrices. Otherwise, we ignore this subdomain.
            if sd_primary in subdomains:
                r, c, data_int, data_avg = _indices_primary(intf)
                row_ind_primary.append(r + face_start[subdomains.index(sd_primary)])
                col_ind_primary.append(c + mortar_start[interfaces.index(intf)])
                data_primary_int.append(data_int)
                data_primary_avg.append(data_avg)
            # Same for the lower dimensional neighbor
            if sd_secondary in subdomains:
                r, c, data_int, data_avg = _indices_secondary(intf)
                row_ind_secondary.append(r + cell_start[subdomains.index(sd_secondary)])
                col_ind_secondary.append(c + mortar_start[interfaces.index(intf)])
                data_secondary_int.append(data_int)
                data_secondary_avg.append(data_avg)

        # The shape of the projection matrices is determined by the number of faces and
        # cells in the subdomains and the number of cells in the interfaces.
        shape_primary = (
            proj_dim * sum([sd.num_faces for sd in subdomains]),
            proj_dim * sum([m.num_cells for m in interfaces]),
        )
        shape_secondary = (
            proj_dim * sum([sd.num_cells for sd in subdomains]),
            proj_dim * sum([m.num_cells for m in interfaces]),
        )

        # Construct the known projection matrices from the collected indices and data.
        # If no data is collected, the projection matrix is zero, with the given shape.
        if len(row_ind_primary) == 0:
            # If no data is collected (none of the given subdomains were neighbors of
            # the interfaces), the projection matrix is zero, with the given shape.
            proj_known_primary_int = sps.csr_matrix(shape_primary)
            proj_known_primary_avg = sps.csr_matrix(shape_primary)
        else:
            # Construct the projection matrices from the collected indices and data.
            # Separate projections for integration and averaging.
            proj_known_primary_int = sps.coo_matrix(
                (
                    np.hstack(data_primary_int),
                    (np.hstack(row_ind_primary), np.hstack(col_ind_primary)),
                ),
                shape=shape_primary,
            ).tocsr()
            proj_known_primary_avg = sps.coo_matrix(
                (
                    np.hstack(data_primary_avg),
                    (np.hstack(row_ind_primary), np.hstack(col_ind_primary)),
                ),
                shape=shape_primary,
            ).tocsr()

        if len(row_ind_secondary) == 0:
            proj_known_secondary_int = sps.csr_matrix(shape_secondary)
            proj_known_secondary_avg = sps.csr_matrix(shape_secondary)
        else:
            proj_known_secondary_int = sps.coo_matrix(
                (
                    np.hstack(data_secondary_int),
                    (np.hstack(row_ind_secondary), np.hstack(col_ind_secondary)),
                ),
                shape=shape_secondary,
            ).tocsr()
            proj_known_secondary_avg = sps.coo_matrix(
                (
                    np.hstack(data_secondary_avg),
                    (np.hstack(row_ind_secondary), np.hstack(col_ind_secondary)),
                ),
                shape=shape_secondary,
            ).tocsr()

        # Compute the object being tested.
        proj = pp.ad.MortarProjections(
            subdomains=subdomains, interfaces=interfaces, mdg=mdg, dim=proj_dim
        )

        # Compare the known and computed projection matrices.
        assert _compare_matrices(proj_known_primary_int, proj.mortar_to_primary_int)
        assert _compare_matrices(proj_known_primary_avg, proj.mortar_to_primary_avg)
        # The mappings from primary to mortar are found by transposing the mappings from
        # mortar to primary, and then switching averaging and integration (this is just
        # how it is).
        assert _compare_matrices(proj_known_primary_avg.T, proj.primary_to_mortar_int)
        assert _compare_matrices(proj_known_primary_int.T, proj.primary_to_mortar_avg)

        # Same for the mapping to the secondary subdomains.
        assert _compare_matrices(proj_known_secondary_int, proj.mortar_to_secondary_int)
        assert _compare_matrices(proj_known_secondary_avg, proj.mortar_to_secondary_avg)
        # See the mapping from primary to mortar above for comments.
        assert _compare_matrices(
            proj_known_secondary_avg.T, proj.secondary_to_mortar_int
        )
        assert _compare_matrices(
            proj_known_secondary_int.T, proj.secondary_to_mortar_avg
        )


@pytest.mark.integtest
@pytest.mark.parametrize("scalar", [True, False])
def test_boundary_grid_projection(mdg: pp.MixedDimensionalGrid, scalar: bool):
    """Three main functionalities being tested:
    1) That we can create a boundary projection operator with the correct size and items.
    2) Specifically that the top-dimensional grid and one of the fracture grids
       contribute to the boundary projection operator, while the third has a projection
       matrix with zero rows.
    3) Projection from a subdomain to a boundary is consistent with its reverse.

    """
    proj_dim = 1 if scalar else mdg.dim_max()
    _, num_faces, _ = geometry_information(mdg, proj_dim)
    num_cells = sum([bg.num_cells for bg in mdg.boundaries()]) * proj_dim

    g_0 = mdg.subdomains(dim=2)[0]
    g_1, g_2 = mdg.subdomains(dim=1)
    # Compute geometry for the mixed-dimensional grid. This is needed for
    # boundary projection operator.
    mdg.compute_geometry()
    projection = pp.ad.BoundaryProjection(mdg, mdg.subdomains(), proj_dim)
    # Obtaining sparse matrices from the AD Operators.
    subdomain_to_boundary = projection.subdomain_to_boundary.parse(mdg)
    boundary_to_subdomain = projection.boundary_to_subdomain.parse(mdg)
    # Check sizes.
    assert subdomain_to_boundary.shape == (num_cells, num_faces)
    assert boundary_to_subdomain.shape == (num_faces, num_cells)

    # Check that the projection matrix for the top-dimensional grid is non-zero.
    # The matrix has eight boundary faces.
    ind0 = 0
    ind1 = g_0.num_faces * proj_dim
    assert np.sum(subdomain_to_boundary[:, ind0:ind1]) == 8 * proj_dim
    # Check that the projection matrix for the first fracture is non-zero. Since the
    # fracture touches the boundary on two sides, we expect two non-zero rows.
    ind0 = ind1
    ind1 += g_1.num_faces * proj_dim
    assert np.sum(subdomain_to_boundary[:, ind0:ind1]) == 2 * proj_dim
    # Check that the projection matrix for the second fracture is non-zero.
    ind0 = ind1
    ind1 += g_2.num_faces * proj_dim
    assert np.sum(subdomain_to_boundary[:, ind0:ind1]) == 2 * proj_dim
    # The projection matrix for the intersection should be zero.
    ind0 = ind1
    assert np.sum(subdomain_to_boundary[:, ind0:]) == 0

    # Make second projection on subset of grids.
    subdomains = [g_0, g_1]
    projection = pp.ad.grid_operators.BoundaryProjection(mdg, subdomains, proj_dim)
    num_faces = proj_dim * (g_0.num_faces + g_1.num_faces)
    num_cells = proj_dim * sum(
        [mdg.subdomain_to_boundary_grid(sd).num_cells for sd in subdomains]
    )
    # Obtaining sparse matrices from the AD Operators.
    subdomain_to_boundary = projection.subdomain_to_boundary.parse(mdg)
    boundary_to_subdomain = projection.boundary_to_subdomain.parse(mdg)
    # Check sizes.
    assert subdomain_to_boundary.shape == (num_cells, num_faces)
    assert boundary_to_subdomain.shape == (num_faces, num_cells)

    # Check that the projection matrix for the top-dimensional grid is non-zero.
    # Same sizes as above.
    ind0 = 0
    ind1 = g_0.num_faces * proj_dim
    assert np.sum(subdomain_to_boundary[:, ind0:ind1]) == 8 * proj_dim
    ind0 = ind1
    ind1 += g_1.num_faces * proj_dim
    assert np.sum(subdomain_to_boundary[:, ind0:ind1]) == 2 * proj_dim

    # Check that subdomain_to_boundary and boundary_to_subdomain are consistent.
    assert np.allclose((subdomain_to_boundary - boundary_to_subdomain.T).data, 0)


@pytest.mark.integtest
# Geometry based operators
def test_trace(mdg: pp.MixedDimensionalGrid):
    """Test Trace operator.

    Parameters:
        mdg: Mixed-dimensional grid.

    This test is not ideal. It follows the implementation of Trace relatively closely,
    but nevertheless provides some coverage, especially if Trace is carelessly changed.
    The test constructs the expected md trace and inv_trace matrices and compares them
    to the ones of Trace. Also checks that an error is raised if a non-scalar trace is
    constructed (not implemented).
    """
    # The operator should work on any subset of mdg.subdomains.
    subdomains = mdg.subdomains(dim=1)

    # Construct expected matrices
    traces, inv_traces = list(), list()
    # No check on this function here.
    # TODO: A separate unit test might be appropriate.
    cell_projections, face_projections = pp.ad.grid_operators._subgrid_projections(
        subdomains, dim=1
    )
    for sd in subdomains:
        local_block = np.abs(sd.cell_faces.tocsr())
        traces.append(local_block * cell_projections[sd].T)
        inv_traces.append(local_block.T * face_projections[sd].T)

    # Compare to operator class
    op = pp.ad.Trace(subdomains)
    _compare_matrices(op.trace, sps.bmat([[m] for m in traces]))
    _compare_matrices(op.inv_trace, sps.bmat([[m] for m in inv_traces]))

    # As of the writing of this test, Trace is not implemented for vector values.
    # If it is ever extended, the test should be extended accordingly (e.g. parametrized with
    # dim=[1, 2]).
    with pytest.raises(NotImplementedError):
        pp.ad.Trace(subdomains, dim=2)


@pytest.mark.integtest
@pytest.mark.parametrize("dim", [1, 4])
def test_divergence(mdg: pp.MixedDimensionalGrid, dim: int):
    """Test Divergence.

    Parameters:
        mdg: Mixed-dimensional grid.
        dim: Dimension of vector field to which Divergence is applied.

    This test is not ideal. It follows the implementation of Divergence relatively
    closely, but nevertheless provides some coverage. Frankly, there is not much more to
    do than comparing against the expected matrices, unless one wants to add more
    integration-type tests e.g. evaluating combinations with other ad entities.

    """
    # The operator should work on any subset of mdg.subdomains.
    subdomains = mdg.subdomains(dim=2) + mdg.subdomains(dim=0)

    # Construct expected matrix
    divergences = list()
    for sd in subdomains:
        # Kron does no harm if dim=1
        local_block = sps.kron(sd.cell_faces.tocsr().T, sps.eye(dim))
        divergences.append(local_block)

    # Compare to operators parsed value
    op = pp.ad.Divergence(subdomains)
    val = op.parse(mdg)
    _compare_matrices(val, sps.block_diag(divergences))


def _compare_matrices(m1, m2):
    # Convert ad sparse arrays to scipy sparse matrices if necessary. Then call the
    # standard comparison function for matrices.
    if isinstance(m1, pp.ad.SparseArray):
        m1 = m1._mat
    if isinstance(m2, pp.ad.SparseArray):
        m2 = m2._mat
    return pp.applications.test_utils.arrays.compare_matrices(m1, m2)


def _list_ind_of_grid(subdomains, g):
    for i, gl in enumerate(subdomains):
        if g == gl:
            return i
    raise ValueError("grid is not in list")


def geometry_information(
    mdg: pp.MixedDimensionalGrid, dim: int
) -> tuple[int, int, int]:
    """Geometry information used in multiple test methods.

    Parameters:
        mdg: Mixed-dimensional grid.
        dim: Dimension. Each of the return values is multiplied by dim.

    Returns:
        n_cells (int): Number of subdomain cells.
        n_faces (int): Number of subdomain faces.
        n_mortar_cells (int): Number of interface cells.
    """
    n_cells = sum([sd.num_cells for sd in mdg.subdomains()]) * dim
    n_faces = sum([sd.num_faces for sd in mdg.subdomains()]) * dim
    n_mortar_cells = sum([intf.num_cells for intf in mdg.interfaces()]) * dim
    return n_cells, n_faces, n_mortar_cells
