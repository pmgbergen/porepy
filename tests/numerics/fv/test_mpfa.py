"""Tests for the MPFA discretization scheme."""
import numpy as np
import scipy.sparse as sps

import pytest
import porepy as pp
from porepy.applications.test_utils.partial_discretization import (
    perform_partial_discretization_specified_nodes,
)


@pytest.fixture
def discretization_matrices():
    """Return a grid and the corresponding mpfa discretization matrices."""
    g = pp.CartGrid([5, 5])
    g.compute_geometry()
    perm = pp.SecondOrderTensor(np.ones(g.num_cells))
    bnd = pp.BoundaryCondition(g)
    flux, bound_flux, _, _, vector_source, _ = pp.Mpfa("flow")._flux_discretization(
        g, perm, bnd, inverter="python"
    )
    return g, perm, bnd, flux, bound_flux, vector_source


def partial_update_parameters(perm, bnd):
    """Return parameter dictionary for partial update tests."""
    specified_data = {
        "second_order_tensor": perm,
        "bc": bnd,
        "inverter": "python",
    }
    return specified_data


@pytest.mark.parametrize(
    "cell_id",
    [
        10,  # cell at domain boundary
        12,  # internal cell
    ],
)
def test_partial_discretization_specified_nodes(cell_id: int, discretization_matrices):
    """Test that the discretization matrices are correct for a partial update.

    Parameters:
        cell_id: The cell whose node will be used to define the partial update. The
            nodes are identified based on the provided cell_id outside of the
            discretization method.

    """
    g, perm, bnd, flux_full, bound_flux_full, vector_src_full = discretization_matrices
    specified_data = partial_update_parameters(perm, bnd)
    keyword = "flow"
    discr = pp.Mpfa(keyword)
    data = perform_partial_discretization_specified_nodes(
        g, discr, specified_data, cell_id
    )

    partial_flux = data[pp.DISCRETIZATION_MATRICES][keyword][discr.flux_matrix_key]
    partial_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
        discr.bound_flux_matrix_key
    ]
    partial_vector_source = data[pp.DISCRETIZATION_MATRICES][keyword][
        discr.vector_source_matrix_key
    ]

    active_faces = data[pp.PARAMETERS][keyword]["active_faces"]

    for partial, full in zip(
        [partial_flux, partial_bound, partial_vector_source],
        [flux_full, bound_flux_full, vector_src_full],
    ):
        assert pp.test_utils.arrays.compare_matrices(
            partial[active_faces], full[active_faces], 1e-10
        )
        # For partial update, only the active faces should be nonzero. Force these to
        # zero and check that the rest is zero.
        pp.matrix_operations.zero_rows(partial, active_faces)
        assert np.allclose(partial.data, 0)


def test_partial_discretization_one_cell_at_a_time():
    g = pp.CartGrid([3, 3])
    g.compute_geometry()

    # Assign random permeabilities, for good measure
    np.random.seed(42)
    kxx = np.random.random(g.num_cells)
    kyy = np.random.random(g.num_cells)
    # Ensure positive definiteness
    kxy = np.random.random(g.num_cells) * kxx * kyy
    perm = pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kxy=kxy)
    bnd = pp.BoundaryCondition(g)
    specified_data = partial_update_parameters(perm, bnd)
    keyword = "flow"
    discr = pp.Mpfa(keyword)

    # Initialize matrices to be filled one cell at a time
    flux = sps.csr_matrix((g.num_faces, g.num_cells))
    bound_flux = sps.csr_matrix((g.num_faces, g.num_faces))
    vector_src = sps.csr_matrix((g.num_faces, g.num_cells * g.dim))
    faces_covered = np.zeros(g.num_faces, bool)

    for cell_id in range(g.num_cells):
        data = perform_partial_discretization_specified_nodes(
            g, discr, specified_data, cell_id
        )
        partial_flux = data[pp.DISCRETIZATION_MATRICES][keyword][discr.flux_matrix_key]
        partial_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_flux_matrix_key
        ]
        partial_vector_source = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.vector_source_matrix_key
        ]

        active_faces = data[pp.PARAMETERS][keyword]["active_faces"]

        if np.any(faces_covered):
            fi = np.where(faces_covered)[0]
            pp.fvutils.remove_nonlocal_contribution(
                fi, 1, partial_flux, partial_bound, partial_vector_source
            )

        faces_covered[active_faces] = True

        flux += partial_flux
        bound_flux += partial_bound
        vector_src += partial_vector_source

    # Use direct discretization for all cells.
    flux_full, bound_flux_full, *_, vector_src_full, _ = pp.Mpfa(
        "flow"
    )._flux_discretization(g, perm, bnd, inverter="python")

    # Compare results
    for partial, full in zip(
        [flux, bound_flux, vector_src], [flux_full, bound_flux_full, vector_src_full]
    ):
        assert pp.test_utils.arrays.compare_matrices(partial, full, 1e-10)
