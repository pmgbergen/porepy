"""Tests for the MPSA discretization scheme."""
import numpy as np
import scipy.sparse as sps

import pytest
import porepy as pp
from porepy.applications.test_utils.partial_discretization import (
    perform_partial_discretization_specified_nodes,
)


@pytest.fixture
def discretization_matrices():
    """Return a grid and the corresponding mpsa discretization matrices."""
    g = pp.CartGrid([5, 5])
    g.compute_geometry()
    stiffness = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
    bnd = pp.BoundaryConditionVectorial(g)
    stress, bound_stress, _, _ = pp.Mpsa("mechanics")._stress_discretization(
        g, stiffness, bnd, inverter="python"
    )
    return g, stiffness, bnd, stress, bound_stress


def partial_update_parameters(stiffness, bnd):
    """Return parameter dictionary for partial update tests."""
    specified_data = {
        "fourth_order_tensor": stiffness,
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
    g, stiffness, bnd, stress_full, bound_stress_full = discretization_matrices
    specified_data = partial_update_parameters(stiffness, bnd)
    keyword = "mechanics"
    discr = pp.Mpsa(keyword)
    data = perform_partial_discretization_specified_nodes(
        g, discr, specified_data, cell_id
    )

    partial_stress = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
    partial_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
        discr.bound_stress_matrix_key
    ]

    active_faces = data[pp.PARAMETERS][keyword]["active_faces"]
    active_faces_nd = pp.fvutils.expand_indices_nd(active_faces, g.dim)

    for partial, full in zip(
        [partial_stress, partial_bound],
        [stress_full, bound_stress_full],
    ):
        assert np.allclose(
            partial.todense()[active_faces_nd],
            full.todense()[active_faces_nd],
        )
        # For partial update, only the active faces should be nonzero. Force these to
        # zero and check that the rest is zero.
        pp.fvutils.remove_nonlocal_contribution(active_faces_nd, 1, partial)
        assert np.allclose(partial.data, 0)


def test_partial_discretization_one_cell_at_a_time():
    g = pp.CartGrid([3, 3])
    g.compute_geometry()

    # Assign random permeabilities, for good measure
    np.random.seed(42)
    mu = np.random.random(g.num_cells)
    lmbda = np.random.random(g.num_cells)
    stiffness = pp.FourthOrderTensor(mu=mu, lmbda=lmbda)
    bnd = pp.BoundaryConditionVectorial(g)
    specified_data = partial_update_parameters(stiffness, bnd)
    keyword = "mechanics"
    discr = pp.Mpsa(keyword)
    data = pp.initialize_default_data(
        g, {}, keyword, specified_parameters=specified_data
    )
    discr.discretize(g, data)

    stress_full = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
    bound_stress_full = data[pp.DISCRETIZATION_MATRICES][keyword][
        discr.bound_stress_matrix_key
    ]

    # Initialize matrices to be filled one cell at a time
    stress = sps.csr_matrix((g.num_faces * g.dim, g.num_cells * g.dim))
    bound_stress = sps.csr_matrix((g.num_faces * g.dim, g.num_faces * g.dim))
    faces_covered = np.zeros(g.num_faces, bool)

    for cell_id in range(g.num_cells):
        data = perform_partial_discretization_specified_nodes(
            g, discr, specified_data, cell_id
        )
        # Extract current matrices
        partial_stress = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.stress_matrix_key
        ]
        partial_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ]

        active_faces = data[pp.PARAMETERS][keyword]["active_faces"]

        if np.any(faces_covered):
            del_faces = pp.fvutils.expand_indices_nd(np.where(faces_covered)[0], g.dim)
            pp.fvutils.remove_nonlocal_contribution(
                del_faces, 1, partial_stress, partial_bound
            )

        faces_covered[active_faces] = True

        stress += partial_stress
        bound_stress += partial_bound

    # Compare results
    for partial, full in zip([stress, bound_stress], [stress_full, bound_stress_full]):
        assert pp.test_utils.arrays.compare_matrices(partial, full, 1e-10)
