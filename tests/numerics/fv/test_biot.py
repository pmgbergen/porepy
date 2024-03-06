"""Tests for the Biot discretization."""
import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils import common_xpfa_tests as xpfa_tests
from porepy.applications.test_utils.partial_discretization import (
    perform_partial_discretization_specified_nodes,
)


@pytest.fixture
def mechanics_keyword():
    return "mechanics"


@pytest.fixture
def flow_keyword():
    return "flow"


@pytest.fixture
def discretization_matrices(flow_keyword, mechanics_keyword):
    """Return a grid and the corresponding Biot discretization matrices."""
    g = pp.CartGrid([5, 5])
    g.compute_geometry()
    stiffness = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
    bnd = pp.BoundaryConditionVectorial(g)
    data = pp.initialize_default_data(
        g,
        {},
        mechanics_keyword,
        specified_parameters=partial_update_parameters(stiffness, bnd, g.num_cells),
    )
    data = pp.initialize_default_data(g, data, flow_keyword)

    discr = pp.Biot()
    discr.discretize(g, data)
    div_u = data[pp.DISCRETIZATION_MATRICES][mechanics_keyword][discr.div_u_matrix_key]
    bound_div_u = data[pp.DISCRETIZATION_MATRICES][mechanics_keyword][
        discr.bound_div_u_matrix_key
    ]
    stab = data[pp.DISCRETIZATION_MATRICES][mechanics_keyword][
        discr.stabilization_matrix_key
    ]
    grad_p = data[pp.DISCRETIZATION_MATRICES][mechanics_keyword][
        discr.grad_p_matrix_key
    ]
    bound_pressure = data[pp.DISCRETIZATION_MATRICES][mechanics_keyword][
        discr.bound_pressure_matrix_key
    ]
    return g, stiffness, bnd, div_u, bound_div_u, grad_p, stab, bound_pressure


def partial_update_parameters(stiffness, bnd, num_cells):
    """Return parameter dictionary for partial update tests."""

    tensor = pp.SecondOrderTensor(kxx=np.ones(num_cells), kyy=10*np.ones(num_cells))

    specified_data = {
        "fourth_order_tensor": stiffness,
        "bc": bnd,
        "inverter": "python",
        "scalar_vector_mappings": {'foo': 1, 'bar': tensor},
    }
    return specified_data


@pytest.mark.parametrize(
    "cell_id,faces_of_cell",
    [
        (10, np.array([12, 13, 40, 45])),  # cell at domain boundary
        (12, np.array([14, 15, 42, 47])),  # internal cell
    ],
)
def test_partial_discretization_specified_nodes(
    cell_id: int,
    faces_of_cell: np.ndarray,
    discretization_matrices,
    mechanics_keyword,
    flow_keyword,
):
    """Test that the discretization matrices are correct for a partial update.

    Parameters:
        cell_id: The cell whose node will be used to define the partial update. The
            nodes are identified based on the provided cell_id outside of the
            discretization method.
        faces_of_cell: The faces neighboring the cell.

    """
    (
        g,
        stiffness,
        bnd,
        div_u_full,
        bound_div_u_full,
        grad_p_full,
        stab_full,
        bound_pressure_full,
    ) = discretization_matrices
    specified_data = partial_update_parameters(stiffness, bnd, g.num_cells)
    discr = pp.Biot(mechanics_keyword)
    discr.keyword = mechanics_keyword
    data = perform_partial_discretization_specified_nodes(
        g, discr, specified_data, cell_id
    )

    partial_div_u = data[pp.DISCRETIZATION_MATRICES][mechanics_keyword][
        discr.div_u_matrix_key
    ]
    partial_bound_div_u = data[pp.DISCRETIZATION_MATRICES][mechanics_keyword][
        discr.bound_div_u_matrix_key
    ]
    partial_grad_p = data[pp.DISCRETIZATION_MATRICES][mechanics_keyword][
        discr.grad_p_matrix_key
    ]
    partial_stab = data[pp.DISCRETIZATION_MATRICES][mechanics_keyword][
        discr.stabilization_matrix_key
    ]
    partial_bound_pressure = data[pp.DISCRETIZATION_MATRICES][mechanics_keyword][
        discr.bound_pressure_matrix_key
    ]

    active_faces = data[pp.PARAMETERS][mechanics_keyword]["active_faces"]
    active_cells = data[pp.PARAMETERS][mechanics_keyword]["active_cells"]

    assert faces_of_cell.size == active_faces.size
    assert np.all(np.sort(faces_of_cell) == np.sort(active_faces))

    active_faces_nd = pp.fvutils.expand_indices_nd(active_faces, g.dim)

    # Compare vector matrices
    for partial, full in zip(
        [partial_grad_p, partial_bound_pressure],
        [grad_p_full, bound_pressure_full],
    ):
        if isinstance(partial, dict):
            for key in partial.keys():
                assert np.allclose(
                    partial[key].todense()[active_faces_nd],
                    full[key].todense()[active_faces_nd],
                )
            # For partial update, only the active faces should be nonzero. Force these
            # to zero and check that the rest is zero.
            pp.fvutils.remove_nonlocal_contribution(active_faces_nd, 1, partial[key])
            assert np.allclose(partial[key].data, 0)

        else:
            assert np.allclose(
                partial.todense()[active_faces_nd],
                full.todense()[active_faces_nd],
            )
            # For partial update, only the active faces should be nonzero. Force these
            # to zero and check that the rest is zero.
            pp.fvutils.remove_nonlocal_contribution(active_faces_nd, 1, partial)
            assert np.allclose(partial.data, 0)
    # Compare scalar matrices
    for partial, full in zip(
        [partial_div_u, partial_bound_div_u, partial_stab],
        [div_u_full, bound_div_u_full, stab_full],
    ):
        if isinstance(partial, dict):
            for key in partial.keys():
                assert np.allclose(
                    partial[key].todense()[cell_id],
                    full[key].todense()[cell_id],
                )
            # For partial update, only the active cells should be nonzero. Force these
            # to zero and check that the rest is zero.
            pp.fvutils.remove_nonlocal_contribution(active_cells, 1, partial[key])
            assert np.allclose(partial[key].data, 0)

        else:
            assert np.allclose(
                partial.todense()[cell_id],
                full.todense()[cell_id],
            )
            # For partial update, only the active cells should be nonzero. Force these
            # to zero and check that the rest is zero.
            pp.fvutils.remove_nonlocal_contribution(active_cells, 1, partial)
            assert np.allclose(partial.data, 0)


def test_split_discretization_into_parts():
    """Test that the discretization matrices are correct if the domain is split into
    subdomains.

    This test is just a shallow wrapper around the common test function for the XPFA
    discretization.
    """
    discr = pp.Biot(keyword="mechanics")
    xpfa_tests.test_split_discretization_into_subproblems(discr)
