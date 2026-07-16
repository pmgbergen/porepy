"""Tests for the linear solver class."""

from typing import Literal

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from porepy.numerics.solvers.linear_solvers.linear_solver import LinearSolverDirect, LinearSystem


def make_linear_system(case: Literal["nonsingular", "singular"]) -> LinearSystem:
    """Data for test cases."""
    if case == "nonsingular":
        mat = csr_matrix(
            np.array(
                [
                    [1, 2, 3],
                    [2, 3, 1],
                    [3, 2, 1],
                ],
                dtype=float,
            )
        )
        rhs = np.ones(3, dtype=float)
    elif case == "singular":
        mat = csr_matrix(
            np.array(
                [
                    [1, 2, 3],
                    [2, 3, 1],
                    [1, 2, 3],
                ],
                dtype=float,
            )
        )
        rhs = np.array([1, 2, 3], dtype=float)
    else:
        raise ValueError(case)
    return LinearSystem(matrix=mat, rhs=rhs)


@pytest.mark.parametrize("case", ["nonsingular", "singular"])
# Testing all backends known backends and how it behaves with a bad backend tag.
# If the backend library is not installed, it falls back to the installed one, so the
# test should pass regardless.
@pytest.mark.parametrize(
    "backend", ["pypardiso", "umfpack", "scipy_sparse", "unknown_backend"]
)
def test_linear_solver_direct(case: str, backend: str):
    """Tests that the direct linear solver works as expected."""
    linear_system = make_linear_system(case=case)
    linear_solver = LinearSolverDirect(backend=backend)
    if backend != "unknown_backend":
        sol, status = linear_solver.solve_linear_system(linear_system)
    else:
        with pytest.raises(ValueError):
            linear_solver.solve_linear_system(linear_system)
        return

    mat = linear_system.matrix
    rhs = linear_system.rhs
    assert mat is not None
    if case == "nonsingular":
        assert status.is_success()
        assert np.allclose(mat @ sol - rhs, 0, rtol=0, atol=1e-10)
    elif case == "singular":
        # Returned array can have anything inside, but with the right shape and dtype.
        assert sol.shape == rhs.shape
        assert sol.dtype == rhs.dtype
    else:
        raise ValueError(case)


def test_linear_solver_direct_rejects_released_matrix() -> None:
    """A direct solver cannot solve a system after its matrix is released."""
    linear_system = make_linear_system(case="nonsingular")
    linear_system.release_matrix_reference()

    with pytest.raises(ValueError, match="matrix was released"):
        LinearSolverDirect(backend="scipy_sparse").solve_linear_system(linear_system)
