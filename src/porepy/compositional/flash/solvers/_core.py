"""Core functionality and definition for the flash solver subpackage.

Contains definitions and signatures for compiled flash equations, solvers and other
structures used in the numba framework.

Important:
    Everything in this *private* module is performance critical.
    It is critical for execution and compilation performance of solvers, as well as for
    code import performance when loading PorePy.

    Changes should only be done with much care.

"""

from __future__ import annotations

import logging
from typing import Callable, Literal

import numba as nb
import numpy as np

from ..._numba_interface import (
    NUMBA_PARALLEL,
    cfunc,
    get_empty_numba_dict,
    njit,
    typeof,
)
from ...utils import FlashSpec, FlashSpec_NUMBA_TYPE

__all__ = [
    "GENERAL_SOLVER_PARAMS",
    "SOLVER_PARAMETERS_TYPE",
    "FLASH_RESIDUAL_SIGNATURE",
    "FLASH_JACOBIAN_SIGNATURE",
    "FLASH_RESIDUAL_FUNCTION_TYPE",
    "FLASH_JACOBIAN_FUNCTION_TYPE",
    "SOLVER_FUNCTION_SIGNATURE",
    "sequential_solver",
    "parallel_solver",
    "multi_solve",
]


logger = logging.getLogger(__name__)


GENERAL_SOLVER_PARAMS: dict[
    Literal["num_components", "num_phases", "max_iterations", "atol_res", "f_dim"],
    float,
] = {}
"""Dummy dictionary typing general solver parameters which are expected to be passed to
a solver.

- ``'num_components'`` the number of components.
- ``'num_phases'`` the number of phases.
- ``'max_iterations'`` maximal number of iterations.
- ``'atol_res'`` the tolerance for the convergence criterion.
- ``'f_dim'`` dimension of the flash system.

"""


SOLVER_PARAMETERS_TYPE = typeof(get_empty_numba_dict())
"""Numba-type definition of the solver parameter dictionary.

A solver parameter dictionary has strings as keys and ``float64`` as values.

Note:
    Numba does not allow multiple types in keys or strings (as of now).
    If a parameter is actually an integer, it must be converted to a float before
    setting it in the dictionary. Solvers must internally convert them back to integers.

"""


FLASH_RESIDUAL_SIGNATURE = nb.f8[:](nb.f8[:])
"""Numba-signature for a flash residual function.

Takes a 1D array of ``float64`` values and returns a 1D array of ``float64``.

"""


@cfunc(FLASH_RESIDUAL_SIGNATURE, cache=True)
def flash_residual_template_func(x: np.ndarray) -> np.ndarray:
    """Template c-function for a flash residual function ``(f8[:]) -> f8[:]``.

    Used for automatic type-inferring.

    Parameters:
        x: Generic flash argument.

    Returns:
        The residual of an equilibrium system.

    """
    return x.copy()


FLASH_RESIDUAL_FUNCTION_TYPE = typeof(flash_residual_template_func)
"""Numba type for a flash residual function, which takes a 1D array and returns a 1D
array (both of ``float64`` values).

Used to type cached, numba-compiled solvers.

See also:
    :func:`flash_residual_template_func`

"""


FLASH_JACOBIAN_SIGNATURE = nb.f8[:, :](nb.f8[:])
"""Numba-signature for a flash Jacobian function.

Takes a 1D array of ``float64`` values and returns a 2D array of ``float64``.

"""


@cfunc(FLASH_JACOBIAN_SIGNATURE, cache=True)
def flash_jacobian_template_func(x: np.ndarray) -> np.ndarray:
    """Template c-function for a flash Jacobian function ``(f8[:]) -> f8[:,:]``.

    Used for automatic type-inferring.

    See also:
        :func:`flash_residual_template_func`

    Parameters:
        x: Generic flash argument.

    Returns:
        The Jacobian of an equilibrium system.

    """
    return np.diag(x)


FLASH_JACOBIAN_FUNCTION_TYPE = typeof(flash_jacobian_template_func)
"""Numba type for a flash Jacobian function, which takes a 1D array and returns a 2D
array (both of ``float64`` values).

Used to type cached, numba-compiled solvers.

See also:
    :func:`flash_jacobian_template_func`

"""


SOLVER_FUNCTION_SIGNATURE = nb.types.Tuple((nb.f8[:], nb.int_, nb.int_))(
    nb.f8[:],
    FLASH_RESIDUAL_FUNCTION_TYPE,
    FLASH_JACOBIAN_FUNCTION_TYPE,
    SOLVER_PARAMETERS_TYPE,
    FlashSpec_NUMBA_TYPE,
)
"""Numba signature for flash solvers.

To be used as the signature argument for :obj:`numba.njit` when compiling a solver.

See :data:`SOLVER_FUNCTION_TYPE` for more information on the signature.

"""


@cfunc(SOLVER_FUNCTION_SIGNATURE, cache=True)
def solver_template_func(
    x: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    solver_params: dict[str, float],
    spec: FlashSpec,
) -> tuple[np.ndarray, int, int]:
    """Template c-function for solvers.

    Parameters:
        x: Initial guess for the flash system to be solved.
        F: Flash residual function..
        DF: Flash Jacobian function.
        solver_params: Dictionary of relevant solver parameters (str-float pairs).

    Returns:
        The solution vector (1D array), an exit code (int), and the number of
        iterations (int).

    """
    return F(x), 0, 3


SOLVER_FUNCTION_TYPE = typeof(solver_template_func)
"""Numba type for a flash solver, which takes

1. an initial guess (1D array),
2. the flash residual function ``(f8[:]) -> f8[:]``,
3. the flash Jacobian function ``(f8[:]) -> f8[:,;]``, and
4. solver parameters (``dict[str, float]``),
5. the flash specification (:class:`~porepy.compositional.flash.abstract_flash.
   FlashSpec`),

and returns 

1. the result (1D array),
2. a convergence code (int), and
3. the number of iterations required.

The exit codes must be as follows:

- 0: converged (success)
- 1: maximal number of iterations reached
- 2: stationary point (unresolved stagnation)
- 3: diverged (``NAN`` or ``infty`` detected in update or residual)
- 4: failure in evaluation of residual or Jacobian (nans).
- 5: error caught by multisolvers

Note:
    Error code 5 is generated by the multi-solver in a generic try-except clause around
    the call to the solver on a single problem. It indicates a failure for unknown
    reasons and should be investigated (likely in evaluation of residual or Jacobian).

See also:
    :func:`solver_template_func`

"""


_multi_solver_signature = nb.types.Tuple(
    # NOTE: Since the return values are created internally, they are contiguous arrays.
    # Numba requires that information explicitly by using ::1 in the last dimension.
    (nb.f8[:, ::1], nb.int_[::1], nb.int_[::1])
)(
    nb.f8[:, :],
    FLASH_RESIDUAL_FUNCTION_TYPE,
    FLASH_JACOBIAN_FUNCTION_TYPE,
    SOLVER_FUNCTION_TYPE,
    SOLVER_PARAMETERS_TYPE,
    FlashSpec_NUMBA_TYPE,
)
"""Multi-solver signature for compiled sequential or parallel application of solvers."""


@njit(_multi_solver_signature, cache=True)
def sequential_solver(
    X0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    solver: Callable[
        [
            np.ndarray,
            Callable[[np.ndarray], np.ndarray],
            Callable[[np.ndarray], np.ndarray],
            dict[str, float],
            FlashSpec,
        ],
        tuple[np.ndarray, int, int],
    ],
    solver_params: dict[str, float],
    spec: FlashSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sequential application of a solver to vectorized input.

    The solver is applied row-wise on ``X0``.

    Parameters:
        X0: 2D array, where each row is an initial guess for an individual problem.
        F: Flash residual function (see :data:`FLASH_RESIDUAL_FUNCTION_TYPE`).
        DF: Flash Jacobian function (see :data:`FLASH_JACOBIAN_FUNCTION_TYPE`).
        solver: Solver function (see :data:`SOLVER_FUNCTION_TYPE`).
        solver_params: Solver parameters passed to every problem.
        spec: Flash specification passed to every problem.

    Returns:
        The results, convergence flags and number of iterations, vectorized where each
        row corresponds to a row in ``X0``.

    """

    # alocating return values
    n = X0.shape[0]
    result = np.zeros_like(X0)
    num_iter = np.zeros(n, dtype=np.int_)
    exitcodes = np.ones(n, dtype=np.int_) * 5

    for i in range(n):
        try:
            res_i, e_i, n_i = solver(X0[i], F, DF, solver_params, spec)
        except Exception:
            res_i = X0[i]
            e_i = 5
            n_i = -1
        exitcodes[i] = e_i
        num_iter[i] = n_i
        result[i] = res_i

    return result, exitcodes, num_iter


@njit(_multi_solver_signature, cache=True, parallel=NUMBA_PARALLEL)
def parallel_solver(
    X0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    solver: Callable[
        [
            np.ndarray,
            Callable[[np.ndarray], np.ndarray],
            Callable[[np.ndarray], np.ndarray],
            dict[str, float],
            FlashSpec,
        ],
        tuple[np.ndarray, int, int],
    ],
    solver_params: dict[str, float],
    spec: FlashSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel application of a solver to vectorized input.

    Otherwise analogous to :func:`sequential_solver`.

    To be used for a large quantity of problems, where parallelization outperforms
    the sequential solver.

    Important:
        As of now, numba does not support ``try.. except`` in the parallel environment.
        This makes this function fragile to exceptions thrown by the solver.
        If an exception is thrown, the whole parallel execution is aborted and the call
        is returned with an exception.

    """
    n = X0.shape[0]
    result = np.zeros_like(X0)
    num_iter = np.zeros(n, dtype=np.int_)
    exitcodes = np.ones(n, dtype=np.int_) * 5

    for i in nb.prange(n):
        res_i, e_i, n_i = solver(X0[i], F, DF, solver_params, spec)
        exitcodes[i] = e_i
        num_iter[i] = n_i
        result[i] = res_i

    return result, exitcodes, num_iter


def multi_solve(
    mode: str,
    X0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    solver: Callable[
        [
            np.ndarray,
            Callable[[np.ndarray], np.ndarray],
            Callable[[np.ndarray], np.ndarray],
            dict[str, float],
            FlashSpec,
        ],
        tuple[np.ndarray, int, int],
    ],
    solver_params: dict[str, float],
    spec: FlashSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper function for multi-solvers.

    Temporary work-around for when parallel execution fails. In that case, the
    sequential multi-solver is used instead, and the failure is logged.

    See also:
        :func:`sequential_solver`, :func:`parallel_solver`

    Parameters:
        mode: Either "sequential" or "parallel". If "parallel" is selected but fails,
            the sequential multi-solver is used instead.
        X0: 2D array, where each row is an initial guess for an individual problem.
        F: Flash residual function (see :data:`FLASH_RESIDUAL_FUNCTION_TYPE`).
        DF: Flash Jacobian function (see :data:`FLASH_JACOBIAN_FUNCTION_TYPE`).
        solver: Solver function (see :data:`SOLVER_FUNCTION_TYPE`).
        solver_params: Solver parameters passed to every problem (see
            :data:`SOLVER_PARAMETERS_TYPE`).
        spec: Flash specification passed to every problem.

    """
    if mode == "sequential":
        return sequential_solver(X0, F, DF, solver, solver_params, spec)
    elif mode == "parallel":
        try:
            return parallel_solver(X0, F, DF, solver, solver_params, spec)
        except Exception:
            logger.warning(
                "Parallel multi-solver failed with exception. "
                "Falling back to sequential multi-solver. "
                "Investigate the failure of the parallel solver.",
            )
            return sequential_solver(X0, F, DF, solver, solver_params, spec)
    else:
        raise ValueError(
            f"Invalid multi-solver mode: {mode}. Choose 'sequential' or 'parallel'."
        )
