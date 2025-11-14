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

from typing import Callable, Literal

# NOTE import numba.typed like this to avoid importing the spurious py.typed file in the
# typed sub-package, which confuses mypy.
import numba
import numba.typed
import numpy as np

from ..._core import NUMBA_PARALLEL, cfunc, typeof
from ..abstract_flash import FlashSpec, FlashSpecMember_NUMBA_TYPE

__all__ = [
    "GENERAL_SOLVER_PARAMS",
    "SOLVER_PARAMETERS_TYPE",
    "FLASH_RESIDUAL_FUNCTION_TYPE",
    "FLASH_JACOBIAN_FUNCTION_TYPE",
    "SOLVER_FUNCTION_SIGNATURE",
    "sequential_solver",
    "parallel_solver",
    "MULTI_SOLVERS",
]


GENERAL_SOLVER_PARAMS: dict[
    Literal["num_components", "num_phases", "max_iterations", "tolerance", "f_dim"],
    float,
] = {}
"""Dummy dictionary typing general solver parameters which are expected to be passed to
a solver.

- ``'num_components'`` the number of components.
- ``'num_phases'`` the number of phases.
- ``'max_iterations'`` maximal number of iterations.
- ``'tolerance'`` the tolerance for the convergence criterion.
- ``'f_dim'`` dimension of the flash system.

"""


# Internal dummy for numba type inference.
_solver_parameters: dict[str, float] = numba.typed.Dict.empty(
    key_type=numba.types.unicode_type, value_type=numba.types.float64
)
# Due to unknown reasons, we have to set some key-value pair in some cases.
_solver_parameters["a"] = 0.0

SOLVER_PARAMETERS_TYPE = typeof(_solver_parameters)
"""Numba-type definition of the solver parameter dictionary.

A solver parameter dictionary has strings as keys and ``float64`` as values.

Note:
    Numba does not allow multiple types in keys or strings (as of now).
    If a parameter is actually an integer, it must be converted to a float before
    setting it in the dictionary. Solvers must internally convert them back to integers.

"""


@cfunc(numba.f8[:](numba.f8[:]), cache=True)
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


@cfunc(numba.f8[:, :](numba.f8[:]), cache=True)
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


SOLVER_FUNCTION_SIGNATURE = numba.types.Tuple((numba.f8[:], numba.int_, numba.int_))(
    numba.f8[:],
    FLASH_RESIDUAL_FUNCTION_TYPE,
    FLASH_JACOBIAN_FUNCTION_TYPE,
    SOLVER_PARAMETERS_TYPE,
    FlashSpecMember_NUMBA_TYPE,
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
    # Some operations are required for numba in order to create the type.
    return F(x) + DF(x) @ x, 1, 1


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

- 0: successful solution procedure
- 1: maximal number of iterations reached
- 2: ``NAN`` or ``infty`` detected in update (aborted)
- 3: failure in the evaluation of the residual
- 4: failure in the evaluation of the Jacobian
- 5: Any other failure

See also:
    :func:`solver_template_func`

"""


_multi_solver_signature = numba.types.Tuple(
    # NOTE: Since the return values are created internally, they are contiguous arrays.
    # Numba requires that information explicitly by using ::1 in the last dimension.
    (numba.f8[:, ::1], numba.int_[::1], numba.int_[::1])
)(
    numba.f8[:, :],
    FLASH_RESIDUAL_FUNCTION_TYPE,
    FLASH_JACOBIAN_FUNCTION_TYPE,
    SOLVER_FUNCTION_TYPE,
    SOLVER_PARAMETERS_TYPE,
    FlashSpecMember_NUMBA_TYPE,
)
"""Multi-solver signature for compiled sequential or parallel application of solvers."""


@numba.njit(_multi_solver_signature, cache=True)
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
            exitcodes[i] = 5
            num_iter[i] = -1
            result[i, :] = np.nan
        else:
            exitcodes[i] = e_i
            num_iter[i] = n_i
            result[i] = res_i

    return result, exitcodes, num_iter


@numba.njit(_multi_solver_signature, cache=True, parallel=NUMBA_PARALLEL)
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
        If an exception is thrown, the whole parallel execution is aborted.

    """
    n = X0.shape[0]
    result = np.zeros_like(X0)
    num_iter = np.zeros(n, dtype=np.int_)
    exitcodes = np.ones(n, dtype=np.int_) * 5

    for i in numba.prange(n):
        res_i, e_i, n_i = solver(X0[i], F, DF, solver_params, spec)
        exitcodes[i] = e_i
        num_iter[i] = n_i
        result[i] = res_i

    return result, exitcodes, num_iter


MULTI_SOLVERS: dict[
    Literal["sequential", "parallel"],
    Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
] = {
    "sequential": sequential_solver,
    "parallel": parallel_solver,
}
"""Map of multi-solver functions, applying some solver either sequentially
or in parallel for vectorized input.

See also:

    - :func:`sequential_solver`
    - :func:`parallel_solver`

"""
