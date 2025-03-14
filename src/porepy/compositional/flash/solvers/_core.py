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

import os
from typing import Callable, Literal, TypeAlias

# NOTE import numba.typed like this to avoid importing the spurious py.typed file in the
# typed sub-package, which confuses mypy.
import numba
import numba.typed
import numpy as np

from ..._core import NUMBA_PARALLEL

_IS_JIT_DISABLED: bool = False
"""Environment flag checking whether numba JIT is enabled or not.

Used for typing alternatives in case it is not, such that the code remains functional.

"""

if "NUMBA_DISABLE_JIT" in os.environ:
    if os.environ["NUMBA_DISABLE_JIT"].lower() in ["1", "true"]:
        _IS_JIT_DISABLED = True


_typeof: Callable[..., TypeAlias]
"""Type inference function depending on whether numba is enabled or not.

If enabled, uses :obj:`numba.typeof`, otherwise the regular Python type.

"""

_cfunc: Callable[..., Callable[[Callable], Callable]]
"""C-type compiler for Callables, depending on whether numba is enabler or not.

If enabled, uses :obj:`numba.cfunc`, otherwise the identity.

"""

if _IS_JIT_DISABLED:
    _typeof = lambda x: type(x)

    def _cfunc(*args, **kwargs):
        return lambda x: x
else:
    _typeof = numba.typeof
    _cfunc = numba.cfunc


__all__ = [
    "GENERAL_SOLVER_PARAMS",
    "SOLVER_PARAMETERS_TYPE",
    "FLASH_RESIDUAL_FUNCTION_TYPE",
    "FLASH_JACOBIAN_FUNCTION_TYPE",
    "SOLVER_FUNCTION_SIGNATURE",
    "serial_solver",
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

SOLVER_PARAMETERS_TYPE = _typeof(_solver_parameters)
"""Numba-type definition of the solver parameter dictionary.

A solver parameter dictionary has strings as keys and ``float64`` as values.

Note:
    Numba does not allow multiple types in keys or strings (as of now).
    If a parameter is actually an integer, it must be converted to a float before
    setting it in the dictionary. Solvers must internally convert them back to integers.

"""


@_cfunc(numba.f8[:](numba.f8[:]), cache=True)
def flash_residual_template_func(x: np.ndarray) -> np.ndarray:
    """Template c-function for a flash residual function ``(f8[:]) -> f8[:]``.

    Used for automatic type-inferring.

    Parameters:
        x: Generic flash argument.

    Returns:
        The residual of an equilibrium system.

    """
    return x.copy()


FLASH_RESIDUAL_FUNCTION_TYPE = _typeof(flash_residual_template_func)
"""Numba type for a flash residual function, which takes a 1D array and returns a 1D
array (both of ``float64`` values).

Used to type cached, numba-compiled solvers.

"""


@_cfunc(numba.f8[:, :](numba.f8[:]), cache=True)
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


FLASH_JACOBIAN_FUNCTION_TYPE = _typeof(flash_jacobian_template_func)
"""Numba type for a flash Jacobian function, which takes a 1D array and returns a 2D
array (both of ``float64`` values).

Used to type cached, numba-compiled solvers.

"""


SOLVER_FUNCTION_SIGNATURE = numba.types.Tuple((numba.f8[:], numba.i4, numba.i4))(
    numba.f8[:],
    FLASH_RESIDUAL_FUNCTION_TYPE,
    FLASH_JACOBIAN_FUNCTION_TYPE,
    SOLVER_PARAMETERS_TYPE,
)
"""Numba signature for flash solvers.

To be used as the signature argument for :obj:`numba.njit` when compiling a solver.

See :data:`SOLVER_FUNCTION_TYPE` for more information on the signature.

"""


@_cfunc(
    numba.types.Tuple((numba.f8[:], numba.i4, numba.i4))(
        numba.f8[:],
        FLASH_RESIDUAL_FUNCTION_TYPE,
        FLASH_JACOBIAN_FUNCTION_TYPE,
        SOLVER_PARAMETERS_TYPE,
    ),
    cache=True,
)
def solver_template_func(
    x: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    solver_params: dict[str, float],
) -> tuple[np.ndarray, int, int]:
    """Template c-function for solvers.

    Parameters:
        x: Initial guess for the flash system to be solved.
        F: Flash residual function..
        DF: Flash Jacobian function.
        solver_params: Dictionary of relevant solver parameters (str-float pairs).

    Returns:
        The solution vector (1D array), a success code (int), and the number of
        iterations (int).

    """
    # Some operations are required for numba in order to create the type.
    return F(x) + DF(x) @ x, 1, 1


SOLVER_FUNCTION_TYPE = _typeof(solver_template_func)
"""Numba type for a flash solver, which takes

1. an initial guess (1D array),
2. the flash residual function ``(f8[:]) -> f8[:]``,
3. the flash Jacobian function ``(f8[:]) -> f8[:,;]``, and
4. solver parameters (``dict[str, float]``),

and returns 

1. the result (1D array),
2. a convergence code (int), and
3. the number of iterations required.

The convergence codes must be as follows:

- 0: successful solution procedure
- 1: maximal number of iterations reached
- 2: ``NAN`` or ``infty`` detected in update (aborted)
- 3: failure in the evaluation of the residual
- 4: failure in the evaluation of the Jacobian
- 5: Any other failure

"""


_multi_solver_signature = numba.types.Tuple(
    # NOTE: Since the return values are created internally, they are contiguous arrays.
    # Numba requires that information explicitly by using ::1 in the last dimension.
    # Also, for some unknown reasons the convergence codes are cast into int64 because
    # they have a default value of 5... casting it back did not help.
    (numba.f8[:, ::1], numba.i8[::1], numba.i4[::1])
)(
    numba.f8[:, :],
    FLASH_RESIDUAL_FUNCTION_TYPE,
    FLASH_JACOBIAN_FUNCTION_TYPE,
    SOLVER_FUNCTION_TYPE,
    SOLVER_PARAMETERS_TYPE,
)
"""Multi-solver signature for compiled serial or parallel application of solvers."""


@numba.njit(_multi_solver_signature, cache=True)
def serial_solver(
    X0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    solver: Callable[
        [
            np.ndarray,
            Callable[[np.ndarray], np.ndarray],
            Callable[[np.ndarray], np.ndarray],
            dict[str, float],
        ],
        tuple[np.ndarray, int, int],
    ],
    solver_params: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Serial application of a solver to vectorized input.

    The serialization is applied row-wise on ``X0``.

    Parameters:
        X0: 2D array, where each row is an initial guess for an individual problem.
        F: Flash residual function (see :data:`FLASH_RESIDUAL_FUNCTION_TYPE`).
        DF: Flash Jacobian function (see :data:`FLASH_JACOBIAN_FUNCTION_TYPE`).
        solver: Solver function (see :data:`SOLVER_FUNCTION_TYPE`).
        solver_params: Solver parameters passed to every problem.

    Returns:
        The results, convergence flags and number of iterations, vectorized where each
        row corresponds to a row in ``X0``.

    """

    # alocating return values
    n = X0.shape[0]
    result = np.zeros_like(X0)
    num_iter = np.zeros(n, dtype=np.int32)
    converged = np.ones(n, dtype=np.int32) * 5

    for i in range(n):
        try:
            res_i, conv_i, n_i = solver(X0[i], F, DF, solver_params)
        except Exception:
            converged[i] = 5
            num_iter[i] = -1
            result[i, :] = np.nan
        else:
            converged[i] = conv_i
            num_iter[i] = n_i
            result[i] = res_i

    return result, converged, num_iter


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
        ],
        tuple[np.ndarray, int, int],
    ],
    solver_params: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel application of a solver to vectorized input.

    Otherwise analogous to :func:`serial_solver`.

    To be used for a large quantity of problems, where parallelization outperforms
    serialization.

    Important:
        As of now, numba does not support ``try.. except`` inside a parallelized
        environment. If any flash problem crashes, they all do.
        Solvers should be robust in terms of errors when used inside the parallel
        application.

    """

    # alocating return values
    n = X0.shape[0]
    result = np.zeros_like(X0)
    num_iter = np.zeros(n, dtype=np.int32)
    converged = np.ones(n, dtype=np.int32) * 5

    for i in numba.prange(n):
        # NOTE Numba can as of now not parallelize if there is a try-except clause
        # try:
        res_i, conv_i, n_i = solver(X0[i], F, DF, solver_params)
        # except Exception:
        #     converged[i] = 5
        #     num_iter[i] = -1
        #     result[i, :] = np.nan
        # else:
        converged[i] = conv_i
        num_iter[i] = n_i
        result[i] = res_i

    return result, converged, num_iter


MULTI_SOLVERS: dict[
    Literal["serial", "parallel"],
    Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
] = {
    "serial": serial_solver,
    "parallel": parallel_solver,
}
"""Map of multi-solver functions, applying some solver either in serial
or in parallel for vectorized input.

See also:

    - :func:`serial_solver`
    - :func:`parallel_solver`

"""
