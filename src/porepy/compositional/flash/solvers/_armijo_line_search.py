"""Module containing a compiled implementation of the Armijo line search method.

Note:
    Armijo line search is functioning but numba does not allow defining functions
    inside other functions (say solver) and passing them to other numba functions.
    Hence, it is not used as of now. TODO WIP

"""

from __future__ import annotations

from typing import Callable, Literal, TypeAlias

import numba
import numpy as np

from ._core import FLASH_RESIDUAL_FUNCTION_TYPE, SOLVER_PARAMETERS_TYPE

__all__ = [
    "DEFAULT_ARMIJO_LINE_SEARCH_PARAMS",
    "armijo_line_search",
]


_ARMIJO_LINE_SEARCH_PARAMS_KEYS: TypeAlias = Literal[
    "armijo_rho", "armijo_kappa", "armijo_max_iterations"
]
"""Keys (names) for Armijo line search parameters."""


DEFAULT_ARMIJO_LINE_SEARCH_PARAMS: dict[_ARMIJO_LINE_SEARCH_PARAMS_KEYS, float] = {
    "armijo_rho": 0.99,
    "armijo_kappa": 0.4,
    "armijo_max_iterations": 50.0,
}
"""Default parameters for :func:`armijo_line_search`.

- ``'armijo_rho': 0.99`` initial step size factor for Armijo line search.
- ``'armijo_kappa': 0.5`` steepness of line for line search.
- ``'armijo_max_iterations': 50.`` maximal number of line search iterations.

"""


ARMIJO_LINE_SEARCH_SIGNATURE = numba.float64(
    numba.float64[:],
    numba.float64[:],
    FLASH_RESIDUAL_FUNCTION_TYPE,
    SOLVER_PARAMETERS_TYPE,
)
"""Numba-signature for the armijo line-search.

The line search function takes

1. An 1D array representing the current iterate ``x``,
2. An 1D array representing the current update ``dx``,
3. A residual function which is being minimized ``F(x) -> min``, and
4. A solver parameter dictionary (str, float).

It returns a float, which is to be used as the step-size for the next iterate
``x + a * dx``.

"""


@numba.njit(ARMIJO_LINE_SEARCH_SIGNATURE, cache=True)
def armijo_line_search(
    X0: np.ndarray,
    DX: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    params: dict[str, float],
) -> float:
    """Armijo line search, reducing the L2-norm of the potential ``F``.

    For more information on the numba types in the signature, see
    :data:`ARMIJO_LINE_SEARCH_SIGNATURE`.

    Parameters:
        X0: Previous iterate for the when solving ``F(X)=0``.
        DX: New update.
        F: A function taking an array and returning an array (1D).
        params: A dictionary of numerical parameters. For more information on required
            parameters, see :data:`DEFAULT_ARMIJO_LINE_SEARCH_PARAMS`.

    Returns:
        A scalar value ``alpha`` such that the L2-norm of ``F(F + alpha * DX)`` is
        smaller than ``F(X + DX)``.

    """
    rho = float(params["armijo_rho"])
    kappa = float(params["armijo_kappa"])
    max_iter = int(params["armijo_max_iterations"])

    f_0 = F(X0)

    # Initial potential and step size.
    pot_0 = np.sum(f_0 * f_0) / 2.0
    rho_i = rho

    # Start with 1 due to usage of power.
    for i in range(1, max_iter + 1):
        # Reduced step-size.
        rho_i = rho**i

        # Evaluating potential at update.
        try:
            f_i = F(X0 + rho_i * DX)
        except Exception:
            # NOTE Here we allow the residual evaluation to fail and skip the line
            # search step, as this might happen when dealing with non-smooth F.
            # By continuing the step size comes closer to the old iterate making the
            # line search more robust, but slowing the overall progress.
            continue

        pot_i = np.sum(f_i * f_i) / 2.0

        if pot_i <= (1 - 2 * kappa * rho_i) * pot_0:
            break

    return rho_i
