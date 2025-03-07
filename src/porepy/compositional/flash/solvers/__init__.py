"""Sub-package containing a collection of numerical methods and solvers used in the
flash."""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np

from ._core import (
    FLASH_JACOBIAN_FUNCTION_TYPE,
    FLASH_RESIDUAL_FUNCTION_TYPE,
    MULTI_SOLVER,
    SOLVER_FUNCTION_TYPE,
    SOLVER_PARAMETERS_TYPE,
    parallel_solver,
    serial_solver,
)
from .brent import brent_method, brent_method_c
from .npipm import DEFAULT_NPIPM_SOLVER_PARAMS, npipm_solver

__all__ = [
    "brent_method",
    "brent_method_c",
    "npipm_solver",
    "serial_solver",
    "parallel_solver",
    "MULTI_SOLVER",
    "SOLVERS",
    "DEFAULT_SOLVER_PARAMS",
    "SOLVER_PARAMETERS_TYPE",
    "SOLVER_FUNCTION_TYPE",
    "FLASH_JACOBIAN_FUNCTION_TYPE",
    "FLASH_RESIDUAL_FUNCTION_TYPE",
]


SOLVERS: dict[
    Literal["npipm"],
    Callable[
        [
            np.ndarray,
            Callable[[np.ndarray], np.ndarray],
            Callable[[np.ndarray], np.ndarray],
            dict[str, float],
        ],
        tuple[np.ndarray, int, int],
    ],
] = {"npipm": npipm_solver}
"""Collection of available solvers.

For a more detailed description of the signature of a solver, see
:data:`~porepy.compositional.flash.solvers._core.SOLVER_FUNCTION_TYPE`.

Currently available:

- ``'npipm'``: A non-parametric interior point method with Newton solver, Armijo line
  search and heavy ball momentum.
  (see :mod:`~porepy.compositional.flash.solvers.npipm`)

"""

DEFAULT_SOLVER_PARAMS: dict[Literal["npipm"], dict[Any, float]] = {
    "npipm": DEFAULT_NPIPM_SOLVER_PARAMS,
}
"""Collection of default solver parameters.

To be used if no parameters are provided by the user.

- ``'npipm'``: See
  :data:`~porepy.compositional.flash.solvers.npipm.DEFAULT_NPIPM_SOLVER_PARAMS`.

"""
