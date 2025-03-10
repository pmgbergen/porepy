"""Sub-package containing a collection of numerical methods and solvers used in the
flash."""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np

__all__ = []

from . import _core
from ._armijo_line_search import DEFAULT_ARMIJO_LINE_SEARCH_PARAMS, armijo_line_search
from ._core import *
from .brent_method import DEFAULT_BRENT_PARAMS, brent
from .npipm_solver import DEFAULT_NPIPM_SOLVER_PARAMS, npipm

__all__ = [
    "armijo_line_search",
    "brent",
    "npipm",
    "DEFAULT_ARMIJO_LINE_SEARCH_PARAMS",
    "DEFAULT_BRENT_PARAMS",
    "DEFAULT_NPIPM_SOLVER_PARAMS",
    "SOLVERS",
    "DEFAULT_SOLVER_PARAMS",
]
__all__.extend(_core.__all__)


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
] = {"npipm": npipm}
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
