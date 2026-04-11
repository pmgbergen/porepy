"""Sub-package containing a collection of numerical methods and solvers used in the
flash."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from . import _core
from ._core import *
from .brent_method import DEFAULT_BRENT_PARAMS, brent
from .npipm_solver import DEFAULT_NPIPM_SOLVER_PARAMS, npipm

if TYPE_CHECKING:
    import numpy as np

    from ...utils import FlashSpec

__all__ = [
    "brent",
    "npipm",
    "DEFAULT_BRENT_PARAMS",
    "DEFAULT_NPIPM_SOLVER_PARAMS",
    "SOLVERS",
    "DEFAULT_SOLVER_PARAMS",
]
__all__.extend(_core.__all__)


SOLVERS: dict[
    str,
    Callable[
        [
            np.ndarray,
            Callable[[np.ndarray], np.ndarray],
            Callable[[np.ndarray], np.ndarray],
            dict[str, float],
            FlashSpec,
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

DEFAULT_SOLVER_PARAMS: dict[str, dict[Any, float]] = {
    "npipm": DEFAULT_NPIPM_SOLVER_PARAMS,
}
"""Collection of default solver parameters.

To be used if no parameters are provided by the user.

- ``'npipm'``: See
  :data:`~porepy.compositional.flash.solvers.npipm.DEFAULT_NPIPM_SOLVER_PARAMS`.

"""
