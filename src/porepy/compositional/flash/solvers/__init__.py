"""Sub-package containing a collection of numerical methods and solvers used in the
flash."""

__all__ = [
    "brent_method",
    "brent_method_c",
    "npipm_solver",
]

from .brent import brent_method, brent_method_c
from .npipm import npipm_solver
