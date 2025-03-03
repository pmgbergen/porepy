"""Module containing tests for the Brent method implemented in
``porepy.compositional.flash.solvers.brent``."""

from __future__ import annotations

from typing import Callable

import numba
import numpy as np
import pytest

from porepy.compositional.flash.solvers.brent import brent_method, brent_method_c


@pytest.mark.parametrize("compile_flag", [False, True])
@pytest.mark.parametrize(
    # Parametrization using a function, the root, and left and right interval points
    "test_case",
    [
        # Example from https://en.wikipedia.org/wiki/Brent%27s_method#Example
        (lambda x: (x + 3) * (x - 1) ** 2, -3, -4, 4 / 3),
        # example from https://de.wikipedia.org/wiki/Brent-Verfahren#Beispiel
        (lambda x: np.e ** (-x) * np.log(x), 1.0, 0.05, 1.7),
    ],
)
def test_brent(
    compile_flag: bool, test_case: tuple[Callable[[float], float], float, float, float]
) -> None:
    """Tests the Brent method, with and without pre-compilation of the tested function.

    The"""

    tol = 1e-16
    maxiter = 100

    func, root_target, a, b = test_case

    if compile_flag:
        func_c = numba.njit("f8(f8)")(func)
        root, converged, iter = brent_method_c(func_c, a, b, maxiter, tol)
    else:
        root, converged, iter = brent_method(func, a, b, maxiter, tol)

    assert converged == 0
    assert np.abs(root - root_target) < tol
    assert iter < maxiter
