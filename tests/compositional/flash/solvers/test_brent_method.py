"""Module containing tests for the Brent method implemented in
``porepy.compositional.flash.solvers.brent``."""

from __future__ import annotations

from typing import Callable

import numba
import numba.typed
import numpy as np
import pytest

from porepy.compositional.flash.solvers import DEFAULT_BRENT_PARAMS, brent


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
def test_brent(test_case: tuple[Callable[[float], float], float, float, float]) -> None:
    """Tests the Brent method with compiled test functions and its default parameters."""

    params = numba.typed.Dict.empty(
        key_type=numba.types.unicode_type, value_type=numba.types.float64
    )
    for k, v in DEFAULT_BRENT_PARAMS.items():
        params[str(k)] = float(v)

    func, root_target, a, b = test_case

    func_c = numba.njit("f8(f8)")(func)
    root, converged, iter = brent(func_c, a, b, params)

    assert converged == 0
    assert np.abs(root - root_target) < DEFAULT_BRENT_PARAMS["brent_tolerance"]
    assert iter < DEFAULT_BRENT_PARAMS["brent_max_iterations"]
