"""Testing module for functionality regarding general solutions of cubic polynomials
dependent on coefficients and their derivatives."""

from __future__ import annotations

# import os
# os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np
import pytest

from porepy.compositional.peng_robinson import A_CRIT, B_CRIT, Z_CRIT
from porepy.compositional.peng_robinson.cubic_polynomial import (
    calculate_roots,
    d_one_root,
    d_three_roots,
    d_triple_root,
    d_two_roots,
    get_root_case,
    one_root,
    three_roots,
    triple_root,
    two_roots,
)


@pytest.mark.parametrize(
    ["coefficients", "solution", "root_case"],
    [
        ((-1, -1, -2), np.array([2.0]), 1),
        ((-3, -3, -1), np.array([np.cbrt(4) + np.cbrt(2) + 1]), 1),
        ((1, -2, -2), np.array([np.sqrt(2), -np.sqrt(2), -1]), 3),
        ((4, 2, -4), np.array([-1 + np.sqrt(3), -1 - np.sqrt(3), -2]), 3),
        (  # (x-1)*(x-2)**2
            (-5, 8, -4),
            np.array(
                [
                    1.0,
                    2.0,
                ]
            ),
            2,
        ),
        (  # (x-1)*(x+2)**2
            (3, 0, -4),
            np.array(
                [
                    1.0,
                    -2.0,
                ]
            ),
            2,
        ),
        (  # (x-sqrt(2))**2
            (-3 * np.sqrt(2), 6, -2 * np.sqrt(2)),
            np.array([np.sqrt(2)]),
            0,
        ),
        (  # Peng-Robinson EoS critical point
            (
                B_CRIT - 1,
                A_CRIT - 2.0 * B_CRIT - 3.0 * B_CRIT**2,
                B_CRIT**3 + B_CRIT**2 - A_CRIT * B_CRIT,
            ),
            np.array([Z_CRIT]),
            0,
        ),
    ],
)
def test_known_root_case_calculations(
    coefficients: tuple[float, float, float],
    solution: np.ndarray,
    root_case: int,
) -> None:
    """For given coefficients of the polynomial, tests if the root case is correctly
    deduced and then if the root is correctly calculated.

    The calculation is once done using the explicit function for the root case, and once
    using the general function. Both results should match the known solution.

    """

    eps = 1e-14
    c2, c1, c0 = coefficients

    calculated_root_case = get_root_case(c2, c1, c0, eps)

    # Test the calculated root case.
    assert calculated_root_case == root_case

    # Custom computations are supposed to be returned sorted as well (ascending).
    solution = np.sort(solution)

    vals: np.ndarray

    match root_case:
        case 0:
            assert solution.size == 1
            vals = triple_root(c2)
        case 1:
            assert solution.size == 1
            vals = one_root(c2, c1, c0)
        case 2:
            assert solution.size == 2
            vals = two_roots(c2, c1, c0)
        case 3:
            assert solution.size == 3
            vals = three_roots(c2, c1, c0)
        case _:
            assert False, "Faulty test"

    # Test computed root.
    np.testing.assert_allclose(vals, solution, atol=eps, rtol=0.0)

    # Simple test that it is indeed a rood.
    residual = vals**3 + c2 * vals**2 + c1 * vals + c0
    np.testing.assert_allclose(residual, 0.0, atol=eps, rtol=0.0)

    # Test that the call to the general function returns the same result.
    genvals = calculate_roots(c2, c1, c0, eps)
    np.testing.assert_allclose(genvals, solution, atol=eps, rtol=0.0)
