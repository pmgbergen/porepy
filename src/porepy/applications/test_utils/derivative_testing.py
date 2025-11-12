"""Module containing functionality for testing the implementation of derivatives of a
function."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

__all__ = [
    "get_EOC_taylor",
    "assert_order_at_least",
]


def get_EOC_taylor(
    func: Callable[[tuple[float, ...]], np.ndarray | float],
    dfunc: Callable[[tuple[float, ...]], np.ndarray | float],
    x0: np.ndarray,
    d: np.ndarray,
    h: np.ndarray,
    verbose: bool = False,
    tol: float = 1e-14,
) -> np.ndarray:
    """Estimate the order of convergence (EOC) of the derivative
    computation of ``func`` at point ``X0`` along direction ``D`` using Taylor
    expansion.

    The EOC is estimated by computing the error between the exact function value and
    the first-order Taylor approximation for a sequence of step sizes.

    The function and its derivative function are assumed to take floats as arguments.
    The number of floats is indicated by the size of ``X0``. ``dfunc`` should return
    a 2D array with shape (m, n) where m is the number of function outputs and n is
    the number of input arguments (size of ``X0``).

    Parameters:
        func: Function for which the derivative is computed.
        dfunc: Function computing the derivative of `func`.
        x0: Point at which the derivative is computed.
        d: Direction along which the derivative is computed.
        h: Array of step sizes to use for the Taylor expansion.
        verbose: If True, prints detailed information about errors and estimated orders.
        tol: Tolerance below which errors are considered zero
            (i.e., exact approximation).

    Returns:
        Estimated EOC values for each consecutive pair of step sizes.

    """
    # Norming direction for sensible scaling.
    d = d / np.linalg.norm(d)

    # Testing derivative along a random direction.
    errorlist = []
    for h_ in h:
        approx = func(*x0) + h_ * (dfunc(*x0) @ d)
        exact = func(*(x0 + h_ * d))
        error = float(np.linalg.norm(exact - approx))
        # If errors are small, their ratios can falsely indicate order loss due to
        # floating point arithmetics.
        if error < tol:
            error = 0.0
        errorlist.append(error)

    errors = np.array(errorlist)

    h_ratios = h[1:] / h[:-1]

    # Safe division for error_ratios
    error_ratios = np.full_like(errors[1:], np.nan)
    mask = errors[:-1] > tol
    error_ratios[mask] = errors[1:][mask] / errors[:-1][mask]

    # Compute orders safely
    orders = np.full_like(error_ratios, np.inf)
    finite_mask = np.isfinite(error_ratios) & (error_ratios > tol)
    orders[finite_mask] = np.log(error_ratios[finite_mask]) / np.log(
        h_ratios[finite_mask]
    )

    if verbose:
        normalized_errors = errors / h**2
        print(
            f"{'h':>10} {'error':>12} {'error/h^2':>12} {'error_ratio':>12} "
            f"{'est_order':>10}"
        )
        print("-" * 60)
        for i in range(len(h)):
            if i == 0:
                print(
                    f"{h[i]:>10.2e} {errors[i]:>12.2e} {normalized_errors[i]:>12.2e} "
                    f"{'-':>12} {'-':>10}"
                )
            else:
                print(
                    f"{h[i]:>10.2e} {errors[i]:>12.2e} {normalized_errors[i]:>12.2e} "
                    f"{error_ratios[i - 1]:>12.2e} {orders[i - 1]:>10.2f}"
                )

    return orders


def assert_order_at_least(
    orders: np.ndarray,
    expected_order: float,
    tol: float = 0.1,
    err_msg: str = "",
    asymptotic: int | None = None,
) -> None:
    """Asserts that the average of the estimated orders are at least the expected order
    minus a tolerance.

    If orders are negative or nan, an error is raised.
    Order values of + infinity are treated as an exact approximation and treated as the
    expected order.

    Parameters:
        orders: List of order values, error ratios divided by refinement ratios.
        expected_order: The value of the expected (average) order value.
        tol: Tolerance for expected order for numerical reasons.
        asymptotic: If given as an integer ``n``, checks that only the last ``n``
            values are not negative (error decreasing) and uses only them for order
            checks. This is to be used for problems which are only asymptotically
            convergent and increasing errors are expected for coarse refinements.

    """
    if isinstance(asymptotic, int):
        assert isinstance(asymptotic, int), "Require integer."
        orders = orders[-asymptotic:]

    if np.any(orders < 0):
        raise ValueError(f"Negative orders, method DIVERGENT: {err_msg}")
    if np.any(np.isnan(orders)):
        raise ValueError(f"Estimated orders contain NAN values: {err_msg}")

    # If order all inf, we have an exact approximation.
    if not np.all(np.isinf(orders)):
        # Treat infinities as expected order
        orders[np.isinf(orders)] = expected_order
        order_avg = np.mean(orders)
        assert np.all(order_avg >= expected_order - tol), (
            f"Expected all orders to be at least {expected_order - tol}, "
            f"but got {order_avg}: {err_msg}"
        )
