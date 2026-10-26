"""Error definitions used by the thermo function and gradient convergence studies."""

import numpy as np


def variation_normalized_l2_error(
    approx: np.ndarray, exact: np.ndarray
) -> tuple[float, int]:
    """L2 error normalised by the variation of the exact values about their mean.

    Points where the exact value is NaN are excluded and counted.

    Parameters:
        approx: Interpolated values.
        exact: Direct values.

    Returns:
        The pair ``(E_2, n_excluded)``.

    """
    finite = np.isfinite(exact) & np.isfinite(approx)
    n_excluded = int((~finite).sum())
    if finite.sum() < 2:
        return float("nan"), n_excluded

    a, e = approx[finite], exact[finite]
    spread = e - e.mean()
    denom = float(np.sqrt(np.sum(spread**2)))
    if denom == 0.0:
        # The field is constant over this set, so a relative measure is
        # meaningless; fall back to absolute and say so by returning the raw
        # difference rather than a ratio.
        return float(np.sqrt(np.sum((a - e) ** 2))), n_excluded
    return float(np.sqrt(np.sum((a - e) ** 2)) / denom), n_excluded


def observed_convergence_order(coarse: float, fine: float) -> float:
    """Order of convergence between two consecutive halvings.

    Parameters:
        coarse: Error at the coarser resolution.
        fine: Error at the finer resolution.

    Returns:
        The observed order, or NaN if either error is non-positive.

    """
    if not (coarse > 0 and fine > 0):
        return float("nan")
    return float(np.log(coarse / fine) / np.log(2.0))


def reference_normalized_l2_error(
    approx: np.ndarray, reference: np.ndarray
) -> tuple[float, int]:
    """Relative l2 error ``||approx - reference|| / ||reference||``.

    Parameters:
        approx: Approximate values.
        reference: Reference values.

    Returns:
        The pair ``(E_2, n_excluded)``; NaN for a zero reference norm or fewer
        than two finite samples.

    """
    finite = np.isfinite(approx) & np.isfinite(reference)
    excluded = int((~finite).sum())
    denominator = np.linalg.norm(reference[finite])
    if finite.sum() < 2 or denominator == 0:
        return float("nan"), excluded
    return (
        float(np.linalg.norm(approx[finite] - reference[finite]) / denominator),
        excluded,
    )
