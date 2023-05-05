"""This module contains heuristic functions used in the composite subpackage."""
from __future__ import annotations

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

__all__ = ["K_val_Wilson"]


def K_val_Wilson(
    p: NumericType,
    p_c: NumericType,
    T: NumericType,
    T_c: NumericType,
    acentric_factor: NumericType,
) -> NumericType:
    """Calculates an estimation of the K-value for a component defined by its
    critical properties and acentric factor.

    The estimation is based on the Wilson correlation, including an addition of
    ``1e-12`` to keep the value positive.

    Parameters:
        p: Pressure.
        p_c: Critical pressure of a component.
        T: Temperature.
        T_c: Critical temperature of a component.
        acentric_factor: The acentric factor of the component.

    Returns:
        The K-value estimation.

    """
    K = pp.ad.exp(5.37 * (1 + acentric_factor) * (1 - T_c / T)) * p_c / p + 1.0e-12
    # K = pp.ad.power(K, 1./3.)
    return K
