"""This module contains heuristic functions used in the composite subpackage."""
from __future__ import annotations

import porepy as pp

from porepy.numerics.ad.operator_functions import NumericType

from .composite_utils import safe_sum


def pseudocritical_temperature(z: list[NumericType], T_c: list[NumericType]) -> NumericType:
    """Calculates an approximation of the critical temperature of a mixture of an
    arbitrary number of components.

    The values in ``z_i`` and ``T_ci`` arr assumed to be properly ordered, i.e.
    ``z[i]`` and ``T_c[i]`` belong to component ``i``.

    Parameters:
        z: A list of fractional values.
        T_c: A list of critical temperatures.

    Returns:
        The sum of critical temperatures weighed with the fractional values.

    """
    return safe_sum([z_i * T_i for z_i, T_i in zip(z, T_c)])


def pseudocritical_pressute(z: list[NumericType], p_c: list[NumericType]) -> NumericType:
    """Calculates an approximation of the critical pressure of a mixture analogous to
    :func:`pseudocritical_temperature`.

    Parameters:
        z: A list of fractional values.
        p_c: A list of critical pressures.

    Returns:
        The sum of critical pressures weighed with the fractional values.

    """
    return safe_sum([z_i * T_i for z_i, T_i in zip(z, p_c)])


def K_val_Wilson(p: NumericType, p_c: NumericType, T: NumericType, T_c: NumericType, acentric_factor: NumericType) -> NumericType:
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
    K = (
        p_c
        / p
        * pp.ad.exp(
            5.37
            * (1 + acentric_factor)
            * (1 - T_c / T)
        )
        + 1.0e-12
    )
    # K = pp.ad.power(K, 1./3.)
    return K