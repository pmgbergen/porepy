"""Collection of ideal fluid properties.

Contains examples for H2O, CO2, H2S and N2.

Heat capacity coefficients for the provided examples is taken from below reference.

References:
    [1] `de Nevers (2012)
    <https://onlinelibrary.wiley.com/doi/book/10.1002/9781118135341>`_ .

"""

import numba as nb
import numpy as np

from .._core import H_REF, NUMBA_FAST_MATH, R_U_MOL, T_REF, njit
from .ideal_fluid import IdealFluid

__all__ = [
    "IdealH2O",
    "IdealCO2",
    "IdealH2S",
    "IdealN2",
]

# region de Nevers fluids

cp_h2o: np.ndarray = np.array([3.47, 1.45e-3, 0.0, 0.121e5])
"""Heat capacity coefficients at constant water for ideal water."""


cp_co2: np.ndarray = np.array([5.457, 1.045e-3, 0.0, -1.57e5])
"""Heat capacity coefficients at constant pressure for ideal carbondioxide."""


cp_h2s: np.ndarray = np.array([3.931, 1.49e-3, 0.0, -0.232e5])
"""Heat capacity coefficients at constant pressure for ideal hydrogen sulfide."""


cp_n2: np.ndarray = np.array([3.28, 0.593e-3, 0.0, 0.04e5])
"""Heat capacity coefficients at constant pressure for ideal nitrogen."""


T_REF_deNevers: float = 298.15
"""Reference temperature used for the heat capacity integration in [1]."""


H_FORMATION_H2O_L_deNevers: float = -285.8e3
"""Enthalpy of formation of liquid water at the basis temperature of
:data:`T_REF_deNevers`."""

H_FORMATION_H2O_G_deNevers: float = 241.8e3
"""Enthalpy of formation of liquid water at the basis temperature of
:data:`T_REF_deNevers`."""


@njit(
    [
        nb.f8(nb.f8[:], nb.f8),
        nb.f8(nb.types.Array(nb.f8, 1, "C", readonly=True), nb.f8),
        nb.f8(nb.types.Array(nb.f8, 1, "C", readonly=False), nb.f8),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def deNevers_cp_of_T(cp: np.ndarray, T: float) -> float:
    """Interpolation formula for the heat capacity at constant pressure, based
    on some coefficients and the temperature.

    Note:
        Use only coefficients provided in the book [1].

    Parameters:
        cp: Heat capacity coefficients.
        T: Temperature.

    Returns:
        The heat capacity in [J / K / mol].

    """
    return R_U_MOL * (cp[0] + cp[1] * T + cp[2] * T**2 + cp[3] / T**2)


@njit(
    [
        nb.f8(nb.f8[:], nb.f8, nb.f8),
        nb.f8(nb.types.Array(nb.f8, 1, "C", readonly=True), nb.f8, nb.f8),
        nb.f8(nb.types.Array(nb.f8, 1, "C", readonly=False), nb.f8, nb.f8),
    ],
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def deNevers_Icp_of_T(cp: np.ndarray, T: float, T0: float) -> float:
    """Definite integral of the heat capacity at constant pressure.

    Note:
        Use only coefficients provided in the book [1].

    Parameters:
        cp: Heat capacity coefficients.
        T: Upper integral bound.
        T0: lower integral bound.

    Returns:
        The change in specific enthalpy from temperature ``T0`` to temperature ``T``.

    """
    return R_U_MOL * (
        cp[0] * (T - T0)
        + cp[1] * (T**2 - T0**2)
        + cp[2] * (T**3 - T0**3)
        - cp[3] * (1.0 / T - 1.0 / T0)
    )


DELTA_H_REF_TRANSFORM: float = H_REF - (
    H_FORMATION_H2O_L_deNevers + deNevers_Icp_of_T(cp_h2o, T_REF, T_REF_deNevers)
)
"""Change in enthalpy for reference temperature transform form :data:`T_REF_deNevers`
to the reference temperature in PorePy :data:`~porepy.compositional._core.H_REF`.

Must be added to the ideal enthalpy calculations for all species obtained from the
book by de Nevers.

"""


def h_id_h2o(T: float) -> float:
    """Temperature-dependent ideal enthalpy function for water with reference
    temperature set to PorePy's reference temperature."""
    # Use formation enthalpy for gas since this is the ideal gas.
    h_f = H_FORMATION_H2O_G_deNevers
    return h_f + deNevers_Icp_of_T(cp_h2o, T, T_REF_deNevers) + DELTA_H_REF_TRANSFORM


def dhdT_id_h2o(T: float) -> float:
    """Derivative of :func:`h_id_h2o` w.r.t. temperature."""
    return deNevers_cp_of_T(cp_h2o, T)


def h_id_co2(T: float) -> float:
    """Temperature-dependent ideal enthalpy function for CO2."""
    # Enthalpy of formation of gas, this is the stable phase at the reference point
    # of de Nevers.
    h_f = -393.5e3
    return h_f + deNevers_Icp_of_T(cp_co2, T, T_REF_deNevers) + DELTA_H_REF_TRANSFORM


def dhdT_id_co2(T: float) -> float:
    """Derivative of :func:`h_id_co2` w.r.t. temperature."""
    return deNevers_cp_of_T(cp_co2, T)


def h_id_h2s(T: float) -> float:
    """Temperature-dependent ideal enthalpy function for H2S."""
    h_f = -20.6e3
    return h_f + deNevers_Icp_of_T(cp_h2s, T, T_REF_deNevers) + DELTA_H_REF_TRANSFORM


def dhdT_id_h2s(T: float) -> float:
    """Derivative of :func:`h_id_h2s` w.r.t. temperature."""
    return deNevers_cp_of_T(cp_h2s, T)


def h_id_n2(T: float) -> float:
    """Temperature-dependent ideal enthalpy function for N2."""
    # Enthalpy of formation defined as zero for diatomic gases in standard state.
    h_f = 0.0
    return h_f + deNevers_Icp_of_T(cp_n2, T, T_REF_deNevers) + DELTA_H_REF_TRANSFORM


def dhdT_id_n2(T: float) -> float:
    """Derivative of :func:`h_id_n2` w.r.t. temperature."""
    return deNevers_cp_of_T(cp_n2, T)


IdealH2O = IdealFluid("H2O", h=h_id_h2o, dhdT=dhdT_id_h2o)
"""Ideal water properties."""


IdealCO2 = IdealFluid("CO2", h=h_id_h2o, dhdT=dhdT_id_h2o)
"""Ideal carbondioxide properties."""


IdealH2S = IdealFluid("H2S", h=h_id_h2o, dhdT=dhdT_id_h2o)
"""Ideal hydrogen sulfide properties."""


IdealN2 = IdealFluid("N2", h=h_id_h2o, dhdT=dhdT_id_h2o)
"""Ideal nitrogen properties."""

# endregion
