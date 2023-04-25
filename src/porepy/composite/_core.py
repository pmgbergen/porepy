"""This private module contains core functionality and assumptions for the entire
composite subpackage.

Changes here should be done with much care.

"""
from __future__ import annotations

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

from .composite_utils import safe_sum

MPa_kJ_SCALE = 1e3

R_IDEAL: float = 8.31446261815324 * 1e-3
"""Universal molar gas constant.

| Math. Dimension:        scalar
| Phys. Dimension:        [kJ / K mol]

"""

P_REF: float = 0.000611657
"""The reference pressure for the composite module is set to the triple point pressure
of pure water.

This value must be used to calculate the reference state when dealing with thermodynamic
properties.

| Math. Dimension:      scalar
| Phys. Dimension:      [MPa]

"""

T_REF: float = 273.16
"""The reference temperature for the composite module is set to the triple point
temperature of pure water.

This value must be used to calculate the reference state when dealing with thermodynamic
properties.

| Math. Dimension:      scalar
| Phys. Dimension:      [K]

"""

V_REF: float = 1.0
"""The reference volume is set to 1.

Computations in porous media, where densities are usually
expressed as per Reference Element Volume, have to be adapted respectively.

| Math. Dimension:      scalar
| Phys. Dimension:      [m^3]

"""

RHO_REF: float = P_REF / (R_IDEAL * T_REF) / V_REF
"""The reference density is computed using the ideal gas law and the reference pressure,
reference temperature, reference volume and universal gas constant.

| Math. Dimension:      scalar
| Phys. Dimension:      [mol / m^3]

"""

U_REF: float = 0.0
"""The reference value for the specific internal energy.

The composite submodule assumes the specific internal energy of the ideal gas at given
reference pressure and temperature to be zero.

| Math. Dimension:      scalar
| Phys. Dimension:      [kJ / mol]

"""

H_REF: float = U_REF + P_REF / RHO_REF
"""The reference value for the specific enthalpy.

based on other reference values it holds:

H_REF = U_REF + P_REF / RHO_REF

| Math. Dimension:      scalar
| Phys. Dimension:      [kJ / mol]

"""

_heat_capacity_ratio: float = 8.0 / 6.0
"""Heat capacity ratio for ideal, triatomic gases."""

CP_REF: float = _heat_capacity_ratio / (_heat_capacity_ratio - 1) * R_IDEAL
"""The specific heat capacity at constant pressure for ideal water vapor.

Water is tri-atomic and hence

C_P = g / (g-1) * R

where g (heat capacity ratio) is set to 8/6 for triatomic molecules.
(`see here <https://en.wikipedia.org/wiki/Heat_capacity_ratio>`_)

| Math. Dimension:      scalar
| Phys. Dimension:      [kJ / K mol]

"""

CV_REF: float = 1.0 / (_heat_capacity_ratio - 1) * R_IDEAL
"""The specific heat capacity at constant volume for ideal water vapor.

Water is tri-atomic and hence

C_V = 1 / (g-1) * R

where g (heat capacity ratio) is set to 8/6 for triatomic molecules.
(`see here <https://en.wikipedia.org/wiki/Heat_capacity_ratio>`_)

| Math. Dimension:      scalar
| Phys. Dimension:      [kJ / K mol]

"""

COMPOSITIONAL_VARIABLE_SYMBOLS = {
    "pressure": "p_mix",
    "enthalpy": "h_mix",
    "temperature": "T_mix",
    "component_fraction": "z",
    "phase_fraction": "y",
    "phase_saturation": "s",
    "phase_composition": "x",
    "solute_fraction": "chi",
}
"""A dictionary mapping names of variables (key) to their symbol, which is used in the
composite framework.

Warning:
    When using the composite framework, it is important to **not** name any other
    variable using the symbols here.

"""


def _rr_pole(i: int, y: list[NumericType], K: list[list[NumericType]]) -> NumericType:
    """Calculates the i-th denominator in the Rachford-Rice equation.

    With :math:`n_c` components, :math:`n_p` phases and :math:`R` the reference phase,
    the i-th denominator is given by

    .. math::

        t_i(y) = 1 - (\\sum\\limits_{j\\neq R}(1 - K_{ij})y_j)

    Parameters:
        i: Index of component. Used to access values in ``K``.
        y: ``len=(n_p-1)``

            List of phase fractions, excluding the reference phase fraction
        K: ``shape=(n_p, n_c)``

            A matrix-like structure or nested list, containing the K-value for component
            ``i`` in phase ``j`` by ``K[j][i]``.

    Returns:
        The expression for the denominator.

    """
    t = [(1 - K[j][i]) * y[j] for j in range(len(y))]

    return 1 - safe_sum(t)


def rachford_rice_equation(
    j: int, z: list[NumericType], y: list[NumericType], K: list[list[NumericType]]
) -> NumericType:
    """Assembles and returns the residual of the j-th Rachford-Rice equations.

    With :math:`n_c` components, :math:`n_p` phases and :math:`R` the reference phase,
    the j-th equation is given by

    .. math::

        f_j(y) = \\sum\\limits_{i=1}^{n_c}
        \\frac{(1-K_{ij})z_i}{1 - \\sum\\limits_{j\\neq R}(1 - K_{ij})y_j}

    Parameters:
        j: Index of phase. Used to access values in ``y`` and ``K``.
        z: ``len=n_c``

            List of overall component fractions
        y: ``len=(n_p-1)``

            List of phase fractions, excluding the reference phase fraction
        K: ``shape=(n_p, n_c)``

            A matrix-like structure or nested list, containing the K-value for component
            ``i`` in phase ``j`` by ``K[j][i]``.

    Returns:
        The residual of the j-th Rachford-Rice equation.

    """
    assert len(y) >= 1, "No phase fractions given."
    assert len(z) >= 1, "No overall component fractions given."

    f = [(1 - K[j][i]) * z[i] / _rr_pole(i, y, K) for i in range(len(z))]

    return safe_sum(f)


def rachford_rice_potential(
    z: list[NumericType], y: list[NumericType], K: list[list[NumericType]]
) -> NumericType:
    """Calculates the potential according to [1] for the j-th Rachford-Rice equation.

    With :math:`n_c` components, :math:`n_p` phases and :math:`R` the reference phase,
    the potential is given by

    .. math::

        F = \\sum\\limits_{i} -(z_i ln(1 - (\\sum\\limits_{j\\neq R}(1 - K_{ij})y_j)))

    References:
        [1] `Okuno and Sepehrnoori (2010) <https://doi.org/10.2118/117752-PA>`_

    Parameters:
        z: ``len=n_c``

            List of overall component fractions
        y: ``len=(n_p-1)``

            List of phase fractions, excluding the reference phase fraction
        K: ``shape=(n_p, n_c)``

            A matrix-like structure or nested list, containing the K-value for component
            ``i`` in phase ``j`` by ``K[j][i]``.

    Returns:
        The value of the potential based on above formula.

    """
    F = [-z[i] * pp.ad.log(pp.ad.abs(_rr_pole(i, y, K))) for i in range(len(z))]
    return safe_sum(F)
