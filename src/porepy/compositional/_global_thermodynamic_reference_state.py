"""Private module defining PorePy's thermodynamic reference state.

Changes here should be done with much care.

The thermodynamic reference state for the entire package is the same as for IAPWS:
Internal energy and entropy of saturated liquid water at the triple point is zero.

.. rubric:: Consequences

For any thermodynamic model:

1. The species water must have zero internal energy and
   entropy in liquid state at its triple pint.
2. The ideal specific enthalpy of water vapor at the triple point should be equal to the
   latent heat at that point, plus the reference enthalpy for liquid water
   (which is small but should be accounted for in tolerance checks.)
3. The ideal internal energy of water vapor must be equal to the reference value
   of the internal energy :data:`U` plus the difference of latent heat and
   :data:`R_U` :math:`\\cdot` :data:`T`.

Thermodynamic models must fulfill these criteria for consistency before entering the
core code. Explicit testing is recommended. Departures from ideal states should be
accounted for with a given tolerance.

References:
    [1] `IAPWS 1997 (Revised) Industrial formulation
    <https://iapws.org/public/documents/UWTF-/IF97-Rev.pdf>`_

"""

from __future__ import annotations

__all__ = [
    "R_U",
    "MW_H2O",
    "P",
    "T",
    "V",
    "RHO_IG",
    "RHO_IG_MOL",
    "U",
    "S",
    "H",
    "CP",
    "CV",
]


R_U: float = 8.31446261815324
"""Universal molar gas constant in ``[J / K mol]``."""

MW_H2O: float = 18.01528e-3
"""Molar weight of water in ``[kg / mol]``"""

P: float = 611.657
"""The reference pressure is set to the triple point pressure of pure water in
``[Pa]``."""

T: float = 273.16
"""The reference temperature is set to the triple point temperature of pure water in
``[K]``."""

V_H2O: float = 1e-3
"""The reference specific volume is set to liquid water specific volume at the triple
point in ``[m^3 / kg]``. """

V_H2O_MOL: float = V_H2O * MW_H2O
"""Reference value for specific molar volume for water using :data:`V_H2O` and
:data:`MW_H2O`."""

RHO_IG_MOL: float = P / (R_U * T)
"""The reference value for ideal gas density is computed using the ideal gas law and
:data:`P`, :data:`T`, and :data:`R_U`, in ``[mol / m^3]``."""

RHO_IG_H2O: float = RHO_IG_MOL * MW_H2O
"""The massic reference ideal gas density of water is computed using
:data:`RHO_IG_MOL` and :data:`MW_H2O`."""

U: float = 0.0
"""The reference state for the specific internal energy ``[J / mol]``, at :data:`T`
and :data:`P` of liquid water. It is set to zero according to IAPWS standard."""

S: float = 0.0
"""The reference state for the specific entropy ``[J / K]``, at :data:`T`
and :data:`P` of liquid water. It is set to zero according to IAPWS standard."""

H: float = U + P * V_H2O_MOL
"""The reference value for the specific enthalpy ``[J / mol]``, using :data:`U`,
:data:`P` and :data:`V_H2O_MOL`."""

_heat_capacity_ratio: float = 8.0 / 6.0
"""Heat capacity ratio for ideal, triatomic gases like water.
Set to :math:`\\frac{8}{6}`."""

CP: float = _heat_capacity_ratio / (_heat_capacity_ratio - 1) * R_U
"""The specific heat capacity at constant pressure for ideal water vapor in
``[J / K mol]``.

It holds :math:`c_p = \\frac{\\gamma}{\\gamma - 1} R_{ideal}`, with
:math:`\\gamma = \\frac{8}{6}`.

See Also:

    https://en.wikipedia.org/wiki/Heat_capacity_ratio

"""

CV: float = 1.0 / (_heat_capacity_ratio - 1) * R_U
"""The specific heat capacity at constant volume for ideal water vapor in
``[J / K mol]``.

It holds :math:`c_v = \\frac{1}{\\gamma - 1} R_{ideal}`, with
:math:`\\gamma = \\frac{8}{6}`.

See Also:

    https://en.wikipedia.org/wiki/Heat_capacity_ratio

"""
