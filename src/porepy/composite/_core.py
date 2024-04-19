"""This private module contains central assumptions and data for the entire
composite subpackage.

Changes here should be done with much care.

"""

from __future__ import annotations

__all__ = [
    "R_IDEAL",
    "P_REF",
    "T_REF",
    "NUMBA_CACHE",
    "COMPOSITIONAL_VARIABLE_SYMBOLS",
]


NUMBA_CACHE: bool = True
"""Flag to instruct the numba compiler to cache (!and use cached!) functions.

This might cause some confusion in the developing process due to some lack in numba's
caching functionality.
(Does not recognize changes in nested functions and hence does not trigger
re-compilation)

Use with care.

Note:
    Functions which do not use other numba-compiled functions are cached by default.
    This flag is for those who do use other functions.

See Also:
    https://numba.readthedocs.io/en/stable/user/jit.html#cache

"""

R_IDEAL: float = 8.31446261815324
"""Universal molar gas constant.

| Math. Dimension:        scalar
| Phys. Dimension:        [J / K mol]

"""

P_REF: float = 611.657
"""The reference pressure for the composite module is set to the triple point pressure
of pure water.

This value must be used to calculate the reference state when dealing with thermodynamic
properties.

| Math. Dimension:      scalar
| Phys. Dimension:      [Pa]

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
| Phys. Dimension:      [J / mol]

"""

H_REF: float = U_REF + P_REF / RHO_REF
"""The reference value for the specific enthalpy.

based on other reference values it holds:

H_REF = U_REF + P_REF / RHO_REF

| Math. Dimension:      scalar
| Phys. Dimension:      [J / mol]

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
| Phys. Dimension:      [J / K mol]

"""

CV_REF: float = 1.0 / (_heat_capacity_ratio - 1) * R_IDEAL
"""The specific heat capacity at constant volume for ideal water vapor.

Water is tri-atomic and hence

C_V = 1 / (g-1) * R

where g (heat capacity ratio) is set to 8/6 for triatomic molecules.
(`see here <https://en.wikipedia.org/wiki/Heat_capacity_ratio>`_)

| Math. Dimension:      scalar
| Phys. Dimension:      [J / K mol]

"""

COMPOSITIONAL_VARIABLE_SYMBOLS = {
    "pressure": "p",
    "enthalpy": "h",
    "temperature": "T",
    "volume": "v",
    "overall_fraction": "z",
    "phase_fraction": "y",
    "phase_saturation": "s",
    "phase_composition": "x",
    "solute_fraction": "c",
}
"""A dictionary mapping names of variables (key) to their symbol, which is used in the
composite framework.

Warning:
    When using the composite framework, it is important to **not** name any other
    variable using the symbols here.

"""
