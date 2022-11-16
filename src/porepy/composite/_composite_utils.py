""" Utility functions and data for the composite submodule. The module is built around the
assumptions made here.

The composite module works (for now) with the following units as base units:

| Pressure:     MPa (Mega Pascal)
| Temperature:  K (Kelvin)
| Mass:         mol (mol)
| Energy:       kJ (Kilo Joule)
| Volume:       m^3 (Cubic Meter)

For the reference state, an ideal tri-atomic gas (like water), with internal energy at the
triple point of water set to zero, was chosen.

All modelled phases and components are to be modelled with respect to this reference.

"""
from __future__ import annotations

__all__ = [
    "R_IDEAL",
    "P_REF",
    "T_REF",
    "V_REF",
    "RHO_REF",
    "U_REF",
    "H_REF",
    "CP_REF",
    "CV_REF",
]

R_IDEAL: float = 0.00831446261815324
"""Universal molar gas constant.

| Math. Dimension:        scalar
| Phys. Dimension:        [kJ / K mol]

"""

P_REF: float = 0.611657
"""The reference pressure for the composite module is set to the triple point pressure
of pure water.

This value must be used to calculate the reference state when dealing with thermodynamic
properties.

References:
    [1] http://iapws.org/relguide/IF97-Rev.html page 7

| Math. Dimension:      scalar
| Phys. Dimension:      [kPa]

"""

T_REF: float = 273.16
"""The reference temperature for the composite module is set to the triple point temperature
of pure water.

This value must be used to calculate the reference state when dealing with thermodynamic
properties.

References:
    [1] http://iapws.org/relguide/IF97-Rev.html page 7

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

# CP_REF: float = _heat_capacity_ratio / (_heat_capacity_ratio - 1) * R_IDEAL
CP_REF: float = 1
"""The specific heat capacity at constant pressure for ideal water vapor.

Water is tri-atomic and hence

C_P = g / (g-1) * R

where g (heat capacity ratio) is set to 8/6 for triatomic molecules.
(see https://en.wikipedia.org/wiki/Heat_capacity_ratio)

| Math. Dimension:      scalar
| Phys. Dimension:      [kJ / K mol]

"""

CV_REF: float = 1.0 / (_heat_capacity_ratio - 1) * R_IDEAL
"""The specific heat capacity at constant volume for ideal water vapor.

Water is tri-atomic and hence

C_V = 1 / (g-1) * R

where g (heat capacity ratio) is set to 8/6 for triatomic molecules.
(see https://en.wikipedia.org/wiki/Heat_capacity_ratio)

| Math. Dimension:      scalar
| Phys. Dimension:      [kJ / K mol]

"""
