"""This module contains utility functions and data for the composite submodule.
The module is built around the assumptions made here.

"""
from __future__ import annotations

import abc

import porepy as pp

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
    "CompositionalSingleton",
    "VARIABLE_SYMBOLS",
]

R_IDEAL: float = 0.00831446261815324
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

VARIABLE_SYMBOLS = {
    "pressure": "p",
    "enthalpy": "h",
    "temperature": "T",
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


class CompositionalSingleton(abc.ABCMeta):
    """Meta class for name- and AD-system-based singletons.

    This ensures that only a single object per AD System is instantiated with that name
    (and returned in successive instantiations).

    If name is not given as a keyword argument,
    the class name is used and the whole class becomes a singleton.

    The intended use is for classes which represent for example variables with specific
    names.
    This approach ensures a conflict-free usage of the central storage of values in the
    AD system.

    Note:
        As of now, the implications of having to use ``abc.ABCMeta`` are not clear.
        Python demands that custom meta-classes must be derived from meta classes used
        in other base classes.

        For now we demand that objects in the compositional framework are this type of
        singleton to avoid nonphysical conflicts like 2 times the same phase or
        component.
        This allows for multiple instantiations of components for phases or
        pseudo-components in various compounds,
        without having to worry about dependencies by reference and
        uniqueness of variables in a given model or AD system.

    Parameters:
        ad_system: A reference to respective AD system.
        name: ``default=None``

            Given name for an object. By default, the class name will be used.

    """

    # contains per AD system the singleton, using the given name as a unique identifier
    __ad_singletons: dict[pp.ad.ADSystem, dict[str, object]] = dict()

    def __call__(cls, ad_system: pp.ad.ADSystem, *args, **kwargs) -> object:
        # search for name, use class name if not given
        name = kwargs.get("name", str(cls.__name__))

        if ad_system in CompositionalSingleton.__ad_singletons:
            if name in CompositionalSingleton.__ad_singletons[ad_system]:
                # If there already is an object with this name instantiated
                # using this ad system, return it
                return CompositionalSingleton.__ad_singletons[ad_system][name]
        # prepare storage
        else:
            CompositionalSingleton.__ad_singletons.update({ad_system: dict()})

        # create a new object and store it
        new_instance = super(CompositionalSingleton, cls).__call__(
            ad_system, *args, **kwargs
        )
        CompositionalSingleton.__ad_singletons[ad_system].update({name: new_instance})
        # return new instance
        return new_instance
