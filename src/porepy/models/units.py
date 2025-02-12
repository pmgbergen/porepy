"""
Utility class for conversion between different units.

This is part of a system for setting the units of measurement of a problem.
"""

from __future__ import annotations

from typing import Optional, TypeVar

import numpy as np

import porepy as pp

number = pp.number


NumericalType = TypeVar("NumericalType", pp.number, np.ndarray)
"""Numerical type variable, either a number (int, float) or a numpy array."""


class Units:
    """Units for material properties.

    This is a sketch of a class for scaling material properties. The idea is that the
    material properties should be stored in SI units, but that the user may want to
    specify them in other units. These are defined in init.

    Base units are attributes of the class, and can be accessed as e.g.
    my_material.length Derived units are properties computed from the base units, and
    can be accessed as e.g. my_material.Pa. This ensures consistency between the base
    and derived units while allowing reference to derived units in usage of the class.

    TODO: Consider whether this needs to be incorporated in TimeStepManager.

    Parameters:
        **kwargs: Dictionary of units. The keys are the name of the unit, and the
            values are the scaling factor. For example, if the user wants to specify
            length in kilometers, time in hours and substance amount in millimolar,
            the dictionary should be

            .. code-block:: Python

                {'m': 1e3, 'kg': 1e-3, 'mol': 1e-3}

            or, equivalently,

            .. code-block:: Python

                {
                    'm':pp.KILO * pp.METER,
                    'kg': pp.MILLI * pp.KILOGRAM,
                    'mol':pp.MILLI * pp.MOLE,
                }

            Permitted keys are:
                - ``m``: Length unit.
                - ``s``: Time unit.
                - ``kg``: Mass unit.
                - ``K``: Temperature unit.
                - ``mol``: Mole unit.
                - ``rad``: Angle unit.

    """

    m: number
    """Length unit, defaults to 1 m."""
    s: number
    """Time unit, defaults to 1 s."""
    kg: number
    """Mass unit, defaults to 1 kg."""
    K: number
    """Temperature unit, defaults to 1 K."""
    mol: number
    """Mole unit, defaults to 1 mol."""
    rad: number
    """Angle unit, defaults to 1 rad."""

    def __init__(self, **kwargs) -> None:
        """Initialize the units."""
        # Sanity check on input values
        for key, value in kwargs.items():
            if not isinstance(value, (float, int)):
                raise ValueError("Input values must be of type number.")
            if key not in ["m", "s", "kg", "K", "mol", "rad"]:
                # If for some reason the user wants to change the base units, this can
                # be done by assigning to the attributes directly or overwriting the
                # __init__. Since we do not recommend this, we do not allow it here.
                raise ValueError("Input keys must be valid base units.")

        # Set known base units
        self.m: number = kwargs.get("m", 1)
        """Length unit in meters."""
        self.s: number = kwargs.get("s", 1)
        """Time unit in seconds."""
        if not np.isclose(self.s, 1):
            raise NotImplementedError("Non-unitary time scaling is not implemented.")

        self.kg: number = kwargs.get("kg", 1)
        """Mass unit in kilograms."""
        self.K: number = kwargs.get("K", 1)
        """Temperature unit in Kelvin."""
        self.mol: number = kwargs.get("mol", 1)
        """Mole unit in moles."""
        self.rad: number = kwargs.get("rad", 1)
        """Angle unit in radians."""

    @property
    def Pa(self) -> number:
        """Pressure (or stress) unit, derived from kg, m and s."""
        return self.kg / (self.m * self.s**2)

    @property
    def J(self) -> number:
        """Energy unit, derived from m, kg and s."""
        return self.kg * self.m**2 / self.s**2

    @property
    def N(self) -> number:
        """Force unit, derived from m, kg and s."""
        return self.kg * self.m / self.s**2

    @property
    def W(self) -> number:
        """Power unit, derived from m, kg and s."""
        return self.kg * self.m**2 / self.s**3

    @property
    def degree(self) -> number:
        """Angle unit, derived from rad."""
        return self.rad * 180 / np.pi

    def convert_units(
        self,
        value: NumericalType,
        units: str,
        to_si: Optional[bool] = False,
    ) -> NumericalType:
        """Convert value between SI and user specified units.

        The method divides the value by the units as defined by the user. As an example,
        if the user has defined the unit for pressure to be 1 MPa, then a value of 1e8
        will be converted to 1e8 / 1e6 = 1e2. Conversely, if to_si is True, the value
        will be converted to SI units, i.e. a value of 1e-2 results in 1e-2 * 1e6 = 1e4.

        Parameters:
            value: Value to be converted. Either a number (int, float) or a numpy array.
            units: Units of ``value`` defined as a string in the form of
                ``unit1 * unit2 * unit3^-1``, e.g., ``"Pa*m^3/kg"``.

                Valid units are the attributes and properties of the :attr:`units`
                defined at instantiation.
                Valid operators are * and ^, including negative powers (e.g. m^-2). A
                dimensionless value can be specified by setting units to "", "1" or "-".
            to_si: ``default=False``

                If True, the value is converted from given ``units`` to SI units.
                If False, the value is assumed to be in SI units, and converted to the
                scale specified by this class.

        Returns:
            Value in the user specified units to be used in the simulation.
            Of same type as the argument ``value``.

        """
        # Make a copy of the value to avoid modifying the original.
        # This is not strictly necessary for scalars, but is done in case the method is
        # used for arrays.
        if isinstance(value, np.ndarray):
            value = value.copy()
        # Trim any spaces.
        units = units.replace(" ", "")
        if units in ["", "1", "-"]:
            return value
        # Traverse string specifying units, and convert to SI units.
        # The string is traversed by first splitting at *.
        # If the substring contains a ^, the substring is split again, and the first
        # element is raised to the power of the second.
        for sub_unit in units.split("*"):
            if "^" in sub_unit:
                sub_unit, power = sub_unit.split("^")
                factor = getattr(self, sub_unit) ** float(power)
            else:
                factor = getattr(self, sub_unit)
            if to_si:
                value *= factor
            else:
                value /= factor
        return value
