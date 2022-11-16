"""
Materials are storage classes for values of physical properties. They are typically used
when composing constitutive laws. A material is instantiated with a Units object, which
defines the units of the physical properties. The material can then be used to convert
from units set by the user (standard SI units) to those specified by the Units object, which
are the ones used internally in the simulation. The conversion hopefully reduces problems with
scaling/rounding errors and condition numbers.
"""
from typing import Optional

import numpy as np

import porepy as pp

number = pp.number


class Material:
    """Material property container and conversion class.

    Modifications to parameter values should be done by subclassing. To set a different
    constant value, simply define a new class attribute with the same name. If a different
    value is needed for a specific subdomain or there is spatial heterogeneity internal to a
    subdomain, the method should be overridden. The latter is assumed to be most relevant for
    solids.
    """

    def __init__(self, units: pp.Units) -> None:
        self._units = units
        """Units of the material."""

    @property
    def units(self):
        """Units of the material.

        Returns:
            Units object.

        """
        return self._units

    @units.setter
    def units(self, units: pp.Units) -> None:
        """Set units of the material.

        Parameters:
            units: Units object.

        """
        self._units = units

    def convert_units(self, value: number, units: str, to_si: Optional[bool]=False) -> number:
        """Convert value between SI and user specified units.

        The method divides the value by the units as defined by the user. As an example, if
        the user has defined the unit for pressure to be 1 MPa, then a value of  1e8 will be
        converted to 1e8 / 1e6 = 1e2. Conversely, if to_si is True, the value will be converted
        to SI units, i.e. a value of 1e-2 results in 1e-2 * 1e6 = 1e4.

        Parameters:
            value: Value to be converted.
            units: Units of value defined as a string in the form of "unit1*unit2/unit3",
                e.g., "Pa*m^3/kg". Valid units are the attributes and properties of the Units
                class. Valid operators are * and ^, including negative powers (e.g. m^-2).
                A dimensionless value can be specified by setting units to "", "1" or "-".
            to_si: If True, the value is converted to SI units. If False, the value is
                converted to the units specified by the user, which are the ones used in the


        Returns:
            Value in the user specified units to be used in the simulation.

        """
        # Trim any spaces
        units = units.replace(" ", "")
        if units in ["", "1", "-"]:
            return value
        # Traverse string specifying units, and convert to SI units
        # The string is traversed by first splitting at *.
        # If the substring contains a ^, the substring is split again, and the first
        # element is raised to the power of the second.
        for sub_unit in units.split("*"):
            if "^" in sub_unit:
                sub_unit, power = sub_unit.split("^")
                factor = getattr(self._units, sub_unit) ** float(power)
            else:
                factor = getattr(self._units, sub_unit)
            if to_si:
                value *= factor
            else:
                value /= factor
        return value


class UnitFluid(Material):
    """
    Class giving scaled values of fluid parameters.
    Each constant (class attribute) typically corresponds to exactly one method which scales
    the value and broadcasts to relevant size, typically number of cells in the specified
    subdomains or interfaces.


    Note:
        Return types are discussed in fluid_density and fluid_thermal_expansion.

        Prefix fluid must be included if we decide for inheritance and not composition for
        the material classes.
    """

    THERMAL_EXPANSION: number = 1.0 / pp.KELVIN
    DENSITY: number = 1.0 * pp.KILOGRAM / pp.METER**3
    VISCOSITY: number = 1.0 * pp.PASCAL * pp.SECOND
    COMPRESSIBILITY: number = 1.0 / pp.PASCAL

    def __init__(self, units: pp.Units):
        super().__init__(units)

    def density(self)-> number:
        """Density [kg/m^3].

        Returns:
            Cell-wise density array.

        """
        return self.convert_units(self.DENSITY, "kg * m^-3")

    def thermal_expansion(self)-> number:
        """Thermal expansion coefficient [1/K].

        Returns:
            Cell-wise thermal expansion coefficient array.

        """
        return self.convert_units(self.THERMAL_EXPANSION, "K^-1")

    def viscosity(self) -> pp.ad.Operator:
        """Viscosity [Pa s].

        Returns:
            Cell-wise viscosity array.

        """
        return self.convert_units(self.VISCOSITY, "Pa*s")

    def compressibility(self)-> number:
        """Compressibility [1/Pa].

        Returns:
            Cell-wise compressibility array.

        """
        return self.convert_units(self.COMPRESSIBILITY, "Pa^-1")


class UnitSolid(Material):
    """Solid material with unit values.

    See UnitFluid.

    """

    THERMAL_EXPANSION: number = 1.0 / pp.KELVIN
    DENSITY: number = 1.0 * pp.KILOGRAM / pp.METER**3
    POROSITY: number = 0.2
    PERMEABILITY: number = 1.0 * pp.METER**2
    NORMAL_PERMEABILITY: number = 1.0 * pp.METER**2
    LAME_LAMBDA: number = 1.0 * pp.PASCAL
    SHEAR_MODULUS: number = 1.0 * pp.PASCAL
    FRICTION_COEFFICIENT: number = 1.0
    FRACTURE_GAP: number = 0.0 * pp.METER
    DILATION_ANGLE: number = 0.0 * pp.RADIAN

    def __init__(self, units):
        super().__init__(units)

    def density(self):
        """Density [kg/m^3].

        Returns:
            Cell-wise density array.

        """
        return self.convert_units(self.DENSITY, "kg * m^-3")

    def thermal_expansion(self)-> number:
        """Thermal expansion coefficient [1/K].

        Returns:
            Cell-wise thermal expansion coefficient.
        """
        return self.convert_units(self.THERMAL_EXPANSION, "K^-1")

    def normal_permeability(self)-> number:
        """Normal permeability [m^2].

        Returns:
            Face-wise normal permeability.

        """
        return self.convert_units(self.NORMAL_PERMEABILITY, "m^2")

    def porosity(self)-> number:
        """Porosity [-].

        Returns:
            Cell-wise porosity.

        """
        return self.convert_units(self.POROSITY, "-")

    def permeability(self)-> number:
        """Permeability [m^2].

        Returns:
            Cell-wise permeability.

        """
        return self.convert_units(self.PERMEABILITY, "m^2")

    def shear_modulus(self)-> number:
        """Young's modulus [Pa].

        Returns:
            Cell-wise shear modulus.

        """
        return self.convert_units(self.SHEAR_MODULUS, "Pa")

    def lame_lambda(self)-> number:
        """Lame's first parameter [Pa].
s
        Returns:
            Cell-wise Lame's first parameter.
        """
        return self.convert_units(self.LAME_LAMBDA, "Pa")

    def gap(self)-> number:
        """Fracture gap [m].

        Returns:
            Cell-wise fracture gap.

        """
        return self.convert_units(self.FRACTURE_GAP, "m")

    def friction_coefficient(self)-> number:
        """Friction coefficient [-].

        Returns:
            Cell-wise friction coefficient.

        """
        return self.convert_units(self.FRICTION_COEFFICIENT, "-")

    def dilation_angle(self)-> number:
        """Dilation angle [rad].

        Returns:
            Cell-wise dilation angle.

        """
        return self.convert_units(self.DILATION_ANGLE, "rad")
