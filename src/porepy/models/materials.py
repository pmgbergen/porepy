"""
Storage classes for material constants.

Materials are storage classes for values of physical properties. They are typically used
when composing constitutive laws. A material is instantiated with a Units object, which
defines the units of the physical properties. The material can then be used to convert
from units set by the user (standard SI units) to those specified by the Units object,
which are the ones used internally in the simulation. The conversion hopefully reduces
problems with scaling/rounding errors and condition numbers.

"""
from typing import Optional

import porepy as pp

number = pp.number


class Material:
    """Material property container and conversion class.

    Modifications to parameter values should be done by subclassing. To set a different
    constant value, simply define a new class attribute with the same name. If a
    different value is needed for a specific subdomain or there is spatial heterogeneity
    internal to a subdomain, the method should be overridden. The latter is assumed to
    be most relevant for solids.
    """

    def __init__(self, constants: dict, units: Optional[pp.Units] = None) -> None:
        # Default units are SI
        self._units: pp.Units

        """Units of the material."""
        self._constants = constants

    @property
    def units(self) -> pp.Units:
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

    @property
    def constants(self) -> dict:
        """Constants of the material.

        Returns:
            Dictionary of constants.

        """
        return self._constants

    def convert_units(
        self, value: number, units: str, to_si: Optional[bool] = False
    ) -> number:
        """Convert value between SI and user specified units.

        The method divides the value by the units as defined by the user. As an example,
        if the user has defined the unit for pressure to be 1 MPa, then a value of  1e8
        will be converted to 1e8 / 1e6 = 1e2. Conversely, if to_si is True, the value
        will be converted to SI units, i.e. a value of 1e-2 results in 1e-2 * 1e6 = 1e4.

        Parameters:
            value: Value to be converted. units: Units of value defined as a string in
                the form of ``unit1 * unit2 * unit3^-1``, e.g., ``"Pa*m^3/kg"``.
                Valid units are the attributes and properties of the Units class. Valid
                operators are * and ^, including negative powers (e.g. m^-2). A
                dimensionless value can be specified by setting units to "", "1" or "-".
            to_si: If True, the value is converted to SI units. If False, the value is
                converted to the units specified by the user, which are the ones used in
                the

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

    Each constant (class attribute) typically corresponds to exactly one method which
    scales the value and broadcasts to relevant size, typically number of cells in the
    specified subdomains or interfaces.

    Note:
        Return types are discussed in fluid_density and fluid_thermal_expansion.

        Prefix fluid must be included if we decide for inheritance and not composition
        for the material classes.

    Parameters:
        constants (dict): Dictionary of constants. Only keys corresponding to a constant
            in the class will be used. The permissible keys are:
                - ``thermal_expansion``: Thermal expansion coefficient [1/K].
                - ``density``: Density [kg/m^3].
                - ``viscosity``: Viscosity [Pa s].
                - ``compressibility``: Compressibility [1/Pa].
            If not specified, default values are used.

    """

    def __init__(self, constants: Optional[dict[str, number]] = None):
        default_constants = {
            "thermal_expansion": 1,
            "density": 1,
            "viscosity": 1,
            "compressibility": 1,
        }
        if constants is not None:
            default_constants.update(constants)
        super().__init__(default_constants)

    def density(self) -> number:
        """Density [kg/m^3].

        Returns:
            Cell-wise density array.

        """
        return self.convert_units(self.constants["density"], "kg * m^-3")

    def thermal_expansion(self) -> number:
        """Thermal expansion coefficient [1/K].

        Returns:
            Cell-wise thermal expansion coefficient array.

        """
        return self.convert_units(self.constants["thermal_expansion"], "K^-1")

    def viscosity(self) -> number:
        """Viscosity [Pa s].

        Returns:
            Cell-wise viscosity array.

        """
        return self.convert_units(self.constants["viscosity"], "Pa*s")

    def compressibility(self) -> number:
        """Compressibility [1/Pa].

        Returns:
            Cell-wise compressibility array.

        """
        return self.convert_units(self.constants["compressibility"], "Pa^-1")


class UnitSolid(Material):
    """Solid material with unit values.

    Each constant (class attribute) typically corresponds to exactly one method which
    scales the value and broadcasts to relevant size, typically number of cells in the
    specified subdomains or interfaces.

    Parameters:
        constants (dict): Dictionary of constants. Only keys corresponding to a constant
            in the class will be used. The permissible keys are:
                - ``thermal_expansion``: Thermal expansion coefficient [1/K].
                - ``density``: Density [kg/m^3].
                - ``porosity``: Porosity [-].
                - ``permeability``: Permeability [m^2].
                - ``normal_permeability``: Normal permeability [m^2].
                - ``lame_lambda``: Lame parameter lambda [Pa].
                - ``shear_modulus``: Shear modulus [Pa].
                - ``friction_coefficient``: Friction coefficient [-].
                - ``fracture_gap``: Fracture gap [m].
                - ``dilation_angle``: Dilation angle [radians].

    """

    def __init__(self, constants: Optional[dict] = None):
        default_constants = {
            "thermal_expansion": 1,
            "density": 1,
            "porosity": 0.2,
            "permeability": 1,
            "normal_permeability": 1,
            "lame_lambda": 1,
            "shear_modulus": 1,
            "friction_coefficient": 1,
            "fracture_gap": 0,
            "dilation_angle": 0,
        }
        if constants is not None:
            default_constants.update(constants)
        super().__init__(default_constants)

    def density(self) -> number:
        """Density [kg/m^3].

        Returns:
            Cell-wise density array.

        """
        return self.convert_units(self.constants["density"], "kg * m^-3")

    def thermal_expansion(self) -> number:
        """Thermal expansion coefficient [1/K].

        Returns:
            Cell-wise thermal expansion coefficient.
        """
        return self.convert_units(self.constants["thermal_expansion"], "K^-1")

    def normal_permeability(self) -> number:
        """Normal permeability [m^2].

        Returns:
            Face-wise normal permeability.

        """
        return self.convert_units(self.constants["normal_permeability"], "m^2")

    def porosity(self) -> number:
        """Porosity [-].

        Returns:
            Cell-wise porosity.

        """
        return self.convert_units(self.constants["porosity"], "-")

    def permeability(self) -> number:
        """Permeability [m^2].

        Returns:
            Cell-wise permeability.

        """
        return self.convert_units(self.constants["permeability"], "m^2")

    def shear_modulus(self) -> number:
        """Young's modulus [Pa].

        Returns:
            Cell-wise shear modulus.

        """
        return self.convert_units(self.constants["shear_modulus"], "Pa")

    def lame_lambda(self) -> number:
        """Lame's first parameter [Pa].
        s
                Returns:
                    Cell-wise Lame's first parameter.
        """
        return self.convert_units(self.constants["lame_lambda"], "Pa")

    def gap(self) -> number:
        """Fracture gap [m].

        Returns:
            Cell-wise fracture gap.

        """
        return self.convert_units(self.constants["fracture_gap"], "m")

    def friction_coefficient(self) -> number:
        """Friction coefficient [-].

        Returns:
            Cell-wise friction coefficient.

        """
        return self.convert_units(self.constants["friction_coefficient"], "-")

    def dilation_angle(self) -> number:
        """Dilation angle [rad].

        Returns:
            Cell-wise dilation angle.

        """
        return self.convert_units(self.constants["dilation_angle"], "rad")
