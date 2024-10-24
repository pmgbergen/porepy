"""
Storage classes for material constants.

Materials are storage classes for values of physical properties. They are typically used
when composing constitutive laws. A material is instantiated with a Units object, which
defines the units of the physical properties. The material can then be used to convert
from units set by the user (standard SI units) to those specified by the Units object,
which are the ones used internally in the simulation. The conversion hopefully reduces
problems with scaling/rounding errors and condition numbers.

"""

from __future__ import annotations

from typing import Optional, Union, overload

import numpy as np

import porepy as pp

number = pp.number


class MaterialConstants:
    """Material property container and conversion class.

    Modifications to parameter values should be done by subclassing. To set a different
    constant value, simply define a new class attribute with the same name. If a
    different value is needed for a specific subdomain or there is spatial heterogeneity
    internal to a subdomain, the method should be overridden. The latter is assumed to
    be most relevant for solids.
    """

    def __init__(self, constants: dict) -> None:
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

    def set_units(self, units: pp.Units) -> None:
        """Set units of the material.

        Parameters:
            units: Units object.

        """
        # TODO: Should we use a @property setter here?
        self._units = units

    @property
    def constants(self) -> dict:
        """Constants of the material.

        Returns:
            Dictionary of constants.

        """
        return self._constants

    @overload
    def convert_units(
        self, value: number, units: str, to_si: Optional[bool] = False
    ) -> number: ...

    @overload
    def convert_units(
        self,
        value: np.ndarray,
        units: str,
        to_si: Optional[bool] = False,
    ) -> np.ndarray: ...

    def convert_units(
        self,
        value: Union[number, np.ndarray],
        units: str,
        to_si: Optional[bool] = False,
    ) -> Union[number, np.ndarray]:
        """Convert value between SI and user specified units.

        The method divides the value by the units as defined by the user. As an example,
        if the user has defined the unit for pressure to be 1 MPa, then a value of  1e8
        will be converted to 1e8 / 1e6 = 1e2. Conversely, if to_si is True, the value
        will be converted to SI units, i.e. a value of 1e-2 results in 1e-2 * 1e6 = 1e4.

        Parameters:
            value: Value to be converted.
            units: Units of value defined as a string in
                the form of ``unit1 * unit2 * unit3^-1``, e.g., ``"Pa*m^3/kg"``.
                Valid units are the attributes and properties of the Units class. Valid
                operators are * and ^, including negative powers (e.g. m^-2). A
                dimensionless value can be specified by setting units to "", "1" or "-".
            to_si: If True, the value is converted to SI units. If False, the value is
                converted to the units specified by the user, which are the ones used in
                the simulation.

        Returns:
            Value in the user specified units to be used in the simulation.

        """
        # Make a copy of the value to avoid modifying the original.
        # This is not strictly necessary for scalars, but is done in case the method is
        # used for arrays.
        if isinstance(value, np.ndarray):
            value = value.copy()
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

    def verify_constants(self, user_constants, default_constants):
        """Verify that the user has specified valid constants.

        Raises:
            ValueError: If the user has specified invalid constants.

        """
        if user_constants is None:
            return
        # Identify any keys in constants that are not in default_constants
        invalid_keys = set(user_constants.keys()) - set(default_constants.keys())
        if invalid_keys:
            raise ValueError(f"Invalid keys in constants: {invalid_keys}. ")


class FluidConstants(MaterialConstants):
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
            in the class will be used. If not specified, default values are used.

    """

    def __init__(self, constants: Optional[dict[str, number]] = None):
        default_constants = self.default_constants
        self.verify_constants(constants, default_constants)
        if constants is not None:
            default_constants.update(constants)
        super().__init__(default_constants)

    @property
    def default_constants(self) -> dict[str, number]:
        """Default constants of the material.

        Returns:
            Dictionary of constants.

        """
        # Default values, sorted alphabetically
        default_constants: dict[str, number] = {
            "compressibility": 0,
            "density": 1,
            "normal_thermal_conductivity": 1,
            "pressure": 0,
            "specific_heat_capacity": 1,
            "temperature": 0,
            "thermal_conductivity": 1,
            "thermal_expansion": 0,
            "viscosity": 1,
        }
        return default_constants

    def compressibility(self) -> number:
        """Compressibility [Pa^-1].

        Returns:
            Compressibility array in converted pressure units.

        """
        return self.convert_units(self.constants["compressibility"], "Pa^-1")

    def density(self) -> number:
        """Density [kg * m^-3].

        Returns:
            Density in converted mass and length units.

        """
        return self.convert_units(self.constants["density"], "kg * m^-3")

    def normal_thermal_conductivity(self) -> number:
        """Normal thermal conductivity [W * m^-1 * K^-1].

        Resides in fluid, not solid, because of the assumption of open fractures.

        Returns:
            Normal thermal conductivity in converted energy, length and temperature
            units.

        """
        return self.convert_units(
            self.constants["normal_thermal_conductivity"], "W * m^-1 * K^-1"
        )

    def pressure(self) -> number:
        """Pressure [Pa].

        Intended usage: Reference pressure.

        Returns:
            Pressure in converted pressure units.

        """
        return self.convert_units(self.constants["pressure"], "Pa")

    def specific_heat_capacity(self) -> number:
        """Specific heat [J * kg^-1 * K^-1].

        Returns:
            Specific heat in converted mass, temperature and time units.

        """
        return self.convert_units(
            self.constants["specific_heat_capacity"], "J * kg^-1 * K^-1"
        )

    def temperature(self) -> number:
        """Temperature [K].

        Intended usage: Reference temperature.

        Returns:
            Temperature in converted temperature units.

        """
        return self.convert_units(self.constants["temperature"], "K")

    def thermal_conductivity(self) -> number:
        """Thermal conductivity [W * m^-1 * K^-1].

        Returns:
            Thermal conductivity in converted mass, length and temperature units.

        """
        return self.convert_units(
            self.constants["thermal_conductivity"], "W * m^-1 * K^-1"
        )

    def thermal_expansion(self) -> number:
        """Thermal expansion coefficient [K^-1].

        Returns:
            Thermal expansion coefficient in converted temperature units.

        """
        return self.convert_units(self.constants["thermal_expansion"], "K^-1")

    def viscosity(self) -> number:
        """Viscosity [Pa * s].

        Returns:
            Viscosity array in converted pressure and time units.

        """
        return self.convert_units(self.constants["viscosity"], "Pa*s")


class SolidConstants(MaterialConstants):
    """Solid material with unit values.

    Each constant (class attribute) typically corresponds to exactly one method which
    scales the value and broadcasts to relevant size, typically number of cells in the
    specified subdomains or interfaces.

    Parameters:
        constants (dict): Dictionary of constants. Only keys corresponding to a constant
            in the class will be used. If not specified, default values are used, mostly
            0 or 1. See the soucre code for permissible keys and default values.
    """

    def __init__(self, constants: Optional[dict] = None):
        default_constants = self.default_constants
        self.verify_constants(constants, default_constants)
        if constants is not None:
            default_constants.update(constants)
        super().__init__(default_constants)

    @property
    def default_constants(self) -> dict[str, number]:
        """Default constants of the material.

        Returns:
            Dictionary of constants.

        """
        # Default values, sorted alphabetically
        # TODO: Numerical method parameters may find a better home soon.
        # TODO: Same goes for characteristic sizes.
        default_constants = {
            "biot_coefficient": 1,
            "characteristic_displacement": 1,
            "characteristic_contact_traction": 1,
            "density": 1,
            "dilation_angle": 0,
            "fracture_gap": 0,
            "fracture_normal_stiffness": 1,
            "fracture_tangential_stiffness": -1.0,
            "friction_coefficient": 1,
            "lame_lambda": 1,
            "maximum_elastic_fracture_opening": 0,
            "normal_permeability": 1,
            "permeability": 1,
            "porosity": 0.1,
            "residual_aperture": 0.1,
            "shear_modulus": 1,
            "skin_factor": 0,
            "specific_heat_capacity": 1,
            "specific_storage": 1,
            "temperature": 0,
            "thermal_conductivity": 1,
            "thermal_expansion": 0,
            "well_radius": 0.1
        }
        return default_constants

    def biot_coefficient(self) -> number:
        """Biot coefficient [-].

        Returns:
            Biot coefficient.

        """
        return self.constants["biot_coefficient"]

    def characteristic_displacement(self) -> number:
        """Characteristic displacement [m].

        Returns:
            Characteristic displacement in converted length units.

        """
        return self.convert_units(self.constants["characteristic_displacement"], "m")

    def characteristic_contact_traction(self) -> number:
        """Characteristic traction [Pa].

        Returns:
            Characteristic traction in converted pressure units.

        """
        return self.convert_units(
            self.constants["characteristic_contact_traction"], "Pa"
        )

    def density(self) -> number:
        """Density [kg * m^-3].

        Returns:
            Density in converted mass and length units.

        """
        return self.convert_units(self.constants["density"], "kg * m^-3")

    def thermal_expansion(self) -> number:
        """Thermal expansion coefficient [K^-1].

        Returns:
            Thermal expansion coefficient in converted temperature units.

        """
        return self.convert_units(self.constants["thermal_expansion"], "K^-1")

    def specific_heat_capacity(self) -> number:
        """Specific heat [J * kg^-1 * K^-1].

        Returns:
            Specific heat in converted energy, mass and temperature units.

        """
        return self.convert_units(
            self.constants["specific_heat_capacity"], "J * kg^-1 * K^-1"
        )

    def normal_permeability(self) -> number:
        """Normal permeability [m^2].

        Returns:
            Normal permeability in converted length units.

        """
        return self.convert_units(self.constants["normal_permeability"], "m^2")

    def thermal_conductivity(self) -> number:
        """Thermal conductivity [W * m^-1 * K^-1].

        Returns:
            Thermal conductivity in converted energy, length and temperature units.

        """
        return self.convert_units(
            self.constants["thermal_conductivity"], "W * m^-1 * K^-1"
        )

    def porosity(self) -> number:
        """Porosity [-].

        Returns:
            Porosity.

        """
        return self.convert_units(self.constants["porosity"], "-")

    def permeability(self) -> number:
        """Permeability [m^2].

        Returns:
            Permeability in converted length units.

        """
        return self.convert_units(self.constants["permeability"], "m^2")

    def residual_aperture(self) -> number:
        """Residual aperture [m].

        Returns:
            Residual aperture.

        """
        return self.convert_units(self.constants["residual_aperture"], "m")

    def shear_modulus(self) -> number:
        """Shear modulus [Pa].

        Returns:
            Shear modulus in converted pressure units.

        """
        return self.convert_units(self.constants["shear_modulus"], "Pa")

    def specific_storage(self) -> number:
        """Specific storage [Pa^-1].

        Returns:
            Specific storage in converted pressure units.

        """
        return self.convert_units(self.constants["specific_storage"], "Pa^-1")

    def lame_lambda(self) -> number:
        """Lame's first parameter [Pa].

        Returns:
            Lame's first parameter in converted pressure units.

        """
        return self.convert_units(self.constants["lame_lambda"], "Pa")

    def fracture_gap(self) -> number:
        """Fracture gap [m].

        Returns:
            Fracture gap in converted length units.

        """
        return self.convert_units(self.constants["fracture_gap"], "m")

    def friction_coefficient(self) -> number:
        """Friction coefficient [-].

        Returns:
            Friction coefficient.

        """
        return self.constants["friction_coefficient"]

    def dilation_angle(self) -> number:
        """Dilation angle [rad].

        Returns:
            Dilation angle in converted angle units.

        """
        return self.convert_units(self.constants["dilation_angle"], "rad")

    def skin_factor(self) -> number:
        """Skin factor [-].

        Returns:
            Skin factor.

        """
        return self.constants["skin_factor"]

    def temperature(self) -> number:
        """Temperature [K].

        Returns:
            Temperature in converted temperature units.

        """
        return self.convert_units(self.constants["temperature"], "K")

    def well_radius(self) -> number:
        """Well radius [m].

        Returns:
            Well radius in converted length units.

        """
        return self.convert_units(self.constants["well_radius"], "m")

    def fracture_normal_stiffness(self) -> number:
        """The normal stiffness of a fracture [Pa * m^-1].

        Intended use is in Barton-Bandis-type models for elastic fracture deformation.

        Returns:
            The fracture normal stiffness in converted units.

        """
        return self.convert_units(
            self.constants["fracture_normal_stiffness"], "Pa*m^-1"
        )

    def fracture_tangential_stiffness(self) -> number:
        """The tangential stiffness of a fracture [Pa * m^-1].

        Note: The current default value is -1.0, with the convention that negative
        values correspond to a fracture that does not deform elastically in the
        tangential direction.

        Returns:
            The fracture tangential stiffness in converted units.

        """
        return self.convert_units(
            self.constants["fracture_tangential_stiffness"], "Pa*m^-1"
        )

    def maximum_elastic_fracture_opening(self) -> number:
        """The maximum opening of a fracture [m].

        Intended use is in Barton-Bandis-type models for elastic fracture deformation.

        Returns:
            The maximal opening of a fracture.

        """
        return self.convert_units(
            self.constants["maximum_elastic_fracture_opening"], "m"
        )



class NumericalConstants(MaterialConstants):
    """TODO: Write description

   

    Parameters:
        constants (dict): Dictionary of constants. Only keys corresponding to a constant
            in the class will be used. If not specified, default values are used, mostly
            0 or 1. See the soucre code for permissible keys and default values.
    """
    def __init__(self, constants: Optional[dict] = None):
        default_constants = self.default_constants
        self.verify_constants(constants, default_constants)
        if constants is not None:
            default_constants.update(constants)
        super().__init__(default_constants)
        
    @property
    def default_constants(self) -> dict[str, number]:
        """Default constants of the material.

        Returns:
            Dictionary of constants.

        """
        # Default values, sorted alphabetically
        # TODO: Numerical method parameters may find a better home soon.
        default_constants = {
            "open_state_tolerance": 1e-5,  # Numerical method parameter
            "contact_mechanics_scaling": 1e-1,  # Numerical method parameter
        }
        return default_constants
    
    def contact_mechanics_scaling(self) -> number:
        """Safety scaling factor, making fractures softer than the matrix [-].

        Returns:
            The softening factor.

        """
        return self.constants["contact_mechanics_scaling"]
    
    def open_state_tolerance(self) -> number:
        """Tolerance parameter for the tangential characteristic contact mechanics [-].

        FIXME:
            Revisit the tolerance.

        Returns:
            The tolerance parameter.

        """
        return self.constants["open_state_tolerance"]