"""
Materials are storage classes for values of physical properties. They are typically used
when composing constitutive laws. A material is instantiated with a Units object, which
defines the units of the physical properties. The material can then be used to convert
from units set by the user (standard SI units) to those specified by the Units object, which
are the ones used internally in the simulation. The conversion hopefully reduces problems with
scaling/rounding errors and condition numbers.
"""
import numpy as np

import porepy as pp

number = pp.number


class Material:
    """Sketch of abstract Material class. Functionality for now related to units.

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

    def convert_units(self, value: number, units: str) -> number:
        """Convert value to SI units.

        The method divides the value by the units as defined by the user. As an example, if
        the user has defined the unit for pressure to be 1 MPa, then a value of 1e6 will be
        converted to 1e6 / 1e6 = 1 and a value of 1e8 will be converted to 1e8 / 1e6 = 1e2.

        Parameters:
            value: Value to be converted.
            units: Units of value defined as a string in the form of "unit1*unit2/unit3",
                e.g., "Pa*m^3/kg". Valid units are the attributes and properties of the Units
                class. Valid operators are * and ^, including negative powers (e.g. m^-2).
                A dimensionless value can be specified by setting units to "", "1" or "-".
        Returns:
            Value in SI units.

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
                value /= getattr(self._units, sub_unit) ** float(power)
            else:
                value /= getattr(self._units, sub_unit)
        return value

    def convert_and_expand(
        self,
        value: number,
        units: str,
        grids: list[pp.GridLike],
        grid_field: str = "num_cells",
        dim: int = 1,
    ):
        """Convert value to SI units, and expand to all grids.

        Parameters:
            value: Value to be converted and expanded
            units: Units of value.
            grids: List of grids (subdomains or interfaces) where the property is defined.
            grid_field: Name of field in grid defining how to expand. Default is num_cells,
                other obvious choice is num_faces.
            dim: Dimension of the property. Default is 1, but can be 2 or 3 for vectors.

        Returns:
            Array of values, one for each cell or face in all grids.
        """
        size = sum([getattr(g, grid_field) for g in grids]) * dim
        return self.convert_units(value, units) * np.ones(size)


class UnitFluid(Material):
    """
    Class giving scaled values of fluid parameters.
    Each constant (class attribute) typically corresponds to exactly one method which scales
    the value and broadcasts to relevant size, typically number of cells in the specified
    subdomains or interfaces.


    .. note::
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

    def density(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Density [kg/m^3].

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise density array.
        """
        return self.convert_and_expand(self.DENSITY, "kg * m^-3", subdomains)

    def thermal_expansion(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Thermal expansion coefficient [1/K].

        Parameters:
            subdomains:

        Returns:
            This return allows the implementation in this class to serve directly as the
            default constant fluid thermal expansion constitutive. This may require some care
            in ordering of mixins: More advanced constitutive relations must have priority
            over the material, e.g.

                class CombinedConstit(DensityFromPressure, UnitFluid, UnitSolid):
                    pass

        """
        return self.convert_and_expand(self.THERMAL_EXPANSION, "K^-1", subdomains)

    # The below method needs rewriting after choosing between the above shown alternatives.
    def viscosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Viscosity [Pa s].

        Parameters:
            subdomains:

        Returns:
            Cell-wise viscosity array.
        """
        return self.convert_and_expand(self.VISCOSITY, "Pa*s", subdomains)

    def compressibility(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Compressibility [1/Pa].

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise compressibility array.

        """
        return self.convert_and_expand(self.COMPRESSIBILITY, "Pa^-1", subdomains)


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
    FRACTURE_GAP: number = 1.0 * pp.METER

    def __init__(self, units):
        super().__init__(units)

    def density(self, subdomains: list[pp.Grid]):
        """Density [kg/m^3]."""
        return self.convert_and_expand(self.DENSITY, "kg * m^-3", subdomains)

    def thermal_expansion(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Thermal expansion coefficient [1/K].

        Parameters:
            subdomains: List of grids where the expansion coefficient is defined.

        Returns:
            Cell-wise thermal expansion coefficient.
        """
        return self.convert_and_expand(self.THERMAL_EXPANSION, "K^-1", subdomains)

    def normal_permeability(self, interfaces: list[pp.MortarGrid]) -> np.ndarray:
        """Normal permeability [m^2].

        Parameters:
            interfaces: List of interfaces where the normal permeability is defined.

        Returns:
            Face-wise normal permeability.

        """
        return self.convert_and_expand(self.NORMAL_PERMEABILITY, "m^2", interfaces)

    def porosity(self, subdomains: list[pp.Grid]) -> np.ndarray:
        return self.convert_and_expand(self.POROSITY, "-", subdomains)

    def permeability(self, sd: pp.Grid) -> np.ndarray:
        """Permeability [m^2].

        Parameters:
            sd: Subdomain where the permeability is defined.

        Returns:
            Cell-wise permeability.
        """
        return self.convert_and_expand(self.PERMEABILITY, "m^2", [sd])

    def shear_modulus(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Young's modulus [Pa].

        Parameters:
            subdomains: List of subdomains where the shear modulus is defined.

        Returns:
            Cell-wise shear modulus in Pascal.
        """
        return self.convert_and_expand(self.SHEAR_MODULUS, "Pa", subdomains)

    def lame_lambda(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Lame's first parameter [Pa].

        Parameters:
            subdomains: List of subdomains where the shear modulus is defined.

        Returns:
            Cell-wise Lame's first parameter in Pascal.
        """
        return self.convert_and_expand(self.LAME_LAMBDA, "Pa", subdomains)

    def fracture_gap(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Fracture gap [m].

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise fracture gap in meters.
        """
        return self.convert_and_expand(self.FRACTURE_GAP, "m", subdomains)

    def friction_coefficient(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Friction coefficient [-].

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise friction coefficient.
        """
        return self.convert_and_expand(self.FRICTION_COEFFICIENT, "-", subdomains)
