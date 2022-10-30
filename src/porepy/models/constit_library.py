"""Library of constitutive equations.

Consists of three types of classes
    Units for scaling
    Materials for constants (Rock, Fluid). These are envisioned as components/attributes of
        model classes, see fluid_mass_balance.SolutionStrategyIncompressibleFlow.set_materials
    Constitutive equations on ad form. This will eventually become the most important part,
        from which a model is assembled based on mixin/inheritance.

See usage_example.py
"""
import warnings
from typing import Callable, Optional, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

number = Union[float, int]


class Units:
    """Units for material properties.

    This is a sketch of a class for scaling material properties. The idea is that the
    material properties should be stored in SI units, but that the user may want to
    specify them in other units. These are defined in init.
    Example:
        Running a simulation in km, days and  MPa is achieved by setting
        my_material = Units(m=1e3, s=86400, Pa=1e6)

    Base units are attributes of the class, and can be accessed as e.g. my_material.length
    Derived units are properties computed from the base units, and can be accessed as e.g.
    my_material.Pa. This ensures consistency between the base and derived units while allowing
    reference to derived units in usage of the class.

    TODO: Consider whether this needs to be incorporated in TimeStepManager.

    """

    m: number = 1 * pp.METER
    """Length unit, defaults to 1 m."""
    s: number = 1 * pp.SECOND
    """Time unit, defaults to 1 s."""
    kg: number = 1 * pp.KILOGRAM
    """Mass unit, defaults to 1 kg."""
    K: number = 1 * pp.KELVIN
    """Temperature unit, defaults to 1 K."""
    mol: number = 1
    """Mole unit, defaults to 1 mol."""

    def __init__(
        self,
        m: number = 1,
        s: number = 1,
        kg: number = 1,
        K: number = 1,
        mol: number = 1,
    ):
        """Initialize the units.

        Parameters:
            kwargs (dict): Dictionary of units. The keys are the name of the unit, and the
                values are the scaling factor. For example, if the user wants to specify
                length in kilometers, time in hours and substance amount in millimolar, the
                dictionary should be
                    dict(m=1e3, s=3600, mol=1e-3)
                or, equivalently,
                    dict(m=pp.KILO * pp.METER, s=pp.HOUR, mol=pp.MILLI * pp.MOLE)
        """
        # Check that all units are numbers and assign them as attributes
        for unit in ["m", "s", "kg", "K", "mol"]:
            val = locals()[unit]
            if not isinstance(val, number):
                raise ValueError(
                    f"All units must be numbers. Parameter {unit} is {type(val)}"
                )
            if val <= 0:
                warnings.warn(
                    "Expected positive value for " + unit + ", got " + str(val)
                )
            setattr(self, unit, val)

    @property
    def Pa(self):
        """Pressure (or stress) unit, derived from kg, m and s."""
        return self.kg / (self.m * self.s**2)

    @property
    def J(self):
        """Energy unit, derived from m, kg and s."""
        return self.kg * self.m**2 / self.s**2

    @property
    def N(self):
        """Force unit, derived from m, kg and s."""
        return self.kg * self.m / self.s**2

    @property
    def W(self):
        """Power unit, derived from m, kg and s."""
        return self.kg * self.m**2 / self.s**3


def ad_wrapper(
    vals: Union[number, np.ndarray],
    array: bool,
    size: Optional[int] = None,
    name: Optional[str] = None,
) -> Union[pp.ad.Array, pp.ad.Matrix]:
    """Create ad array or diagonal matrix.

    Utility method.

    Args:
        vals: Values to be wrapped. Floats are broadcast to an np array.
        array: Whether to return a matrix or vector.
        size: Size of the array or matrix. If not set, the size is inferred from vals.
        name: Name of ad object.

    Returns:

    """
    if type(vals) is not np.ndarray:
        assert size is not None, "Size must be set if vals is not an array"
        vals: np.ndarray = vals * np.ones(size)

    if array:
        return pp.ad.Array(vals, name)
    else:
        if size is None:
            size = vals.size
        matrix = sps.diags(vals, shape=(size, size))
        return pp.ad.Matrix(matrix, name)


class Material:
    """Sketch of abstract Material class. Functionality for now related to units.

    Modifications to parameter values should be done by subclassing. To set a different
    constant value, simply define a new class attribute with the same name. If a different
    value is needed for a specific subdomain or there is spatial heterogeneity internal to a
    subdomain, the method should be overridden. The latter is assumed to be most relevant for
    rocks.
    """

    def __init__(self, units) -> None:
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
    def units(self, units: Units) -> None:
        """Set units of the material.

        Args:
            units: Units object.
        """
        self._units = units

    def convert_units(self, value: number, units: str) -> number:
        """Convert value to SI units.

        The method divides the value by the units as defined by the user. As an example, if
        the user has defined the unit for pressure to be 1 MPa, then a value of 1e6 will be
        converted to 1e6 / 1e6 = 1 and a value of 1e8 will be converted to 1e8 / 1e6 = 1e2.

        Args:
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
        self, value: number, units: number, grids, grid_field="num_cells"
    ):
        """Convert value to SI units, and expand to all grids.

        Args:
            value: Value to be converted and expanded
            units: Units of value.
            grids: List of grids (subdomains or interfaces) where the property is defined.
            grid_field: Name of field in grid defining how to expand. Default is num_cells,
                other obvious choice is num_faces.

        Returns:
            Array of values, one for each cell or face in all grids.
        """
        size = sum([getattr(g, grid_field) for g in grids])
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

    def __init__(self, units):
        super().__init__(units)

    def density(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Density [kg/m^3].

        Args:
            subdomains: List of subdomains.

        Returns:
            Cell-wise density array.
        """
        return self.convert_and_expand(self.DENSITY, "kg * m^-3", subdomains)

    def thermal_expansion(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Thermal expansion coefficient [1/K].

        Args:
            subdomains:

        Returns:
            This return allows the implementation in this class to serve directly as the
            default constant fluid thermal expansion constitutive. This may require some care
            in ordering of mixins: More advanced constitutive relations must have priority
            over the material, e.g.

                class CombinedConstit(DensityFromPressure, UnitFluid, UnitRock):
                    pass

            See also fluid_density.


        """
        return self.convert_and_expand(self.THERMAL_EXPANSION, "K^-1", subdomains)

    # The below method needs rewriting after choosing between the above shown alternatives.
    def viscosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Viscosity [Pa s].

        Args:
            subdomains:

        Returns:
            Cell-wise viscosity array.
        """
        return self.convert_and_expand(self.VISCOSITY, "Pa*s", subdomains)


class UnitSolid(Material):
    """
    WIP. See UnitFluid.
    """

    THERMAL_EXPANSION: number = 1.0 / pp.KELVIN
    DENSITY: number = 1.0 * pp.KILOGRAM / pp.METER**3
    POROSITY: number = 0.2
    PERMEABILITY: number = 1.0 * pp.METER**2
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

        Args:
            subdomains: List of grids where the expansion coefficient is defined.

        Returns:
            Cell-wise thermal expansion coefficient.
        """
        return self.convert_and_expand(self.THERMAL_EXPANSION, "K^-1", subdomains)

    def normal_permeability(self, interfaces: list[pp.MortarGrid]):
        return self.convert_and_expand(self.NORMAL_PERMEABILITY, "m^2", interfaces)

    def porosity(self, subdomains: list[pp.Grid]):
        return self.convert_and_expand(self.POROSITY, "-", subdomains)

    def permeability(self, g: pp.Grid) -> np.ndarray:
        return self.convert_and_expand(self.PERMEABILITY, "m^2", [g])

    def shear_modulus(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Young's modulus [Pa].

        Args:
            subdomains: List of subdomains where the shear modulus is defined.

        Returns:
            Cell-wise shear modulus in Pascal.
        """
        return self.convert_and_expand(self.SHEAR_MODULUS, "Pa", subdomains)

    def lame_lambda(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Lame's first parameter [Pa].

        Args:
            subdomains: List of subdomains where the shear modulus is defined.

        Returns:
            Cell-wise Lame's first parameter in Pascal.
        """
        return self.convert_and_expand(self.LAME_LAMBDA, "Pa", subdomains)

    def fracture_gap(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Fracture gap [m].

        Args:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise fracture gap in meters.
        """
        return self.convert_and_expand(self.FRACTURE_GAP, "m", subdomains)

    def friction_coefficient(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Friction coefficient [-].

        Args:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise friction coefficient.
        """
        return self.convert_and_expand(self.FRICTION_COEFFICIENT, "-", subdomains)


"""
Below are some examples of Mixins which are low-level components of a set of constitutive
equations. First three different versions of fluid density, then one for permeability.

FIXME: Choose whether materials or the classes below are responsible for expanding to number
of cells. Probably safest to do that below in case of issues with vector values or cell/face
ambiguity.
"""


class ConstantFluidDensity:

    """Underforstått:

    def __init__(self, fluid: UnitFluid):
        self.fluid = ...

    eller tilsvarende. Se SolutionStrategiesIncompressibleFlow.
    """

    def fluid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Matrix:
        val = self.fluid.density(subdomains)
        num_cells = sum([sd.num_cells for sd in subdomains])
        return ad_wrapper(val, False, num_cells, "fluid_density")


class FluidDensityFromPressure:
    def fluid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")
        # Reference variables are defined in Variables class.
        dp = self.pressure(subdomains) - self.reference_pressure(self, subdomains)
        # I suggest using the fluid's constant density as the reference value. While not
        # explicit, this saves us from defining reference properties i hytt og pine.
        # We could consider letting this class inherit from ConstantDensity (and call super
        # to obtain reference value), but I don't see what the benefit would be.
        rho_ref = self.fluid.density(subdomains)
        rho = rho_ref * exp(dp / self.fluid.compressibility(subdomains))
        return rho


class FluidDensityFromPressureAndTemperature(FluidDensityFromPressure):
    """Extend previous case"""

    def fluid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        rho = super().fluid_density(subdomains)
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")
        dtemp = self.temperature(subdomains) - self.reference_temperature(
            self, subdomains
        )
        rho = rho * exp(-dtemp / self.fluid.THERMAL_EXPANSION)
        return rho


class ConstantViscosity:
    def viscosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return self.fluid.viscosity(subdomains)


class DarcyFlux:
    def pressure_trace(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Pressure on the subdomain boundaries.

        Args:
            subdomains: List of subdomains where the pressure is defined.

        Returns:
            Pressure on the subdomain boundaries. Parsing the operator will return a
            face-wise array
        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr = self.darcy_flux_discretization(subdomains)
        p: pp.ad.MergedVariable = self.pressure(subdomains)
        pressure_trace = (
            discr.bound_pressure_cell * p
            + discr.bound_pressure_face
            * projection.mortar_to_primary_int
            * self.interface_fluid_flux(interfaces)
            + discr.bound_pressure_face * self.bc_values_flow(subdomains)
            + discr.vector_source * self.vector_source(subdomains)
        )
        return pressure_trace

    def darcy_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Darcy flux.

        Args:
            subdomains: List of subdomains where the Darcy flux is defined.

        Returns:
            Face-wise Darcy flux in cubic meters per second.
        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: pp.ad.Discretization = self.darcy_flux_discretization(subdomains)
        flux: pp.ad.Operator = (
            discr.flux * self.pressure(subdomains)
            + discr.bound_flux
            * (
                self.bc_values_flow(subdomains)
                + projection.mortar_to_primary_int
                * self.interface_fluid_flux(interfaces)
            )
            + discr.vector_source * self.vector_source(subdomains)
        )
        flux.set_name("Fluid flux")
        return flux

    def darcy_flux_discretization(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Discretization:
        """
        .. note::
            The ad.Discretizations may be purged altogether. Their current function is very
            similar to the ad.Geometry in that both basically wrap numpy/scipy arrays in ad
            arrays and collect them in a block matrix. This similarity could possibly be
            exploited. Revisit at some point.

        Args:
            subdomains:

        Returns:

        """
        return pp.ad.MpfaAd(self.flow_discretization_parameter_key, subdomains)


class LinearElasticMechanicalStress:
    """Linear elastic stress tensor.

    To be used in mechanical problems, e.g. force balance.

    """

    def mechanical_stress(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Linear elastic mechanical stress."""
        for sd in subdomains:
            assert sd.dim == self.nd
        discr = pp.ad.MpsaAd(subdomains, nd=self.nd)
        interfaces = self.subdomains_to_interfaces(subdomains)
        bc = self.bc_values_mechanics(subdomains)
        proj = pp.ad.MortarProjections(subdomains, interfaces, nd=self.nd)
        stress = (
            discr.stress * self.displacement(subdomains)
            + discr.bound_stress * bc
            + discr.bound_stress
            * self.subdomain_projections(self.nd).face_restriction(subdomains)
            * proj.mortar_to_primary_avg
            * self.displacement(interfaces)
        )
        stress.set_name("mechanical_stress")
        return stress


# Foregriper litt her for å illustrere utvidelse til poromekanikk.
# Det blir
# PoroConstit(LinearElasticRock, PressureStress):
#    def stress(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
#        return self.pressure_stress(subdomains) + self.mechanical_stress(subdomains)
class PressureStress:
    """Stress tensor from pressure.

    To be used in poromechanical problems.
    """

    pressure: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Pressure variable. Should be defined in the class inheriting from this mixin."""
    reference_pressure: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Reference pressure. Should be defined in the class inheriting from this mixin."""

    def pressure_stress(self, subdomains):
        """Pressure contribution to stress tensor.

        Args:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Pressure stress operator.
        """
        for sd in subdomains:
            assert sd.dim == self.nd
        discr = pp.ad.BiotAd(self.mechanics_parameter_key, subdomains)
        stress: pp.ad.Operator = (
            discr.grad_p * self.pressure(subdomains)
            # The reference pressure is only defined on sd_primary, thus there is no need
            # for a subdomain projection.
            - discr.grad_p * self.reference_pressure(subdomains)
        )
        stress.set_name("pressure_stress")
        return stress


class LinearElasticRock(LinearElasticMechanicalStress):
    """Linear elastic properties of rock.

    Includes "primary" stiffness parameters (lame_lambda, shear_modulus) and "secondary"
    parameters (bulk_modulus, lame_mu, poisson_ratio). The latter are computed from the former.
    Also provides a method for computing the stiffness matrix as a FourthOrderTensor.
    """

    def shear_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Shear modulus [Pa].

        Args:
            subdomains: List of subdomains where the shear modulus is defined.

        Returns:
            Cell-wise shear modulus operator [Pa].
        """
        return ad_wrapper(
            self.rock.shear_modulus(subdomains), False, name="shear_modulus"
        )

    def lame_lambda(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Lame's first parameter [Pa].

        Args:
            subdomains: List of subdomains where the shear modulus is defined.

        Returns:
            Cell-wise Lame's first parameter operator [Pa].
        """
        return ad_wrapper(self.rock.lame_lambda(subdomains), False, name="lame_lambda")

    def youngs_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Young's modulus [Pa].

        Args:
            subdomains: List of subdomains where the Young's modulus is defined.

        Returns:
            Cell-wise Young's modulus in Pascal.
        """
        val = (
            self.rock.shear_modulus(subdomains)
            * (
                3 * self.rock.lame_lambda(subdomains)
                + 2 * self.rock.shear_modulus(subdomains)
            )
            / (self.rock.lame_lambda(subdomains) + self.rock.shear_modulus(subdomains))
        )
        return ad_wrapper(val, False, name="youngs_modulus")

    def stiffness_tensor(self, subdomain: pp.Grid) -> pp.FourthOrderTensor:
        """Stiffness tensor [Pa].

        Args:
            subdomain: Subdomain where the stiffness tensor is defined.

        Returns:
            Cell-wise stiffness tensor in SI units.
        """
        lmbda = self.rock.lame_lambda([subdomain])
        mu = self.rock.shear_modulus([subdomain])
        return pp.FourthOrderTensor(mu, lmbda)


class FracturedSolid:
    """Fractured rock properties.

    This class is intended for use with fracture deformation models.
    """

    def reference_gap(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Reference gap [m].

        Args:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise reference gap operator [m].
        """
        return ad_wrapper(self.rock.gap(subdomains), True, name="reference_gap")

    def friction_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Friction coefficient.

        Args:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise friction coefficient operator.
        """
        return ad_wrapper(
            self.rock.friction_coefficient(subdomains),
            False,
            name="friction_coefficient",
        )


class FrictionBound:
    """Friction bound for fracture deformation.

    This class is intended for use with fracture deformation models.
    """

    normal_component: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Operator extracting normal component of vector. Should be defined in class combined with  this mixin."""
    traction: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Traction variable. Should be defined in class combined with from this mixin."""
    friction_coefficient: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Friction coefficient. Should be defined in class combined with this mixin."""

    def friction_bound(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Friction bound [m].

        Args:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise friction bound operator [Pa].

        """
        t_n: pp.ad.Operator = self.normal_component(subdomains) * self.traction(
            subdomains
        )
        bound: pp.ad.Operator = (-1) * self.friction_coefficient(subdomains) * t_n
        bound.set_name("friction_bound")
        return bound


class ConstantPorousMedium:
    def permeability(self, subdomain: pp.Grid) -> pp.SecondOrderTensor:
        # This will be set as before (pp.PARAMETERS) since it is a discretization parameter
        # Hence not list[subdomains]
        # perm = pp.SecondOrderTensor(
        #    self.rock.permeability(subdomain) * np.ones(subdomain.num_cells)
        # )
        return self.rock.permeability(subdomain)

    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        poro = self.rock.porosity(subdomains)

        return poro
