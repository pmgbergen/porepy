"""Library of constitutive equations.

Consists of three types of classes
    Units for scaling
    Materials for constants (Rock, Fluid). These are envisioned as components/attributes of
        model classes, see fluid_mass_balance.SolutionStrategyIncompressibleFlow.set_materials
    Constitutive equations on ad form. This will eventually become the most important part,
        from which a model is assembled based on mixin/inheritance.

See usage_example.py
"""
from typing import Optional, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp


class SIUnits:
    """Sketch of unit class. Far from sure about format"""

    K: float = 1.0 * pp.KELVIN  # or temperature
    Pa: float = 1.0 * pp.PASCAL
    kg: float = 1.0 * pp.KILOGRAM
    m: float = 1.0 * pp.METER
    s: float = 1.0 * pp.SECOND

    pass


def ad_wrapper(
    vals: Union[float, np.ndarray],
    array: bool,
    size: Optional[int] = None,
    name: Optional[str] = None,
) -> Union[pp.ad.Array, pp.ad.Matrix]:
    """Create ad array or diagonal matrix.

    Utility method.
    Kommentar: Denne kan droppes. Nyttig hvis man ikke vet om en funksjon bør gi array eller
    matrix. Også praktisk å slippe å skrive ut (for matriser). Hvis den skal leve bør den
    flyttes og få nytt navn.

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
    """Sketch of abstract Material class. Functionality for now related to units"""

    def __init__(self, units) -> None:
        self._units = units

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, units: SIUnits):
        self._units = units

    @staticmethod
    def convert_units(value, units):
        """Convert value to SI units.

        The method divides the value by the units as defined by the user. As an example, if
        the user has defined the unit for pressure to be 1 MPa, then a value of 1e6 will be
        converted to 1e6 / 1e6 = 1 and a value of 1e8 will be converted to 1e8 / 1e6 = 1e2.

        Args:
            value: Value to be converted.
            units: Units of value.

        Returns:
            Value in SI units.

        """
        return value / units

    def convert_and_expand(self, value, units, grids, grid_field="num_cells"):
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
    (and possibly ad wraps) the value.

    For now, I have assumed materials will be components of the model.
    FIXME: Assign them somewhere
    The parameter consistently corresponding to this is (fluid) density.
    IMPLEMENTATION FOR OTHER PARAMETERS SHOULD NOT BE TRUSTED!

    .. note::
        Return types are discussed in fluid_density and fluid_thermal_expansion.

        Prefix fluid must be included if we decide for inheritance and not composition for
        the material classes.
    """

    THERMAL_EXPANSION: float = 1.0 / pp.KELVIN
    DENSITY: float = 1.0 * pp.KILOGRAM / pp.METER**3

    def __init__(self, units):
        super().__init__(units)

    def density(self, subdomains: list[pp.Grid]) -> Union[float, np.ndarray]:
        """Vi prøver først np.ndarray.

        Args:
            subdomains:

        Returns:
            Idea of np.ndarray (cell-wise in this case) is to allow for heterogeneity.
            Enforcing float may be safer.
            However, if we wrap it in an ad array, this will serve directly as the default
            constant fluid density constitutive equation. See fluid_thermal_expansion
        """
        units = self.units.kg / self.units.m**3
        return self.convert_and_expand(self.DENSITY, units, subdomains)

    def thermal_expansion(self, subdomains: list[pp.Grid]) -> pp.ad.Matrix:
        """Trolig ikke ad-matrix pga komposisjon, ikke mixin.

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
        val = self.convert_units(self.THERMAL_EXPANSION, 1 / self.units.K)
        num_cells = sum([sd.num_cells for sd in subdomains])
        return ad_wrapper(val, False, num_cells, "fluid_thermal_expansion")

    # The below method needs rewriting after choosing between the above shown alternatives.
    def viscosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # Se kommentar rett over.
        val = self.convert_units(1, self.units.m**2 / self.units.s)
        num_cells = sum([sd.num_cells for sd in subdomains])
        return ad_wrapper(val, False, num_cells, "viscosity")


class UnitRock(Material):
    """
    WIP. See UnitFluid.
    """

    THERMAL_EXPANSION: float = 1.0 / pp.KELVIN
    DENSITY: float = 1.0 * pp.KILOGRAM / pp.METER**3
    POROSITY: float = 0.2
    PERMEABILITY: float = 1.0 * pp.METER**2
    LAME_LAMBDA: float = 1.0 * pp.PASCAL
    SHEAR_MODULUS: float = 1.0 * pp.PASCAL

    def __init__(self, units):
        super().__init__(units)


    def density(self, subdomains: list[pp.Grid]):
        """Density [kg/m^3]."""
        units = self.units.kg / self.units.m**3
        return self.convert_and_expand(self.DENSITY, units, subdomains)

    def thermal_expansion(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Thermal expansion coefficient [1/K].

        Args:
            subdomains: List of grids where the expansion coefficient is defined.

        Returns:
            Cell-wise thermal expansion coefficient.
        """
        return self.convert_and_expand(self.THERMAL_EXPANSION, 1 / self.units.K, subdomains)

    def normal_permeability(self, interfaces: list[pp.MortarGrid]):
        num_cells = sum([sd.num_cells for sd in interfaces])

        return self.constit.ad_wrapper(
            self.NORMAL_PERMEABILITY, False, num_cells, "normal_permeability"
        )

    def porosity(self, subdomains: list[pp.Grid]):

        num_cells = sum([sd.num_cells for sd in subdomains])

        return ad_wrapper(self.POROSITY, False, num_cells, "porosity")

    def permeability(self, g: pp.Grid) -> np.ndarray:
        return self.convert_and_expand(self.PERMEABILITY, self.units.m**2, [g])

    def shear_modulus(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Young's modulus [Pa].

        Args:
            subdomains: List of subdomains where the shear modulus is defined.

        Returns:
            Cell-wise shear modulus in Pascal.
        """
        return self.convert_and_expand(self.SHEAR_MODULUS, self.units.Pa, subdomains)

    def lame_lambda(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Lame's first parameter [Pa].

        Args:
            subdomains: List of subdomains where the shear modulus is defined.

        Returns:
            Cell-wise Lame's first parameter in Pascal.
        """
        return self.convert_and_expand(self.LAME_LAMBDA, self.units.Pa, subdomains)


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


class LinearElasticRock:
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
        return ad_wrapper(self.rock.shear_modulus(subdomains), False, name="shear_modulus")

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
        val = self.rock.shear_modulus(subdomains) * (
            3 * self.rock.lame_lambda(subdomains) + 2 * self.rock.shear_modulus(subdomains)
        ) / (self.rock.lame_lambda(subdomains) + self.rock.shear_modulus(subdomains))
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


class ConstantRock:
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
