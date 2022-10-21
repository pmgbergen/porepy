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
    size: int,
    array: bool,
    name: Optional[str] = None,
) -> Union[pp.ad.Array, pp.ad.Matrix]:
    """Create ad array or diagonal matrix.

    Utility method.
    Kommentar: Denne kan droppes. Nyttig hvis man ikke vet om en funksjon bør gi array eller
    matrix. Også praktisk å slippe å skrive ut (for matriser). Hvis den skal leve bør den
    flyttes og få nytt navn.

    Args:
        vals: Values to be wrapped. Floats are broadcast to an np array.
        size: Size of the array or matrix.
        array: Whether to return a matrix or vector.
        name: Name of ad object.

    Returns:

    """
    if type(vals) is not np.ndarray:
        vals: np.ndarray = vals * np.ones(size)

    if array:
        return pp.ad.Array(vals, name)
    else:
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

    def convert_units(self, value, units):
        return value / units


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
        return self.convert_units(self.DENSITY, self.units.kg / self.units.m**3)

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
        return ad_wrapper(val, num_cells, False, "fluid_thermal_expansion")

    # The below method needs rewriting after choosing between the above shown alternatives.
    def viscosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # Se kommentar rett over.
        val = self.convert_units(1, self.units.m**2 / self.units.s)
        num_cells = sum([sd.num_cells for sd in subdomains])
        return ad_wrapper(val, num_cells, False, "viscosity")


class UnitRock(Material):
    """
    WIP. See UnitFluid.
    """

    THERMAL_EXPANSION: float = 1.0 / pp.KELVIN
    DENSITY: float = 1.0 * pp.KILOGRAM / pp.METER**3
    POROSITY: float = 0.2

    def rock_density(self, subdomains: list[pp.Grid]):
        return self.convert_units(self.DENSITY, self.units.kg / self.units.m**3)

    def rock_thermal_expansion(self, subdomains):
        return self.convert_units(self.THERMAL_EXPANSION, 1 / self.units.K)

    def normal_permeability(self, interfaces: list[pp.MortarGrid]):
        num_cells = sum([sd.num_cells for sd in interfaces])

        return self.constit.ad_wrapper(
            self.rock.NORMAL_PERMEABILITY, num_cells, False, "normal_permeability"
        )

    def porosity(self, subdomains: list[pp.Grid]):

        num_cells = sum([sd.num_cells for sd in subdomains])

        return ad_wrapper(self.POROSITY, num_cells, False, "porosity")

    def permeability(self, g: pp.Grid) -> float:
        return self.convert_units(1, self.units.m**2)


"""
Below are some examples of Mixins which are low-level components of a set of constitutive
equations. First three different versions of fluid density, then one for permeability.

FIXME: Choose whether materials or the classes below are responsible for expanding to number
of cells. Probably safest to do that below in case of issues with vector values or cell/face
ambiguity.
"""


class ConstantDensity:

    """Underforstått:

    def __init__(self, fluid: UnitFluid):
        self.fluid = ...

    eller tilsvarende. Se SolutionStrategiesIncompressibleFlow.
    """

    def fluid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Matrix:
        val = self.fluid.density(subdomains)
        num_cells = sum([sd.num_cells for sd in subdomains])
        return ad_wrapper(val, num_cells, False, "fluid_thermal_expansion")


class DensityFromPressure:
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


class DensityFromPressureAndTemperature(DensityFromPressure):
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
