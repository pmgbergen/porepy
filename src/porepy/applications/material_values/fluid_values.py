"""This file contains values for fluid parameters.

For now we provide parameter values for the following fluids:

- Water (at 20 degrees Celsius)

They can be used in a simulation by passing e.g., ``fluid = pp.FluidComponent(
**pp.fluid_values.water)`` as a material parameter on model initiation.

"""

from typing import NotRequired, TypedDict


class FluidComponentDict(TypedDict):
    """A class to be used for type hinting dictionaries with thermodynamic constants for
    fluid components which are expected by various constitutive laws."""

    name: str
    """A string denoting the name."""
    compressibility: float
    """Isentropic compressibility in [Pa^-1]."""
    density: float
    """Mass density in [kg m^-3]."""
    specific_heat_capacity: float
    """Isochoric specific heat capacity in [J kg^-1 K^-1]."""
    thermal_conductivity: float
    """Thermal conductivity in [kg m^-3]."""
    thermal_expansion: float
    """Thermal expansion in [K^-1]."""
    viscosity: float
    """Absolute viscosity in [Pa s]."""
    normal_thermal_conductivity: NotRequired[float]
    """(Optional) Normal thermal conductivity on interfaces in [kg m^-3]."""


water: FluidComponentDict = {
    "name": "water",
    "compressibility": 4.559 * 1e-10,
    "density": 998.2,
    "specific_heat_capacity": 4182.0,
    "thermal_conductivity": 0.5975,
    "thermal_expansion": 2.068e-4,
    "viscosity": 1.002e-3,
}
"""The values (except thermal conductivity) are gathered from:

- https://kdusling.github.io/teaching/Applied-Fluids/WaterProperties?T=20C
- Kell, G. S. Density, thermal expansivity, and compressibility of liquid water from
  0.deg. to 150.deg.. Correlations and tables for atmospheric pressure and saturation
  reviewed and expressed on 1968 temperature scale. https://doi.org/10.1021/je60064a005

The first reference is a calculator-like site based on the second reference.

Thermal conductivity is gathered from:

- Ramires et al. Standard Reference Data for the Thermal Conductivity of Water.
  https://doi.org/10.1063/1.555963

Note:
    The values provided in the paper for the thermal conductivity were for intervals
    of 5 K. The value provided here is found by a linear approximation of the thermal
    conductivity between 290K (16.85C) and 295K (21.85C).

"""


extended_water_values_for_testing = water.copy()
"""Extended water values for testing which include a value for normal thermal
conductivity."""

extended_water_values_for_testing.update({"normal_thermal_conductivity": 0.5975})
