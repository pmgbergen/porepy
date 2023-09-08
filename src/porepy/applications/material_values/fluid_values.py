"""This file contains values for fluid parameters.

For now we provide parameter values for the following fluids:
* Water (at 20 degrees Celsius)

The dictionary containing parameter values is obtained by e.g. pp.fluid_values.water.
They can be used in a simulation by passing `fluid = pp.FluidConstants(pp.fluid_values.
water)` as a material parameter on model initiation.


Water:
---------------

The values (except thermal conductivity) are gathered from:
* https://kdusling.github.io/teaching/Applied-Fluids/WaterProperties?T=20C
* Kell, G. S. (1975). Density, thermal expansivity, and compressibility of liquid water
from 0.deg. to 150.deg.. Correlations and tables for atmospheric pressure and saturation
reviewed and expressed on 1968 temperature scale. Journal of Chemical & Engineering
Data, 20(1), 97-105. https://doi.org/10.1021/je60064a005

The first reference is a calculator-like site based on the second reference.


Thermal conductivity is gathered from:
* Ramires, M. L. V., Nieto de Castro, C. A., Nagasaka, Y., Nagashima, A., Assael, M. J.,
& Wakeham, W. A. (1995). Standard Reference Data for the Thermal Conductivity of Water.
Journal of Physical and Chemical Reference Data, 24(3), 1377-1381.
https://doi.org/10.1063/1.555963

Note: The values provided in the paper for the thermal conductivity were for intervals
of 5 K. The value provided here is found by a linear approximation of the thermal
conductivity between 290K (16.85C) and 295K (21.85C).

"""

water = {
    "compressibility": 0.4559 * 1e-9,  # [Pa^-1], isentropic compressibility
    "density": 998.2,  # [kg m^-3]
    "specific_heat_capacity": 4182.0,  # [J kg^-1 K^-1], isochoric specific heat
    "thermal_conductivity": 0.5975,  # [kg m^-3]
    "thermal_expansion": 0.0002068,  # [K^-1]
    "viscosity": 0.001002,  # [Pa s], absolute viscosity
}
