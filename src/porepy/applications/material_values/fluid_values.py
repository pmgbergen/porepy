"""This file contains values for fluid parameters.

For now we provide parameter values for the following fluids:
* Water (at 20 degrees Celsius)

The dictionary containing parameter values is obtained by e.g. pp.fluid_values.water.
They can be used in a simulation by passing `fluid = pp.FluidComponent(**pp.fluid_values.
water)` as a material parameter on model initiation.


Water:
---------------

The values (except thermal conductivity) are gathered from:
* https://kdusling.github.io/teaching/Applied-Fluids/WaterProperties?T=20C
* Kell, G. S. Density, thermal expansivity, and compressibility of liquid water from
0.deg. to 150.deg.. Correlations and tables for atmospheric pressure and saturation
reviewed and expressed on 1968 temperature scale. https://doi.org/10.1021/je60064a005

The first reference is a calculator-like site based on the second reference.


Thermal conductivity is gathered from:
* Ramires et al. Standard Reference Data for the Thermal Conductivity of Water.
https://doi.org/10.1063/1.555963

Note: The values provided in the paper for the thermal conductivity were for intervals
of 5 K. The value provided here is found by a linear approximation of the thermal
conductivity between 290K (16.85C) and 295K (21.85C).

"""

water = {
    "name": "water",
    "compressibility": 4.559 * 1e-10,  # [Pa^-1], isentropic compressibility
    "density": 998.2,  # [kg m^-3]
    "specific_heat_capacity": 4182.0,  # [J kg^-1 K^-1], isochoric specific heat
    "thermal_conductivity": 0.5975,  # [kg m^-3]
    "thermal_expansion": 2.068e-4,  # [K^-1]
    "viscosity": 1.002e-3,  # [Pa s], absolute viscosity
}


"""
Water values have been extended for testing purposes

"""

extended_water_values_for_testing = water.copy()

extended_water_values_for_testing.update(
    {
        "normal_thermal_conductivity": 0.5975,  # [kg m^-3]
    }
)
