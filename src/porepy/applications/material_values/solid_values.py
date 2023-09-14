"""This file contains values for solid parameters.

For now we provide parameter values for the following solids:
* Granite
* Basalt

The dictionary containing parameter values is obtained by e.g. pp.solid_values.granite.
They can be used in a simulation by passing `solid = pp.SolidConstants(pp.solid_values.
granite)` as a material parameter on model initiation.
"""

"""
Granite:
---------------

Specific heat capacity, thermal conductivity and density are gathered from:
* Miranda et al. Effect of temperature on the thermal conductivity of a granite with
high heat production from Central Portugal. https://doi.org/10.1007/s41513-018-0096-9


Density, thermal expansion/contraction coefficient, porosity, shear modulus and lame
lambda are gathered from:
* Sousa, L. Petrophysical properties and durability of granites employed as
building stone: A comprehensive evaluation. https://doi.org/10.1007/s10064-013-0553-9

The values are for a kind of granite in Portugal. Shear modulus and lame lambda is
computed from the P- and S-wave speeds in the paper.


Biot coefficient is gathered from:
* Selvadurai et al. A multi-phasic approach for estimating the Biot coefficient for
Grimsel granite. https://doi.org/10.5194/se-10-2001-2019


Coefficient of friction is gathered from:
* Byerlee, J. D. Frictional characteristics of granite under high confining pressure.
https://doi.org/https://doi.org/10.1029/JZ072i014p03639

Figure 23: Coefficient of friction is the slope of the normal stress/tangential stress
curve.


Permeability is gathered from:
* Petrini et al. Inversion in the permeability evolution of deforming Westerly granite
near the brittle-ductile transition. https://doi.org/10.1038/s41598-021-03435-0

The authors of the article present initial permeabilitiy for various westerly granite
samples in Table 1. All are of order 1e-18 and 1e-19.


Specific storage is gathered from:
* Hsieh et al. Field Determination of the Three-Dimensional Hydraulic Conductivity
Tensor of Anisotropic Media: 2. Methodology and Application to Fractured Rocks.
https://doi.org/https://doi.org/10.1029/WR021i011p01667

Values are provided in 1/m. We convert to 1/Pa using chapter 4 in:
* https://books.gw-project.org/groundwater-storage-in-confined-aquifers

The average specific storage value from the paper:
S_s = 4.65e-6 m^-1 = 4.65e-6 / (rho_water * g) Pa^-1 = 4.74e-10 Pa^-1.

(Using rho_water = 1000 kg m^-3, g = 9.81 m s^-2)

"""

granite = {
    "biot_coefficient": 0.47,  # [-]
    "density": 2683.0,  # [kg * m^-3]
    "friction_coefficient": 0.6,  # [-]
    "lame_lambda": 7020826106,  # [Pa]
    "permeability": 5.0e-18,  # [m^2]
    "porosity": 1.3e-2,  # [-]
    "shear_modulus": 1.485472195e10,  # [Pa]
    "specific_heat_capacity": 720.7,  # [J * kg^-1 * K^-1]
    "specific_storage": 4.74e-10,  # [Pa^-1]
    "thermal_conductivity": 3.1,  # [W * m^-1 * K^-1]
    "thermal_expansion": 9.66e-6,  # [K^-1]
}


"""
Basalt:
---------------

Density and permeability:
* Sigurdsson, O., Guðmundsson, Á., Friðleifsson, G. Ó., Franzson, H., Gudlaugsson, S.,
& Stefánsson, V. (2000). Database on igneous rock properties in Icelandic geothermal
systems. Status and unexpected results. Proc. World Geothermal Congress.


Porosity:
* Kristinsdóttir et al. Electrical conductivity and P-wave velocity in rock samples from
high-temperature Icelandic geothermal fields.
https://doi.org/https://doi.org/10.1016/j.geothermics.2009.12.001


Thermal expansion coefficient and thermal conductivity:
* https://www.britannica.com/science/rock-geology/Thermal-properties

The value was listed with the unit [cal s^-1 cm^-1 C^-1] and is converted to [W m^-1
K^-1].


Shear modulus and Lame's first parameter:
* Auriac et al. InSAR observations and models of crustal deformation due to a glacial
surge in Iceland. https://doi.org/10.1093/gji/ggu205

Calculated from Young's modulus and Poisson's ratio found in figure 9. The best model
(white cross) predicts E = 46.4 GPa and v = 0.17.


Specific heat capacity:
* Xiaoqing et al. Analysis of the thermophysical properties and influencing factors of
various rock types from the Guizhou province.
https://doi.org/10.1051/e3sconf/20185303059


Specific storage:
* Kuang et al. A review of specific storage in aquifers.
https://doi.org/10.1016/j.jhydrol.2019.124383.

Basalt: 1.30e-7 to 4.70e-6 [m^-1]
Granite: 3.18e9 to 1.00e-3 [m^-1]

Values retrived from Table 1: Specific storage and hydraulic conductivity of different
types of unconsolidated deposits and rocks.

Converted from 1/m to 1/Pa using same method as described for granite above.


Coefficient of friction:
* Zhong et al. Experimental investigation on frictional properties of stressed basalt
fractures. https://doi.org/10.1016/j.jrmge.2022.12.020.

The measured basalt friction coefficients are in the range of 0.67-0.74.


Biot coefficient is calculated using the formula:
** alpha = 1 - K/Ks, where K and Ks are the bulk modulus of the rock and the matrix bulk
modulus of the rock respectively. We assume that K can be calculated from Young's
modulus and Poisson's ratio found in Auriac et al. and Ks is retrived from
* Qin, Han, & Zhao. Measurement of grain bulk modulus on sandstone samples from the
Norwegian Continental Shelf. https://doi.org/10.1029/2022JB024550

K = 23 GPa, using E = 46.4 GPa and v = 0.17
Ks= 50 GPa, sandstone with ~ 10% porosity

Note:
Biot coefficient and Ks for basalt is difficult to find and here the assumption
that Ks of basalt with 10% porosity is similar to sandstone with 10% porosity.

"""

basalt = {
    "biot_coefficient": 0.5,  # [-]
    "density": 2950,  # [kg * m^-3]
    "friction_coefficient": 0.7,  # [-]
    "lame_lambda": 4.3e9,  # [Pa]
    "permeability": 1e-16,  # [m^2]
    "porosity": 0.10,  # [-]
    "shear_modulus": 2.0e10,  # [Pa]
    "specific_heat_capacity": 500.0,  # [J * kg^-1 * K^-1]
    "specific_storage": 2.5e-10,  # [Pa^-1]
    "thermal_conductivity": 1.6736,  # [W * m^-1 * K^-1]
    "thermal_expansion": 5.0e-6,  # [K^-1]
}
