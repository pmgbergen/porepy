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
* Miranda, M. M., Matos, C. R., Rodrigues, N. V., Pereira, A. J. S. C., & Costa, J. J.
(2019). Effect of temperature on the thermal conductivity of a granite with high heat
production from Central Portugal. Journal of Iberian Geology, 45(1), 147-161.
https://doi.org/10.1007/s41513-018-0096-9


Density, thermal expansion/contraction coefficient, porosity, shear modulus and lame
lambda are gathered from:
* Sousa, L. (2013). Petrophysical properties and durability of granites employed as
building stone: A comprehensive evaluation. Bulletin of Engineering Geology and the
Environment, 73. https://doi.org/10.1007/s10064-013-0553-9

The values are for a kind of granite in Portugal. Shear modulus and lame lambda is
computed from the P- and S-wave speeds in the paper.


Biot coefficient is gathered from:
* Selvadurai, P., Selvadurai, P. A., & Nejati, M. (2019). A multi-phasic approach for
estimating the Biot coefficient for Grimsel granite. Solid Earth, 10(6), 2001-2014.
https://doi.org/10.5194/se-10-2001-2019


Coefficient of friction is gathered from:
* Byerlee, J. D. (1967). Frictional characteristics of granite under high confining
pressure. Journal of Geophysical Research (1896-1977), 72(14), 3639-3648.
https://doi.org/https://doi.org/10.1029/JZ072i014p03639

Figure 23: Coefficient of friction is the slope of the normal stress/tangential stress
curve.


Permeability is gathered from:
* Petrini, C., Madonna, C., & Gerya, T. (2021). Inversion in the permeability evolution
of deforming Westerly granite near the brittle-ductile transition. Scientific Reports,
11(1), 24027. https://doi.org/10.1038/s41598-021-03435-0

The authors of the article present initial permeabilitiy for various westerly granite
samples in Table 1. All are of order 1e-18 and 1e-19.


Specific storage is gathered from:
* Hsieh, P. A., Neuman, S. P., Stiles, G. K., & Simpson, E. S. (1985). Field
Determination of the Three-Dimensional Hydraulic Conductivity Tensor of Anisotropic
Media: 2. Methodology and Application to Fractured Rocks. Water Resources Research,
21(11), 1667-1676. https://doi.org/https://doi.org/10.1029/WR021i011p01667

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
* Kristinsdóttir, L. H., Flóvenz, Ó. G., Árnason, K., Bruhn, D., Milsch, H.,
Spangenberg, E., & Kulenkampff, J. (2010). Electrical conductivity and P-wave velocity
in rock samples from high-temperature Icelandic geothermal fields. Geothermics, 39(1),
94-105. https://doi.org/https://doi.org/10.1016/j.geothermics.2009.12.001 


Thermal expansion coefficient and thermal conductivity: 
* https://www.britannica.com/science/rock-geology/Thermal-properties

The value was listed with the unit [cal s^-1 cm^-1 C^-1] and is converted to [W m^-1
K^-1].


Specific heat capacity:
* Xiaoqing, S., Ming, J., & Peiwen, X. (2018). ANALYSIS OF THE THERMOPHYSICAL PROPERTIES
AND INFLUENCING FACTORS OF VARIOUS ROCK TYPES FROM THE GUIZHOU PROVINCE. E3S Web Conf.,
53, 03059. https://doi.org/10.1051/e3sconf/20185303059 

TODO: Add Lame lambda and shear modulus from:
* Auriac, A., Sigmundsson, F., Hooper, A., Spaans, K. H., Björnsson, H., Pálsson, F.,
Pinel, V., & Feigl, K. L. (2014). InSAR observations and models of crustal deformation
due to a glacial surge in Iceland. Geophysical Journal International, 198(3), 1329-1341.
https://doi.org/10.1093/gji/ggu205 

Both parameter values are calculated from the Young's modulus and Poisson's ratio found
in the paper. 


TODO:
Add values for Biot coefficient, specific storage and coefficient of friction


"""

basalt = {
    "biot_coefficient": None,  # [-]
    "density": 2950.0,  # [kg * m^-3]
    "friction_coefficient": None,  # [-]
    "lame_lambda": None,  # [Pa]
    "permeability": 1e-16,  # [m^2]
    "porosity": 0.10,  # [-]
    "shear_modulus": None,  # [Pa]
    "specific_heat_capacity": 500.0,  # [J * kg^-1 * K^-1]
    "specific_storage": None,  # [Pa^-1]
    "thermal_conductivity": 1.6736,  # [W * m^-1 * K^-1]
    "thermal_expansion": 5.0e-6,  # [K^-1]
}
