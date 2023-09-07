"""This file contains values for solid parameters.

For now we provide parameter values for the following solids:
* Granite
* Basalt

The dictionary containing parameter values is obtained by e.g. pp.solid_values.granite.
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
    "lame_lambda": 7020826.0,  # [Pa]
    "permeability": 5.0e-18,  # [m^2]
    "porosity": 1.3e-2,  # [-]
    "shear_modulus": 14854722.0,  # [Pa]
    "specific_heat_capacity": 720.7,  # [J * kg^-1 * K^-1]
    "specific_storage": 4.74e-10,  # [Pa^-1]
    "thermal_conductivity": 3.1,  # [W * m^-1 * K^-1]
    "thermal_expansion": 9.66e-6,  # [K^-1]
}


"""
Basalt:
---------------

TODO: Add values and references.

"""

basalt = {
    "biot_coefficient": None,  # [-]
    "density": None,  # [kg * m^-3]
    "friction_coefficient": None,  # [-]
    "lame_lambda": None,  # [Pa]
    "permeability": None,  # [m^2]
    "porosity": None,  # [-]
    "shear_modulus": None,  # [Pa]
    "specific_heat_capacity": None,  # [J * kg^-1 * K^-1]
    "specific_storage": None,  # [Pa^-1]
    "thermal_conductivity": None,  # [W * m^-1 * K^-1]
    "thermal_expansion": None,  # [K^-1]
}
