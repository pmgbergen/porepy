"""This file contains values for solid parameters.

The values provided here should be taken as representative, and not be mistaken to be
from one specific location or be of one specific rock sample. It was very difficult to
find one reference providing all the material values we needed, and therefore there are
several references per material.

For now we provide parameter values for the following solids:
* Granite
* Basalt

The dictionaries containing parameter values is obtained by e.g.
pp.solid_values.granite. They can be used in a simulation by passing `solid =
pp.SolidConstants(pp.solid_values. granite)` as a material parameter on model
initiation.
"""

"""
Granite:
---------------

Specific heat capacity, thermal conductivity and density are gathered from:
* Miranda et al. Effect of temperature on the thermal conductivity of a granite with
high heat production from Central Portugal. https://doi.org/10.1007/s41513-018-0096-9


Density, thermal expansion coefficient, porosity, shear modulus and lame lambda are
gathered from:
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
* Schultz, R. A. Limits on strength and deformation properties of jointed basaltic rock
masses.

Calculated from average Young's modulus (E = 78 GPa) and Poisson's ratio (v = 0.25)
provided in the abstract.


Specific heat capacity:
* Xiaoqing et al. Analysis of the thermophysical properties and influencing factors of
various rock types from the Guizhou province.
https://doi.org/10.1051/e3sconf/20185303059


Specific storage:
* Kuang et al. A review of specific storage in aquifers.
https://doi.org/10.1016/j.jhydrol.2019.124383.

Basalt: 1.30e-7 to 4.70e-6 [m^-1]

Values retrived from Table 1: Specific storage and hydraulic conductivity of different
types of unconsolidated deposits and rocks.

Converted from 1/m to 1/Pa using same method as described for granite above.

Coefficient of friction:
* Zhong et al. Experimental investigation on frictional properties of stressed basalt
fractures. https://doi.org/10.1016/j.jrmge.2022.12.020.

The measured basalt friction coefficients are in the range of 0.67-0.74.


Biot coefficient is gathered from:
* Detournay et al. Numerical technique to estimate the Biot coefficient for rock masses.
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4229729

Chose the average value of the estimate provided in table 6.

"""

basalt = {
    "biot_coefficient": 0.35,  # [-]
    "density": 2950.0,  # [kg * m^-3]
    "friction_coefficient": 0.7,  # [-]
    "lame_lambda": 31.2e9,  # [Pa]
    "permeability": 1e-16,  # [m^2]
    "porosity": 0.10,  # [-]
    "shear_modulus": 31.2e9,  # [Pa]
    "specific_heat_capacity": 603.0,  # [J * kg^-1 * K^-1]
    "specific_storage": 2.5e-10,  # [Pa^-1]
    "thermal_conductivity": 1.6736,  # [W * m^-1 * K^-1]
    "thermal_expansion": 5.0e-6,  # [K^-1]
}

"""
Granite values have been extended with nontrivial values for testing purposes. No
guarantees are given for the physical/geological correctness of these values.

References:
According to the MRST book, the skin factor is some value between -6 and 100.

"""

extended_granite_values_for_testing = granite.copy()
extended_granite_values_for_testing.update(
    {
        "dilation_angle": 0.1,  # [rad]
        "fracture_gap": 1e-3,  # [m]
        "fracture_normal_stiffness": 1.1e8,  # [Pa m^-1]
        "maximum_elastic_fracture_opening": 1e-3,  # [m]
        "normal_permeability": 5.0e-15,  # [m^2]
        "residual_aperture": 1e-3,  # [m]
        "skin_factor": 37,  # [-]
        "temperature": 293.15,  # [K]
        "well_radius": 0.1,  # [m]
    }
)
for n in ["lame_lambda", "shear_modulus"]:
    extended_granite_values_for_testing[n] *= 1e-3  # Improve conditioning
