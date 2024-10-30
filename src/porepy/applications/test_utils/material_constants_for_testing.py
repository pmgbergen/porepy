from porepy.applications.material_values.solid_values import granite
from porepy.applications.material_values.fluid_values import water
"""Material constants used in the model tests."""



numerical_values_for_testing = {"characteristic_displacement": 0.2,  # [m]
                                }
"""
Water values have been extended for testing purposes

"""

extended_water_values_for_testing = water.copy()

extended_water_values_for_testing.update(
    {
        "normal_thermal_conductivity": 0.5975,  # [kg m^-3]
        "pressure": 101325.0,  # [Pa]
        "temperature": 293.15,  # [K]
    }
)
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
        "skin_factor": 37.0,  # [-]
        "temperature": 293.15,  # [K]
        "well_radius": 0.1,  # [m]
    }
)
for n in ["lame_lambda", "shear_modulus"]:
    extended_granite_values_for_testing[n] *= 1e-3  # Improve conditioning