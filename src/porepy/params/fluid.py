""" Hard coded typical parameters that may be of use in simulations.

Contains standard values (e.g. found in Wikipedia) for density, thermal properties etc.

Note that thermal expansion coefficients are linear (m/mK) for rocks, but
volumetric (m^3/m^3K) for fluids.
"""
import numpy as np

import porepy as pp

module_sections = ["parameters"]


class UnitFluid:
    @pp.time_logger(sections=module_sections)
    def __init__(self, theta_ref=None):
        if theta_ref is None:
            self.theta_ref = 20 * (pp.CELSIUS)
        else:
            self.theta_ref = theta_ref

        self.COMPRESSIBILITY = 1 / pp.PASCAL
        self.BULK = 1 / self.COMPRESSIBILITY

    @pp.time_logger(sections=module_sections)
    def thermal_expansion(self, delta_theta):
        """ Units: m^3 / m^3 K, i.e. volumetric """
        return 1

    @pp.time_logger(sections=module_sections)
    def density(self, theta=None):  # theta in CELSIUS
        """ Units: kg / m^3 """
        return 1

    @pp.time_logger(sections=module_sections)
    def thermal_conductivity(self, theta=None):  # theta in CELSIUS
        """ Units: W / m K """
        return 1

    @pp.time_logger(sections=module_sections)
    def specific_heat_capacity(self, theta=None):  # theta in CELSIUS
        """ Units: J / kg K """
        return 1

    @pp.time_logger(sections=module_sections)
    def dynamic_viscosity(self, theta=None):  # theta in CELSIUS
        """Units: Pa s"""
        return 1

    @pp.time_logger(sections=module_sections)
    def hydrostatic_pressure(self, depth, theta=None):
        rho = self.density(theta)
        return rho * depth * pp.GRAVITY_ACCELERATION + pp.ATMOSPHERIC_PRESSURE


class Water:
    @pp.time_logger(sections=module_sections)
    def __init__(self, theta_ref=None):
        if theta_ref is None:
            self.theta_ref = 20 * (pp.CELSIUS)
        else:
            self.theta_ref = theta_ref

        self.COMPRESSIBILITY = 4e-10 / pp.PASCAL  # Moderate dependency on theta
        self.BULK = 1 / self.COMPRESSIBILITY

    @pp.time_logger(sections=module_sections)
    def thermal_expansion(self, delta_theta):
        """ Units: m^3 / m^3 K, i.e. volumetric """
        return (
            0.0002115
            + 1.32 * 1e-6 * delta_theta
            + 1.09 * 1e-8 * np.power(delta_theta, 2)
        )

    @pp.time_logger(sections=module_sections)
    def density(self, theta=None):  # theta in CELSIUS
        """ Units: kg / m^3 """
        if theta is None:
            theta = self.theta_ref
        theta_0 = 10 * (pp.CELSIUS)
        rho_0 = 999.8349 * (pp.KILOGRAM / pp.METER ** 3)
        return rho_0 / (1.0 + self.thermal_expansion(theta - theta_0))

    @pp.time_logger(sections=module_sections)
    def thermal_conductivity(self, theta=None):  # theta in CELSIUS
        """ Units: W / m K """
        if theta is None:
            theta = self.theta_ref
        return (
            0.56
            + 0.002 * theta
            - 1.01 * 1e-5 * np.power(theta, 2)
            + 6.71 * 1e-9 * np.power(theta, 3)
        )

    @pp.time_logger(sections=module_sections)
    def specific_heat_capacity(self, theta=None):  # theta in CELSIUS
        """ Units: J / kg K """
        if theta is None:
            theta = self.theta_ref
        return 4245 - 1.841 * theta

    @pp.time_logger(sections=module_sections)
    def dynamic_viscosity(self, theta=None):  # theta in CELSIUS
        """Units: Pa s"""
        if theta is None:
            theta = self.theta_ref
        theta = pp.CELSIUS_to_KELVIN(theta)
        mu_0 = 2.414 * 1e-5 * (pp.PASCAL * pp.SECOND)
        return mu_0 * np.power(10, 247.8 / (theta - 140))

    @pp.time_logger(sections=module_sections)
    def hydrostatic_pressure(self, depth, theta=None):
        rho = self.density(theta)
        return rho * depth * pp.GRAVITY_ACCELERATION + pp.ATMOSPHERIC_PRESSURE
