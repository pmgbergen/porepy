""" Hard coded typical parameters that may be of use in simulations.

Contains standard values (e.g. found in Wikipedia) for density, thermal properties etc.

Note that thermal expansion coefficients are linear (m/mK) for rocks, but
volumetric (m^3/m^3K) for fluids.
"""
from typing import Optional

import numpy as np

import porepy as pp

module_sections = ["parameters"]


class UnitFluid(object):
    """Mother of all fluids, with properties equal 1.

    Attributes:
        COMPRESSIBILITY: fluid compressibility
        BULK: bulk modulus
    """

    def __init__(self, theta_ref: Optional[float] = None):
        """Initialization of unit fluid.

        Parameters:
            theta_ref (float, optional): reference temperature in Celsius.
        """
        if theta_ref is None:
            self.theta_ref = 20 * (pp.CELSIUS)
        else:
            self.theta_ref = theta_ref

        self.COMPRESSIBILITY = 1 / pp.PASCAL
        self.BULK = 1 / self.COMPRESSIBILITY

    def thermal_expansion(self, delta_theta: float) -> float:
        """Returns thermal expansion with unit m^3 / m^3 K, i.e. volumetric.

        Parameters:
            delta_theta (float): temperature increment in Celsius.

        Returns:
            float: themal expansion
        """
        return 1

    def density(self, theta: Optional[float] = None) -> float:
        """Returns fluid density with unit: kg / m^3.

        Parameters:
            theta (float): temperature in Celsius.

        Returns:
            float: density
        """
        return 1

    def thermal_conductivity(self, theta: Optional[float] = None) -> float:
        """Returns thermal conductivity with unit : W / m K.

        Parameters:
            theta (float, optional): temperature in Celsius

        Returns:
            float: thermal conductivity
        """
        return 1

    def specific_heat_capacity(self, theta: Optional[float] = None) -> float:
        """Returns specific heat capacity with  units: J / kg K.

        Parameters:
            theta (float, optional): temperature in Celsius

        Returns:
            float: specific heat capacity
        """
        return 1

    def dynamic_viscosity(self, theta: Optional[float] = None) -> float:
        """Returns dynamic viscosity with unit: Pa s.

        Parameters:
            theta (float, optional): temperature in Celsius

        Returns:
            float: dynamic viscosity
        """
        return 1

    def hydrostatic_pressure(
        self, depth: float, theta: Optional[float] = None
    ) -> float:
        """Returns hydrostatic pressure in Pa.

        Parameters:
            depth (float): depth in meters
            theta (float): temperature in Celsius

        Returns:
            float: hydrostatic pressure
        """
        rho = self.density(theta)
        return rho * depth * pp.GRAVITY_ACCELERATION + pp.ATMOSPHERIC_PRESSURE


class Water(UnitFluid):
    def __init__(self, theta_ref=None):
        if theta_ref is None:
            self.theta_ref = 20 * (pp.CELSIUS)
        else:
            self.theta_ref = theta_ref

        self.COMPRESSIBILITY = 4e-10 / pp.PASCAL  # Moderate dependency on theta
        self.BULK = 1 / self.COMPRESSIBILITY

    def thermal_expansion(self, delta_theta: float) -> float:
        """Returns thermal expansion with unit m^3 / m^3 K, i.e. volumetric.

        Parameters:
            delta_theta (float): temperature increment in Celsius.

        Returns:
            float: themal expansion
        """
        return (
            0.0002115
            + 1.32 * 1e-6 * delta_theta
            + 1.09 * 1e-8 * np.power(delta_theta, 2)
        )

    def density(self, theta: Optional[float] = None) -> float:
        """Returns fluid density with unit: kg / m^3.

        Parameters:
            theta (float): temperature in Celsius.

        Returns:
            float: density
        """
        if theta is None:
            theta = self.theta_ref
        theta_0 = 10 * (pp.CELSIUS)
        rho_0 = 999.8349 * (pp.KILOGRAM / pp.METER**3)
        return rho_0 / (1.0 + self.thermal_expansion(theta - theta_0))

    def thermal_conductivity(self, theta: Optional[float] = None) -> float:
        """Returns thermal conductivity with unit : W / m K.

        Parameters:
            theta (float, optional): temperature in Celsius

        Returns:
            float: thermal conductivity
        """
        if theta is None:
            theta = self.theta_ref
        return (
            0.56
            + 0.002 * theta
            - 1.01 * 1e-5 * np.power(theta, 2)
            + 6.71 * 1e-9 * np.power(theta, 3)
        )

    def specific_heat_capacity(self, theta: Optional[float] = None) -> float:
        """Returns specific heat capacity with  units: J / kg K.

        Parameters:
            theta (float, optional): temperature in Celsius

        Returns:
            float: specific heat capacity
        """
        if theta is None:
            theta = self.theta_ref
        return 4245 - 1.841 * theta

    def dynamic_viscosity(self, theta: Optional[float] = None) -> float:
        """Returns dynamic viscosity with unit: Pa s.

        Parameters:
            theta (float, optional): temperature in Celsius

        Returns:
            float: dynamic viscosity
        """
        if theta is None:
            theta = self.theta_ref
        theta = pp.CELSIUS_to_KELVIN(theta)
        mu_0 = 2.414 * 1e-5 * (pp.PASCAL * pp.SECOND)
        return mu_0 * np.power(10, 247.8 / (theta - 140))

    def hydrostatic_pressure(
        self, depth: float, theta: Optional[float] = None
    ) -> float:
        """Returns hydrostatic pressure in Pa.

        Parameters:
            depth (float): depth in meters
            theta (float): temperature in Celsius

        Returns:
            float: hydrostatic pressure
        """
        rho = self.density(theta)
        return rho * depth * pp.GRAVITY_ACCELERATION + pp.ATMOSPHERIC_PRESSURE
