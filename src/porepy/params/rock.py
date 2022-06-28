""" Hard coded typical parameters that may be of use in simulations.

Contains standard values (e.g. found in Wikipedia) for permeability, elastic
moduli etc.

Note that thermal expansion coefficients are linear (m/mK) for rocks, but
volumetric (m^3/m^3) for fluids.
"""
from typing import Optional

import porepy as pp

module_sections = ["parameters"]


def poisson_from_lame(mu: float, lmbda: float) -> float:
    """Compute Poisson's ratio from Lamé parameters.

    Parameters:
        mu (float): shear modulus (second Lamé parameter)
        lmbda (float): first Lamé parameter

    Returns:
        float: Poisson's ratio

    """
    return lmbda / (2 * (mu + lmbda))


def lame_from_young_poisson(e: float, nu: float) -> tuple(float, float):
    """Compute Lamé parameters from Young's modulus and Poisson's ratio.

    Parameters:
        e (float): Young's modulus
        nu (float): Poisson's ratio

    Returns:
        tuple(float, float): pair of first Lamé parameter and shear modulus

    """
    lmbda = e * nu / ((1 + nu) * (1 - 2 * nu))
    mu = e / (2 * (1 + nu))

    return lmbda, mu


def young_from_lame(lmbda: float, mu: float) -> float:
    """
    Compute Young's modulus from Lamé parameters.

    Parameters:
        lmbda (float): First Lamé parameter
        mu: Shear modulus (second Lamé parameter)

    Returns:
        float: Young's modulus
    """
    return mu * (3 * lmbda + 2 * mu) / (lmbda + mu)


def bulk_from_lame(lmbda: float, mu: float) -> float:
    """
    Compute bulk modulus from Lamé parameters.

    Parameters:
        lmbda (float): First Lamé parameter
        mu: Shear modulus (second Lamé parameter)

    Returns:
        float: bulk modulus
    """
    return lmbda + 2 / 3 * mu


class UnitRock(object):
    """Mother of all rocks, all values are unity.

    Attributes:
        PERMEABILITY: Permeability
        POROSITY: Porosity
        LAMBDA: First Lamé parameter
        MU: Shear modulus / Second Lamé parameter
        YOUNG_MODULUS: Young's modulus
        POISSON_RATIO: Poisson's ratio

    """

    def __init__(self, theta_ref=None):
        self.PERMEABILITY = 1
        self.THERMAL_EXPANSION = 1
        self.DENSITY = 1
        self.POROSITY = 1
        self.MU = 1
        self.LAMBDA = 1
        self.YOUNG_MODULUS = young_from_lame(self.MU, self.LAMBDA)
        self.POISSON_RATIO = poisson_from_lame(self.MU, self.LAMBDA)

        if theta_ref is None:
            self.theta_ref = 1
        else:
            self.theta_ref = theta_ref

    def specific_heat_capacity(self, theta: Optional[float] = None) -> float:
        """Determines specific heat capacity.

        Arguments:
            theta (float, optional): temperature

        Returns:
            float: specific heat capacity
        """
        return 1.0

    def thermal_conductivity(self, _) -> float:
        """Return themal conductivity.

        May depend on temperature, but currently is fixed.

        Returns:
            float: thermal conductivity
        """
        return 1.0


class SandStone(UnitRock):
    """Generic values for Sandstone.

    Data partially from:
        http://civilblog.org/2015/02/13/what-are-the-values-of-modulus-of-elasticity-poissons-ratio-for-different-rocks/

    """

    def __init__(self, theta_ref: Optional[float] = None):
        """Initialize material parameters for sand stone.

        Parameters:
            theta_ref (float, optional): reference temperature in Celsius
        """

        # Fairly permeable rock.
        self.PERMEABILITY = 1 * pp.DARCY
        self.POROSITY = 0.2
        # Reported range for Young's modulus is 0.5-8.6
        self.YOUNG_MODULUS = 5 * pp.KILOGRAM / pp.CENTI**2 * 1e5
        # Reported range for Poisson's ratio is 0.066-0.125
        self.POISSON_RATIO = 0.1

        self.LAMBDA, self.MU = lame_from_young_poisson(
            self.YOUNG_MODULUS, self.POISSON_RATIO
        )
        if theta_ref is None:
            self.theta_ref = 20 * pp.CELSIUS
        else:
            self.theta_ref = theta_ref

        self.DENSITY = 2650 * pp.KILOGRAM / pp.METER**3

    def specific_heat_capacity(self, theta: Optional[float] = None) -> float:
        """Determines specific heat capacity of sandstone.

        Arguments:
            theta (float, optional): temperature

        Returns:
            float: specific heat capacity
        """
        if theta is None:
            theta = self.theta_ref
        c_ref = 823.82
        eta = 8.9 * 1e-2
        theta_ref = 10 * pp.CELSIUS
        return c_ref + eta * (theta - theta_ref)


class Shale(UnitRock):
    """Generic values for shale.


    Data partially from:
        http://civilblog.org/2015/02/13/what-are-the-values-of-modulus-of-elasticity-poissons-ratio-for-different-rocks/

    """

    def __init__(self, theta_ref: Optional[float] = None):
        """Initialize material parameters for shale.

        Parameters:
            theta_ref (float, optional): reference temperature in Celsius
        """
        # No source for permeability and porosity.
        self.PERMEABILITY = 1e-5 * pp.DARCY
        self.POROSITY = 0.01
        # Reported range for Young's modulus is 0.8-3.0
        self.YOUNG_MODULUS = 1.5 * pp.KILOGRAM / pp.CENTI**2 * 1e5
        # Reported range for Poisson's ratio is 0.11-0.54 (the latter is strange)
        self.POISSON_RATIO = 0.3

        self.LAMBDA, self.MU = lame_from_young_poisson(
            self.YOUNG_MODULUS, self.POISSON_RATIO
        )

        if theta_ref is None:
            self.theta_ref = 20 * pp.CELSIUS
        else:
            self.theta_ref = theta_ref

        self.DENSITY = 2650 * pp.KILOGRAM / pp.METER**3

    def specific_heat_capacity(self, theta: Optional[float] = None) -> float:
        """Determines specific heat capacity of shale.

        Arguments:
            theta (float, optional): temperature

        Returns:
            float: specific heat capacity
        """
        if theta is None:
            theta = self.theta_ref
        c_ref = 794.37
        eta = 10.26 * 1e-2
        theta_ref = 10 * pp.CELSIUS
        return c_ref + eta * (theta - theta_ref)


class Granite(UnitRock):
    """Generic values for granite.
    Data partially from:
        http://civilblog.org/2015/02/13/what-are-the-values-of-modulus-of-elasticity-poissons-ratio-for-different-rocks/
    And:
    https://www.jsg.utexas.edu/tyzhu/files/Some-Useful-Numbers.pdf
    """

    def __init__(self, theta_ref: Optional[float] = None):
        """Initialize material parameters for granite.

        Parameters:
            theta_ref (float, optional): reference temperature in Celsius
        """
        # No source for permeability and porosity
        self.PERMEABILITY = 1e-8 * pp.DARCY
        self.POROSITY = 0.01
        # Reported range for Young's modulus by jsg is 10-70GPa
        self.YOUNG_MODULUS = 40.0 * pp.GIGA * pp.PASCAL
        # Reported range for Poisson's ratio is 0.125-0.25
        self.POISSON_RATIO = 0.2

        # Reported density
        self.DENSITY = 2700.0 * pp.KILOGRAM / pp.METER**3
        self.LAMBDA, self.MU = lame_from_young_poisson(
            self.YOUNG_MODULUS, self.POISSON_RATIO
        )
        # Units of thermal expansion: m / m K, i.e. linear.
        # From https://www.engineeringtoolbox.com/linear-expansion-coefficients-d_95.html
        self.THERMAL_EXPANSION = 8e-6 * pp.METER / (pp.METER * pp.CELSIUS)
        if theta_ref is None:
            self.theta_ref = 20.0 * pp.CELSIUS
        else:
            self.theta_ref = theta_ref

    def specific_heat_capacity(self, theta: Optional[float] = None) -> float:
        """Determines specific heat capacity of granite.

        Arguments:
            theta (float, optional): temperature

        Returns:
            float: specific heat capacity
        """

        if theta is None:
            theta = self.theta_ref
        c_ref = 790.0
        eta = 0
        theta_ref = 0
        return c_ref + eta * (theta - theta_ref)

    def thermal_conductivity(self, theta: Optional[float] = None) -> float:
        """Return themal conductivity of granite.

        Parameters:
            theta (float, optional): temperature in Celsius (not used)

        Returns:
            float: thermal conductivity
        """
        return 3.07
