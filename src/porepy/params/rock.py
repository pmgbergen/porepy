""" Hard coded typical parameters that may be of use in simulations.

Contains standard values (e.g. found in Wikipedia) for permeability, elastic
moduli etc.

"""
from porepy.params import units


def poisson_from_lame(mu, lmbda):
    """ Compute Poisson's ratio from Lame parameters

    Parameters:
        mu (double): first Lame parameter
        lmbda (double): Second Lame parameter

    Returns:
        double: Poisson's ratio

    """
    return lmbda / (2 * (mu + lmbda))


def lame_from_young_poisson(e, nu):
    """ Compute Lame parameters from Young's modulus and Poisson's ratio.

    Parameters:
        e (double): Young's modulus
        nu (double): Poisson's ratio

    Returns:
        double: First Lame parameter
        double: Second Lame parameter / shear modulus

    """
    lmbda = e * nu / ((1 + nu) * (1 - 2 * nu))
    mu = e / (2 * (1 + nu))

    return lmbda, mu


class UnitRock(object):
    """ Mother of all rocks, all values are unity.

    Attributes:
        PERMEABILITY:
        POROSITY:
        LAMBDA: First Lame parameter
        MU: Second lame parameter / shear modulus
        YOUNG_MODULUS: Young's modulus
        POISSON_RATIO:

    """

    PERMEABILITY = 1
    POROSITY = 1
    MU = 1
    LAMBDA = 1
    YOUNG_MODULUS = 1
    POSSION_RATIO = 1


class SandStone(UnitRock):
    """ Generic values for Sandstone.

    Data partially from:
        http://civilblog.org/2015/02/13/what-are-the-values-of-modulus-of-elasticity-poissons-ratio-for-different-rocks/

    """

    def __init__(self):

        # Fairly permeable rock.
        self.PERMEABILITY = 1 * units.DARCY
        self.POROSITY = 0.2
        # Reported range for Young's modulus is 0.5-8.6
        self.YOUNG_MODULUS = 5 * units.KILOGRAM / units.CENTI ** 2 * 1e5
        # Reported range for Poisson's ratio is 0.066-0.125
        self.POISSON_RATIO = 0.1

        self.LAMBDA, self.MU = lame_from_young_poisson(
            self.YOUNG_MODULUS, self.POISSON_RATIO
        )


class Shale(UnitRock):
    """ Generic values for shale.


    Data partially from:
        http://civilblog.org/2015/02/13/what-are-the-values-of-modulus-of-elasticity-poissons-ratio-for-different-rocks/

    """

    def __init__(self):
        # No source for permeability and porosity.
        self.PERMEABILITY = 1e-5 * units.DARCY
        self.POROSITY = 0.01
        # Reported range for Young's modulus is 0.8-3.0
        self.YOUNG_MODULUS = 1.5 * units.KILOGRAM / units.CENTI ** 2 * 1e5
        # Reported range for Poisson's ratio is 0.11-0.54 (the latter is strange)
        self.POISSON_RATIO = 0.3

        self.LAMBDA, self.MU = lame_from_young_poisson(
            self.YOUNG_MODULUS, self.POISSON_RATIO
        )


class Granite(UnitRock):
    """ Generic values for granite.


    Data partially from:
        http://civilblog.org/2015/02/13/what-are-the-values-of-modulus-of-elasticity-poissons-ratio-for-different-rocks/

    """

    def __init__(self):
        # No source for permeability and porosity
        self.PERMEABILITY = 1e-8 * units.DARCY
        self.POROSITY = 0.01
        # Reported range for Young's modulus is 2.6-7.0
        self.YOUNG_MODULUS = 5 * units.KILOGRAM / units.CENTI ** 2 * 1e5
        # Reported range for Poisson's ratio is 0.125-0.25
        self.POISSON_RATIO = 0.2

        self.LAMBDA, self.MU = lame_from_young_poisson(
            self.YOUNG_MODULUS, self.POISSON_RATIO
        )
