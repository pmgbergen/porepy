""" Contains the simple UnitIdealFluid component and UnitIncompressibleFluid. """

import porepy as pp
import numpy as np

from .substance import FluidSubstance, SolidSubstance


class UnitIncompressibleFluid(FluidSubstance):
    """ Represents the academic example fluid with all constant and unitary properties.
    Intended usage are testing and debugging, but also demonstrations.
    
    For a proper documentation of all properties, see Component class. """

    @staticmethod
    def molar_mass() -> float:
        return 1.

    def molar_density(self) -> pp.ad.Operator:
        pp.ad.Array(np.ones(self.cd.nc))


class UnitIdealFluid(UnitIncompressibleFluid):
    """ Represents the academic ideal fluid with constant, unitary properties.
    
    The Ideal Gas Law is implemented using the Ideal Gas Constant.
    """

    def molar_density(self, pressure, temperature) -> pp.ad.Operator:

        return pp.ad.Function(
            lambda p, T: p / (self.ideal_gas_constant * T),
            "Ideal_molar_density",
        )(pressure, temperature)


class UnitSolid(SolidSubstance):
    """ Represent the academic unit solid, with constant unitary properties.
    """
    
    @staticmethod
    def molar_mass() -> float:
        return 1.

    def molar_density(self) -> pp.ad.Operator:
        pp.ad.Array(np.ones(self.cd.nc))

    @staticmethod
    def base_porosity() -> float:
        return 1.

    @staticmethod
    def base_permeability() -> float:
        return 1.