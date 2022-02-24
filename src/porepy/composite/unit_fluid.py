""" Contains the simpe UnitIdealFluid component and UnitIncompressibleFluid. """

import porepy as pp
import numpy as np

from .component import Component


class UnitIncompressibleFluid(Component):
    """ Represents the academic example fluid with all constant and unitery properties.
    Intended usage are testing and debugging, but also demonstrations.
    
    For a proper documentation of all properties, see Component class. """

    def molar_mass(self) -> float:
        return 1.

    def molar_density(self, **kwargs) -> pp.ad.Operator:
        pp.ad.Array(np.ones(self.nc))


class UnitIdealFluid(UnitIncompressibleFluid):
    """ Represents the academic ideal fluid with constant, unitary properties.
    
    The Ideal Gas Law is implemented using the Ideal Gas Constant.
    """

    def molar_density(self, pressure, temperature) -> pp.ad.Operator:

        return pp.ad.Function(
            lambda p, T: p / (self.ideal_gas_constant * T),
            "Ideal_molar_density",
        )(pressure, temperature)