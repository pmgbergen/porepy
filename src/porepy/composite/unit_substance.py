""" Contains the simple UnitIdealFluid component and UnitIncompressibleFluid. """

import porepy as pp
import numpy as np

from .substance import FluidSubstance, SolidSubstance
from._composite_utils import COMPUTATIONAL_VARIABLES


class UnitIncompressibleFluid(FluidSubstance):
    """
    Represents the academic example fluid with all properties constant and unitary.
    Intended usage is testing, debugging and demonstration.
    
    For a proper documentation of all properties, see parent class.
    """

    @staticmethod
    def molar_mass() -> float:
        return 1.

    def molar_density(self) -> pp.ad.Operator:
        return pp.ad.Array(np.ones(self.cd.nc))

    def Fick_diffusivity(self, *args, **kwargs) -> pp.ad.Operator:
        return pp.ad.Array(np.ones(self.cd.nc))

    def thermal_conductivity(self, *args, **kwargs) -> pp.ad.Operator:
        return pp.ad.Array(np.ones(self.cd.nc))

class UnitIdealFluid(UnitIncompressibleFluid):
    """
    Represents the academic ideal fluid with constant, unitary properties.
    The Ideal Gas Law is implemented using the Ideal Gas Constant.
    Intended usage is testing, debugging and demonstration.
    
    For a proper documentation of all properties, see parent class.
    """

    def molar_density(self, pressure, temperature) -> pp.ad.Operator:

        # we check if the necessary variables are instantiated
        # NOTE: we could try to expand this to use also enthalpy (linearized formula)
        idx1 = self.cd.is_variable(COMPUTATIONAL_VARIABLES["pressure"])
        idx2 = self.cd.is_variable(COMPUTATIONAL_VARIABLES["temperature"])

        if idx1 and idx2:
            pressure = self.cd(COMPUTATIONAL_VARIABLES["pressure"])
            temperature = self.cd(COMPUTATIONAL_VARIABLES["temperature"])

            return pp.ad.Function(
                lambda p, T: p / (self.ideal_gas_constant * T),
                "Ideal_molar_density",
            )(pressure, temperature)
        else: #TODO evaluate this approach
            raise RuntimeError("Cannot call molar density of" + 
            "substance '%s': state variables not instantiated."%(self.name))


class UnitSolid(SolidSubstance):
    """
    Represent the academic unit solid, with constant unitary properties.
    Intended usage is testing, debugging and demonstration.
    
    For a proper documentation of all properties, see parent class.
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

    def Fick_diffusivity(self, *args, **kwargs) -> pp.ad.Operator:
        return pp.ad.Array(np.ones(self.cd.nc))

    def thermal_conductivity(self, *args, **kwargs) -> pp.ad.Operator:
        return pp.ad.Array(np.ones(self.cd.nc))