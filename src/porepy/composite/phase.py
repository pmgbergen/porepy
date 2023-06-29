import abc

import numpy as np
import porepy as pp

import pdb

# then, i guess, pressure_phase = p +- capillary(saturation_phase)


class Phase(abc.ABC):
    """ """

    def __init__(self, name: str = "", rho0: float = 1) -> None:
        self._name = name
        # self._s: pp.ad.MixedDimensionalVariable = ad_system.create_variables(
        #     f"{COMPOSITIONAL_VARIABLE_SYMBOLS['phase_saturation']}_{self.name}",
        #     subdomains=ad_system.mdg.subdomains(),
        # )
        self._rho0 = rho0
        self.apply_constraint = None
        # self._s = None
        self.subdomain = None
        self.equation_system = None

    @property
    def name(self) -> str:
        """Name of this phase given at instantiation."""
        return self._name

    # @property
    # def saturation(self):  # -> pp.ad.MixedDimensionalVariable:
    #     """ """
    #     return self._s

    @property
    def saturation(self):
        s = self.saturation_operator([self.subdomain]).evaluate(self.equation_system)
        return s

    def saturation_operator(self, subdomains: list[pp.Grid]):
        """ """
        if self.apply_constraint:
            s = pp.ad.Scalar(1, "one") - self.equation_system.md_variable(
                "saturation", subdomains
            )
        else:
            s = self.equation_system.md_variable("saturation", subdomains)
        return s

    # def saturation_operator(self, subdomains: list) -> pp.ad.MixedDimensionalVariable:
    #     """TODO: this is wrong, it works only in my case... fix also the constraint"""
    #     return self._s

    # Physical properties: ----------------------------------------------------------------

    def mass_density(self, p):  # TODO: p is useless, but thinl twice...
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [kg / m^3]
        Note:
        Parameters:
            p: Pressure.
            T: Temperature.
        Returns:

        better to redefine this method case by case, consider it an example
        TODO: there is something wrong here... improve it
        """

        # constant density:
        if isinstance(p, pp.ad.AdArray):
            rho = self._rho0 * pp.ad.AdArray(
                np.ones(p.val.shape), 0 * p.jac
            )  # TODO: is it right?
        else:
            rho = self._rho0 * np.ones(p.shape)

        # variable density:
        beta = 1e-5  # 5e-10
        p0 = 1  # 1e5
        rho = self._rho0 * pp.ad.functions.exp(
            beta * p0 * (p / p0 - 1)
        )  # i like dimless groups...

        # mass_density = (
        #     pp.constitutive_laws.FluidDensityFromPressure
        # )
        # subdomains = self.mdg.subdomains()
        # rho = mass_density.fluid_density(subdomains)

        return rho

    def mass_density_operator(self, subdomains, pressure):
        """
        see pressure(rho) in consitutive laws
        """
        p = pressure(subdomains)
        mass_rho_operator = pp.ad.Function(self.mass_density, "mass_density_operator")
        rho = mass_rho_operator(p)
        # NO, you need the loop over grids, you know...
        # NO, you don't need it, p is [ph, pl] and rho is a functoin depending only on p, so you don't need do distingush the grids

        return rho
