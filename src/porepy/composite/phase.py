import abc

import scipy as sp
import numpy as np
import porepy as pp

import pdb

# then, i guess, pressure_phase = p +- capillary(saturation_phase)


class Phase(abc.ABC):
    """ """

    def __init__(self, name: str = "", rho0: float = 1, p0: float = 1, beta: float = 1e-10) -> None:
        self._name = name
        self._rho0 = rho0
        self._p0 = p0
        self._beta = beta
        self.apply_constraint = None
        self.equation_system = None
        self.subdomain = None

    @property
    def name(self) -> str:
        """Name of this phase given at instantiation."""
        return self._name

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

    # Physical properties: ----------------------------------------------------------------

    def mass_density(self, p): #, vector_dim=False, sds_dof=False): 
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [kg / m^3]
        Note:
        Parameters:
            p: Pressure.
            T: Temperature.
        Returns:

        better to redefine this method case by case, consider it an example
        """

        # constant density:
        if isinstance(p, pp.ad.AdArray):
            rho = self._rho0 * pp.ad.AdArray(
                np.ones(p.val.shape), 0 * p.jac
            )  # TODO: is it right?
        else:
            rho = self._rho0 * np.ones(p.shape)

        # variable density:
        rho = self._rho0 * pp.ad.functions.exp(
            self._beta * self._p0 * (p / self._p0 - 1)
        )  # i like dimless groups...

        return rho

    def mass_density_operator(self, subdomains, pressure): #, vector_dim=False):
        """
        see pressure(rho) in consitutive laws
        """

        p = pressure(subdomains)
        mass_rho_operator = pp.ad.Function(self.mass_density, "mass_density_operator")
        rho = mass_rho_operator(p) # old

        return rho
