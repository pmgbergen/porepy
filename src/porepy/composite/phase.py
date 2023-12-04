import abc

import scipy as sp
import numpy as np
import porepy as pp

import pdb


"""
TODO: the setting of a subdomain is weird, did so to be compliant with both hu and operator. Find a better solution
"""


class Phase(abc.ABC):
    """ """

    def __init__(
        self, name: str = "", rho0: float = 1, p0: float = 1, beta: float = 1e-10
    ) -> None:
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
    def saturation(self) -> pp.ad.AdArray:
        s = self.saturation_operator([self.subdomain]).evaluate(self.equation_system)
        return s

    def saturation_operator(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """ """
        if self.apply_constraint:
            s = pp.ad.Scalar(1, "one") - self.equation_system.md_variable(
                "saturation", subdomains
            )
        else:
            s = self.equation_system.md_variable("saturation", subdomains)
        return s

    # Physical properties: ----------------------------------------------------------------

    def mass_density(self, p: pp.ad.AdArray) -> pp.ad.AdArray:
        """ """

        # # constant density:
        # if isinstance(p, pp.ad.AdArray):
        #     rho = self._rho0 * pp.ad.AdArray(np.ones(p.val.shape), 0 * p.jac)
        # else:
        #     rho = self._rho0 * np.ones(p.shape)

        # variable density:
        rho = self._rho0 * pp.ad.functions.exp(
            self._beta * self._p0 * (p / self._p0 - 1)
        )  # i like dimless groups...

        return rho

    def mass_density_operator(
        self, subdomains: list[pp.Grid], pressure: pp.ad.Operator
    ) -> pp.ad.Operator:
        """ """

        p = pressure(subdomains)
        mass_rho_operator = pp.ad.Function(self.mass_density, "mass_density_operator")
        rho = mass_rho_operator(p)

        return rho
