"""Contains the phase class for the Peng-Robinson EoS."""
from __future__ import annotations

from typing import Callable

import porepy as pp

from ..phase import Phase


class PR_Phase(Phase):
    """Representation of a phase using the Peng-Robinson EoS.

    Thermodynamic properties are represented by references to callables, which are set by the
    Peng-Robinson composition class.

    This class is not intended to be used or instantiated except by the respective composition
    class.

    """

    def __init__(self, name: str, ad_system: pp.ad.ADSystem) -> None:
        super().__init__(name, ad_system)

        self._h: Callable
        self._rho: Callable
        self._mu: Callable
        self._kappa: Callable

    def density(self, p, T):
        X = (self.fraction_of_component(component) for component in self)
        return self._rho(p, T, *X)

    def specific_enthalpy(self, p, T):
        X = (self.fraction_of_component(component) for component in self)
        return self._h(p, T, *X)

    def dynamic_viscosity(self, p, T):
        X = (self.fraction_of_component(component) for component in self)
        return self._mu(p, T, *X)

    def thermal_conductivity(self, p, T):
        X = (self.fraction_of_component(component) for component in self)
        return self._kappa(p, T, *X)
