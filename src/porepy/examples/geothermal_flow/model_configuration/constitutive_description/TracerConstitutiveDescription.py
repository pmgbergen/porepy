from __future__ import annotations

from typing import Callable, Sequence, cast

import numpy as np

import porepy as pp
import porepy.compositional as ppc
from porepy.models.abstract_equations import LocalElimination
from porepy.models.protocol import PorePyModel


def gas_saturation_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    p, h, z_NaCl = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_NaCl)

    nc = len(thermodynamic_dependencies[0])
    vals = 0.5 * np.ones_like(z_NaCl)
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    return vals, diffs


def temperature_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_NaCl = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_NaCl)

    nc = len(thermodynamic_dependencies[0])

    factor = 0.25
    vals = np.array(h) * factor
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[1, :] = 1.0 * factor
    return vals, diffs


def H2O_liq_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_NaCl = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_NaCl)

    nc = len(thermodynamic_dependencies[0])
    vals = np.array(1 - z_NaCl)
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = -1.0
    return vals, diffs


def NaCl_liq_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_NaCl = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_NaCl)

    nc = len(thermodynamic_dependencies[0])
    vals = np.array(z_NaCl)
    # row-wise storage of derivatives, (4, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = +1.0
    return vals, diffs


def H2O_gas_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_NaCl = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_NaCl)

    nc = len(thermodynamic_dependencies[0])
    vals = np.array(1 - z_NaCl)
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = -1.0
    return vals, diffs


def NaCl_gas_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_NaCl = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_NaCl)

    nc = len(thermodynamic_dependencies[0])
    vals = np.array(z_NaCl)
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = +1.0
    return vals, diffs


chi_functions_map = {
    "H2O_liq": H2O_liq_func,
    "NaCl_liq": NaCl_liq_func,
    "H2O_gas": H2O_gas_func,
    "NaCl_gas": NaCl_gas_func,
}


class LiquidLikeCorrelations(ppc.EquationOfState):
    """Class implementing the calculation of thermodynamic properties.

    Note:
        By thermodynamic properties, this framework refers to the below
        indicated quantities, and **not** quantitie which are variables.

        Fractions (partial and saturations) and other intensive quantities like
        temperature need a separate treatment because they are always modelled as
        variables, whereas properties are always dependent expressions.

    """

    def rho_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        # vals = (55508.435061792) * np.ones(nc)
        vals = (1000.0) * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def mu_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (0.001) * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def h(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (1000.0) * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def kappa(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (1.8) * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def compute_phase_properties(
        self, phase_state: ppc.PhysicalState, *thermodynamic_input: np.ndarray
    ) -> ppc.PhaseProperties:
        """Function will be called to compute the values for a phase.
        ``phase_type`` indicates the phsycal type (0 - liq, 1 - gas).
        ``thermodynamic_dependencies`` are as defined by the user.
        """

        p, h, z_NaCl = thermodynamic_input
        # same for all input (number of cells)
        assert len(p) == len(h) == len(z_NaCl)
        nc = len(thermodynamic_input[0])

        # mass density of phase
        rho, drho = self.rho_func(*thermodynamic_input)  # (n,), (3, n) array

        # specific enthalpy of phase
        h, dh = self.h(*thermodynamic_input)  # (n,), (3, n) array
        # dynamic viscosity of phase
        mu, dmu = self.mu_func(*thermodynamic_input)  # (n,), (3, n) array
        # thermal conductivity of phase
        kappa, dkappa = self.kappa(*thermodynamic_input)  # (n,), (3, n) array

        # Fugacity coefficients
        # not required for this formulation, since no equilibrium equations
        # just show-casing it here
        phis = np.empty((2, nc))  # (2, n) array  (2 components)
        dphis = np.empty(
            (2, 3, nc)
        )  # (2, 3, n)  array (2 components, 3 dependencies, n cells)

        return ppc.PhaseProperties(
            state=phase_state,
            rho=rho,
            drho=drho,
            h=h,
            dh=dh,
            mu=mu,
            dmu=dmu,
            kappa=kappa,
            dkappa=dkappa,
            phis=phis,
            dphis=dphis,
        )


class GasLikeCorrelations(ppc.EquationOfState):
    """Class implementing the calculation of thermodynamic properties.

    Note:
        By thermodynamic properties, this framework refers to the below
        indicated quantities, and **not** quantitie which are variables.

        Fractions (partial and saturations) and other intensive quantities like
        temperature need a separate treatment because they are always modelled as
        variables, whereas properties are always dependent expressions.

    """

    def rho_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = 1.0 * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def mu_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (0.00001) * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def h(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (1000.0) * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def kappa(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (1.0e-2) * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def compute_phase_properties(
        self, phase_state: ppc.PhysicalState, *thermodynamic_input: np.ndarray
    ) -> ppc.PhaseProperties:
        """Function will be called to compute the values for a phase.
        ``phase_type`` indicates the phsycal type (0 - liq, 1 - gas).
        ``thermodynamic_dependencies`` are as defined by the user.
        """

        p, h, z_NaCl = thermodynamic_input
        assert len(p) == len(h) == len(z_NaCl)
        nc = len(thermodynamic_input[0])

        # mass density of phase
        rho, drho = self.rho_func(*thermodynamic_input)  # (n,), (3, n) array

        # specific enthalpy of phase
        h, dh = self.h(*thermodynamic_input)  # (n,), (3, n) array
        # dynamic viscosity of phase
        mu, dmu = self.mu_func(*thermodynamic_input)  # (n,), (3, n) array
        # thermal conductivity of phase
        kappa, dkappa = self.kappa(*thermodynamic_input)  # (n,), (3, n) array

        # Fugacity coefficients
        # not required for this formulation, since no equilibrium equations
        # just show-casing it here
        phis = np.empty((2, nc))  # (2, n) array  (2 components)
        dphis = np.empty(
            (2, 3, nc)
        )  # (2, 3, n)  array (2 components, 3 dependencies, n cells)

        return ppc.PhaseProperties(
            state=phase_state,
            rho=rho,
            drho=drho,
            h=h,
            dh=dh,
            mu=mu,
            dmu=dmu,
            kappa=kappa,
            dkappa=dkappa,
            phis=phis,
            dphis=dphis,
        )


class FluidMixture(PorePyModel):
    """Mixture mixin creating the brine mixture with two components."""

    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.compositional_flow.VariablesEnergyBalance`."""

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.fluid_mass_balance.VariableSinglePhaseFlow`."""

    def get_components(self) -> Sequence[pp.FluidComponent]:
        """Setting H20 as first component in Sequence makes it the reference component.
        z_H20 will be eliminated."""
        return ppc.load_fluid_constants(["H2O", "NaCl"], "chemicals")

    def get_phase_configuration(
        self, components: Sequence[ppc.Component]
    ) -> Sequence[tuple[ppc.EquationOfState, ppc.PhysicalState, str]]:
        eos_L = LiquidLikeCorrelations(components)
        eos_G = GasLikeCorrelations(components)
        return [
            (eos_L, ppc.PhysicalState.liquid, "liq"),
            (eos_G, ppc.PhysicalState.gas, "gas"),
        ]

    def dependencies_of_phase_properties(
        self, phase: ppc.Phase
    ) -> list[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        z_NaCl = [
            comp.fraction
            for comp in self.fluid.components
            if comp != self.fluid.reference_component
        ]
        dependencies = cast(
            list[Callable[[pp.GridLikeSequence], pp.ad.Variable]],
            [self.pressure, self.enthalpy] + z_NaCl,
        )
        return dependencies


class SecondaryEquations(LocalElimination):
    """Mixin to provide expressions for dangling variables.

    The CF framework has the following quantities always as independent variables:

    - independent phase saturations
    - partial fractions (independent since no equilibrium)
    - temperature (needs to be expressed through primary variables in this model, since
      no p-h equilibrium)

    """

    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""

    has_independent_partial_fraction: Callable[[ppc.Component, ppc.Phase], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""

    def set_equations(self) -> None:
        super().set_equations()

        subdomains = self.mdg.subdomains()
        matrix = self.mdg.subdomains(dim=self.mdg.dim_max())[0]

        ### Providing constitutive law for gas saturation based on correlation
        rphase = self.fluid.reference_phase  # liquid phase
        # gas phase is independent
        independent_phases = [p for p in self.fluid.phases if p != rphase]

        # domains on which to eliminate are all subdomains and the matrix boundary
        on_domains: list[pp.Grid | pp.BoundaryGrid] = subdomains + [
            cast(pp.BoundaryGrid, self.mdg.subdomain_to_boundary_grid(matrix))
        ]

        for phase in independent_phases:
            self.eliminate_locally(
                phase.saturation,  # callable giving saturation on ``subdomains``
                self.dependencies_of_phase_properties(
                    phase
                ),  # callables giving primary variables on subdoains
                gas_saturation_func,  # numerical function implementing correlation
                on_domains,  # all subdomains on which to eliminate s_gas
                # dofs = {'cells': 1},  # default value
            )

        ### Providing constitutive laws for partial fractions based on correlations
        for phase in self.fluid.phases:
            for comp in phase:
                if self.has_independent_partial_fraction(comp, phase):
                    self.eliminate_locally(
                        phase.partial_fraction_of[comp],
                        self.dependencies_of_phase_properties(phase),
                        chi_functions_map[comp.name + "_" + phase.name],
                        on_domains,
                    )

        ### Provide constitutive law for temperature
        self.eliminate_locally(
            self.temperature,
            self.dependencies_of_phase_properties(rphase),  # since same for all.
            temperature_func,
            on_domains,
        )
