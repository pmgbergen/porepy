from __future__ import annotations

from typing import Callable, Optional, Sequence, cast

import numpy as np

import porepy as pp
from porepy.models.abstract_equations import LocalElimination


def gas_saturation_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])
    vals = z_CO2
    vals = np.clip(vals, 1.0e-16, 1.0-1.0e-16)

    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = +1.0
    return vals, diffs


def temperature_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])

    factor = 250.0
    vals = np.array(h) * factor
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[1, :] = 1.0 * factor
    return vals, diffs


def H2O_liq_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])
    vals = np.ones_like(z_CO2)
    vals = np.clip(vals, 1.0e-16, 1.0-1.0e-16)
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    return vals, diffs


def CO2_liq_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CO2)
    vals = np.clip(vals, 1.0e-16, 1.0-1.0e-16)

    # row-wise storage of derivatives, (4, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    return vals, diffs


def H2O_gas_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CO2)
    vals = np.clip(vals, 1.0e-16, 1.0-1.0e-16)

    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    return vals, diffs


def CO2_gas_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])
    vals = np.ones_like(z_CO2)
    vals = np.clip(vals, 1.0e-16, 1.0-1.0e-16)

    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    return vals, diffs


chi_functions_map = {
    "H2O_liq": H2O_liq_func,
    "CO2_liq": CO2_liq_func,
    "H2O_gas": H2O_gas_func,
    "CO2_gas": CO2_gas_func,
}


class LiquidLikeCorrelations(pp.compositional.EquationOfState):
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
        vals = (1000.0) * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def mu_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (1.0e-3) * np.ones(nc) * 1.0e-6
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def h(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (2.0) * np.ones(nc) * 1.0e-6
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def kappa(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (2.0) * np.ones(nc) * 1.0e-6
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def compute_phase_properties(
        self,
        phase_state: pp.compositional.PhysicalState,
        *thermodynamic_input: np.ndarray,
        params: Optional[Sequence[np.ndarray | float]] = None,
    ) -> pp.compositional.PhaseProperties:
        """Function will be called to compute the values for a phase.
        ``phase_type`` indicates the phsycal type (0 - liq, 1 - gas).
        ``thermodynamic_dependencies`` are as defined by the user.
        """

        p, h, z_CO2 = thermodynamic_input
        # same for all input (number of cells)
        assert len(p) == len(h) == len(z_CO2)
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

        return pp.compositional.PhaseProperties(
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


class GasLikeCorrelations(pp.compositional.EquationOfState):
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
        vals = (1000.0) * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def mu_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (1.0e-3) * np.ones(nc) * 1.0e-6
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def h(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (2.0) * np.ones(nc) * 1.0e-6
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def kappa(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (2.0) * np.ones(nc) * 1.0e-6
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def compute_phase_properties(
        self,
        phase_state: pp.compositional.PhysicalState,
        *thermodynamic_input: np.ndarray,
        params: Optional[Sequence[np.ndarray | float]] = None,
    ) -> pp.compositional.PhaseProperties:
        """Function will be called to compute the values for a phase.
        ``phase_type`` indicates the phsycal type (0 - liq, 1 - gas).
        ``thermodynamic_dependencies`` are as defined by the user.
        """

        p, h, z_CO2 = thermodynamic_input
        assert len(p) == len(h) == len(z_CO2)
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

        return pp.compositional.PhaseProperties(
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


class FluidMixture(pp.PorePyModel):
    """Mixture mixin creating the brine mixture with two components."""

    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def get_components(self) -> Sequence[pp.FluidComponent]:
        """Setting H20 as first component in Sequence makes it the reference component.
        z_H20 will be eliminated."""
        return pp.compositional.load_fluid_constants(["H2O", "CO2"], "chemicals")

    def get_phase_configuration(
        self, components: Sequence[pp.Component]
    ) -> Sequence[
        tuple[pp.compositional.EquationOfState, pp.compositional.PhysicalState, str]
    ]:
        eos_L = LiquidLikeCorrelations(components)
        eos_G = GasLikeCorrelations(components)
        # phase_state, name, eos
        return [
            (pp.compositional.PhysicalState.liquid, "liq", eos_L),
            (pp.compositional.PhysicalState.gas, "gas", eos_G),
        ]

    def dependencies_of_phase_properties(
        self, phase: pp.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        z_CO2 = [
            comp.fraction
            for comp in self.fluid.components
            if comp != self.fluid.reference_component
        ]
        return [self.pressure, self.enthalpy] + z_CO2  # type:ignore[return-value]


class SecondaryEquations(LocalElimination):
    """Mixin to provide expressions for dangling variables.

    The CF framework has the following quantities always as independent variables:

    - independent phase saturations
    - partial fractions (independent since no equilibrium)
    - temperature (needs to be expressed through primary variables in this model, since
      no p-h equilibrium)

    """

    dependencies_of_phase_properties: Callable[
        ..., Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]
    ]
    """Defined in the Brine mixture mixin."""

    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""

    def set_equations(self) -> None:
        super().set_equations()
        subdomains = self.mdg.subdomains()

        matrix = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        matrix_boundary = cast(
            pp.BoundaryGrid, self.mdg.subdomain_to_boundary_grid(matrix)
        )
        subdomains_and_matrix = subdomains + [matrix_boundary]

        ### Providing constitutive law for gas saturation based on correlation
        rphase = self.fluid.reference_phase  # liquid phase
        # gas phase is independent
        independent_phases = [p for p in self.fluid.phases if p != rphase]

        for phase in independent_phases:
            self.eliminate_locally(
                phase.saturation,  # callable giving saturation on ``subdomains``
                self.dependencies_of_phase_properties(
                    phase
                ),  # callables giving primary variables on subdoains
                gas_saturation_func,  # numerical function implementing correlation
                subdomains_and_matrix,  # all subdomains on which to eliminate s_gas
            )

        ### Providing constitutive laws for partial fractions based on correlations
        for phase in self.fluid.phases:
            for comp in phase:
                check = self.has_independent_partial_fraction(comp, phase)
                if check:
                    print("component-phase has independent fraction: ", (comp.name, phase.name))
                    self.eliminate_locally(
                        phase.partial_fraction_of[comp],
                        self.dependencies_of_phase_properties(phase),
                        chi_functions_map[comp.name + "_" + phase.name],
                        subdomains_and_matrix,
                    )

        ### Provide constitutive law for temperature
        self.eliminate_locally(
            self.temperature,
            self.dependencies_of_phase_properties(rphase),  # since same for all.
            temperature_func,
            subdomains_and_matrix,
        )
