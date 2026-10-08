from __future__ import annotations

from typing import Callable, Optional, Sequence, cast

import numpy as np

import porepy as pp
from porepy.models.abstract_equations import LocalElimination

from ...obl_sampler import VTKSampler


class LiquidDriesnerCorrelations(pp.compositional.EquationOfState):
    @property
    def obl_sampler(self) -> VTKSampler:
        return self._obl_sampler

    @obl_sampler.setter
    def obl_sampler(self, obl_sampler: VTKSampler) -> None:
        self._obl_sampler = obl_sampler

    def kappa(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = (2.0) * np.ones(nc) * 1.0e-6  # 2 MW / (m C)
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
        if not hasattr(self, "_obl_sampler"):
            raise AttributeError(
                "Instance of the obl_sampler attribute is not present."
            )

        p, h, z_NaCl = thermodynamic_input
        par_points = np.array((z_NaCl, h, p)).T
        self.obl_sampler.sample_at(par_points)
        n = len(p)  # same for all input (number of cells)

        # Mass density of phase
        rho = self.obl_sampler.sampled_could.point_data["Rho_l"]
        drhodz = self.obl_sampler.sampled_could.point_data["grad_Rho_l"][:, 0]
        drhodH = self.obl_sampler.sampled_could.point_data["grad_Rho_l"][:, 1]
        drhodp = self.obl_sampler.sampled_could.point_data["grad_Rho_l"][:, 2]
        drho = np.vstack((drhodp, drhodH, drhodz))

        # specific enthalpy of phase
        h = self.obl_sampler.sampled_could.point_data["H_l"] * 1.0e-3
        dhdz = self.obl_sampler.sampled_could.point_data["grad_H_l"][:, 0] * 1.0e-3
        dhdH = self.obl_sampler.sampled_could.point_data["grad_H_l"][:, 1] * 1.0e-3
        dhdp = self.obl_sampler.sampled_could.point_data["grad_H_l"][:, 2] * 1.0e-3
        dh = np.vstack((dhdp, dhdH, dhdz))

        # dynamic viscosity of phase
        mu = self.obl_sampler.sampled_could.point_data["mu_l"] * 1.0e-6
        dmudz = self.obl_sampler.sampled_could.point_data["grad_mu_l"][:, 0] * 1.0e-6
        dmudH = self.obl_sampler.sampled_could.point_data["grad_mu_l"][:, 1] * 1.0e-6
        dmudp = self.obl_sampler.sampled_could.point_data["grad_mu_l"][:, 2] * 1.0e-6
        dmu = np.vstack((dmudp, dmudH, dmudz))

        # thermal conductivity of phase
        kappa, dkappa = self.kappa(*thermodynamic_input)  # (n,), (3, n) array

        # Fugacity coefficients
        # not required for this formulation, since no equilibrium equations
        # just show-casing it here
        phis = np.empty((2, n))  # (2, n) array  (2 components)
        dphis = np.empty(
            (2, 3, n)
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


class GasDriesnerCorrelations(pp.compositional.EquationOfState):
    @property
    def obl_sampler(self) -> VTKSampler:
        return self._obl_sampler

    @obl_sampler.setter
    def obl_sampler(self, obl_sampler: VTKSampler) -> None:
        self._obl_sampler = obl_sampler

    def kappa(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = (2.0) * np.ones(nc) * 1.0e-6  # 2 MW / (m C)
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

        if not hasattr(self, "_obl_sampler"):
            raise AttributeError(
                "Instance of the obl_sampler attribute is not present."
            )

        p, h, z_NaCl = thermodynamic_input
        par_points = np.array((z_NaCl, h, p)).T
        self.obl_sampler.sample_at(par_points)
        n = len(p)  # same for all input (number of cells)

        # Mass density of phase
        rho = self.obl_sampler.sampled_could.point_data["Rho_v"]
        drhodz = self.obl_sampler.sampled_could.point_data["grad_Rho_v"][:, 0]
        drhodH = self.obl_sampler.sampled_could.point_data["grad_Rho_v"][:, 1]
        drhodp = self.obl_sampler.sampled_could.point_data["grad_Rho_v"][:, 2]
        drho = np.vstack((drhodp, drhodH, drhodz))

        # specific enthalpy of phase
        h = self.obl_sampler.sampled_could.point_data["H_v"] * 1.0e-3
        dhdz = self.obl_sampler.sampled_could.point_data["grad_H_v"][:, 0] * 1.0e-3
        dhdH = self.obl_sampler.sampled_could.point_data["grad_H_v"][:, 1] * 1.0e-3
        dhdp = self.obl_sampler.sampled_could.point_data["grad_H_v"][:, 2] * 1.0e-3
        dh = np.vstack((dhdp, dhdH, dhdz))

        # dynamic viscosity of phase
        mu = self.obl_sampler.sampled_could.point_data["mu_v"] * 1.0e-6
        dmudz = self.obl_sampler.sampled_could.point_data["grad_mu_v"][:, 0] * 1.0e-6
        dmudH = self.obl_sampler.sampled_could.point_data["grad_mu_v"][:, 1] * 1.0e-6
        dmudp = self.obl_sampler.sampled_could.point_data["grad_mu_v"][:, 2] * 1.0e-6
        dmu = np.vstack((dmudp, dmudH, dmudz))

        # thermal conductivity of phase
        kappa, dkappa = self.kappa(*thermodynamic_input)  # (n,), (3, n) array

        # Fugacity coefficients
        # not required for this formulation, since no equilibrium equations
        # just show-casing it here
        phis = np.empty((2, n))  # (2, n) array  (2 components)
        dphis = np.empty(
            (2, 3, n)
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


class CO2LiquidDriesnerCorrelations(pp.compositional.EquationOfState):
    """CO2-rich liquid phase (h-slot), MOBILE. Reads Rho_h / H_h / mu_h from the table."""

    @property
    def obl_sampler(self) -> VTKSampler:
        return self._obl_sampler

    @obl_sampler.setter
    def obl_sampler(self, obl_sampler: VTKSampler) -> None:
        self._obl_sampler = obl_sampler

    def kappa(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = (2.0) * np.ones(nc) * 1.0e-6  # 2 MW / (m C)
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def compute_phase_properties(
        self,
        phase_state: pp.compositional.PhysicalState,
        *thermodynamic_input: np.ndarray,
        params: Optional[Sequence[np.ndarray | float]] = None,
    ) -> pp.compositional.PhaseProperties:
        if not hasattr(self, "_obl_sampler"):
            raise AttributeError(
                "Instance of the obl_sampler attribute is not present."
            )

        p, h, z_NaCl = thermodynamic_input
        par_points = np.array((z_NaCl, h, p)).T
        self.obl_sampler.sample_at(par_points)
        n = len(p)

        # Mass density of halite
        rho = self.obl_sampler.sampled_could.point_data["Rho_h"]
        drhodz = self.obl_sampler.sampled_could.point_data["grad_Rho_h"][:, 0]
        drhodH = self.obl_sampler.sampled_could.point_data["grad_Rho_h"][:, 1]
        drhodp = self.obl_sampler.sampled_could.point_data["grad_Rho_h"][:, 2]
        drho = np.vstack((drhodp, drhodH, drhodz))

        # specific enthalpy of halite
        h = self.obl_sampler.sampled_could.point_data["H_h"] * 1.0e-3
        dhdz = self.obl_sampler.sampled_could.point_data["grad_H_h"][:, 0] * 1.0e-3
        dhdH = self.obl_sampler.sampled_could.point_data["grad_H_h"][:, 1] * 1.0e-3
        dhdp = self.obl_sampler.sampled_could.point_data["grad_H_h"][:, 2] * 1.0e-3
        dh = np.vstack((dhdp, dhdH, dhdz))

        # dynamic viscosity of the CO2-liquid phase (mobile: read mu_h)
        mu = self.obl_sampler.sampled_could.point_data["mu_h"] * 1.0e-6
        dmudz = self.obl_sampler.sampled_could.point_data["grad_mu_h"][:, 0] * 1.0e-6
        dmudH = self.obl_sampler.sampled_could.point_data["grad_mu_h"][:, 1] * 1.0e-6
        dmudp = self.obl_sampler.sampled_could.point_data["grad_mu_h"][:, 2] * 1.0e-6
        dmu = np.vstack((dmudp, dmudH, dmudz))

        kappa, dkappa = self.kappa(*thermodynamic_input)

        phis = np.empty((2, n))
        dphis = np.empty((2, 3, n))

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

    obl_sampler: VTKSampler

    def get_components(self) -> Sequence[pp.FluidComponent]:
        """H2O first -> reference component (z_H2O eliminated). CO2 is the tracked component."""
        return pp.compositional.load_fluid_constants(["H2O", "CO2"], "chemicals")

    def get_phase_configuration(
        self, components: Sequence[pp.Component]
    ) -> Sequence[
        tuple[pp.compositional.EquationOfState, pp.compositional.PhysicalState, str]
    ]:
        eos_L = LiquidDriesnerCorrelations(components)
        eos_G = GasDriesnerCorrelations(components)
        eos_C = CO2LiquidDriesnerCorrelations(components)
        # assign common obl_sampler object
        eos_L.obl_sampler = self.obl_sampler
        eos_G.obl_sampler = self.obl_sampler
        eos_C.obl_sampler = self.obl_sampler
        # slot map: l = aqueous (reference liquid), v = CO2 gas, h = CO2 liquid (mobile).
        # The h-slot keeps PhysicalState.solid only as metadata (flux is governed by the
        # 3-phase relative_permeability in CO2ModelConfiguration, not by the state tag),
        # which avoids a second PhysicalState.liquid phase in the mixture.
        return [
            (pp.compositional.PhysicalState.liquid, "liq", eos_L),
            (pp.compositional.PhysicalState.gas, "gas", eos_G),
            (pp.compositional.PhysicalState.solid, "co2l", eos_C),
        ]

    def dependencies_of_phase_properties(
        self, phase: pp.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        z_NaCl = [
            comp.fraction
            for comp in self.fluid.components
            if comp != self.fluid.reference_component
        ]
        return [self.pressure, self.enthalpy] + z_NaCl  # type:ignore[return-value]


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
    """Provided by :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""

    obl_sampler: VTKSampler

    has_independent_partial_fraction: Callable[
        [pp.compositional.Component, pp.compositional.Phase], bool
    ]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""

    def gas_saturation_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        p, h, z_NaCl = thermodynamic_dependencies
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h, p)).T
        self.obl_sampler.sample_at(par_points)

        # Gas saturation
        S_v = self.obl_sampler.sampled_could.point_data["S_v"]
        dS_vdz = self.obl_sampler.sampled_could.point_data["grad_S_v"][:, 0]
        dS_vdH = self.obl_sampler.sampled_could.point_data["grad_S_v"][:, 1]
        dS_vdp = self.obl_sampler.sampled_could.point_data["grad_S_v"][:, 2]
        dS_v = np.vstack((dS_vdp, dS_vdH, dS_vdz))
        # clip to the physical range and FLATTEN the gradient where the clip is
        # active: returning the raw spline gradient at a clipped value lets the
        # linearized elimination push the saturation out of [0, 1].
        # Upper bound is 1 - S_h - S_L_EPS (S_h from the SAME sample), not 1: this keeps the
        # reference/liquid saturation s_liq = 1 - S_v - S_h >= S_L_EPS, so the option-B rel-perm
        # (which is fed the unclipped unity complement 1 - s_gas - s_h) never sees a NEGATIVE
        # liquid saturation at a phase front. Mirrors weis s_l = np.clip(1 - s_v - s_h, 0, 1);
        # clipping S_v to [0, 1] alone does not bound S_v + S_h <= 1. Inert where S_h ~ 0.
        S_L_EPS = 1.0e-6
        S_h_co = self.obl_sampler.sampled_could.point_data["S_h"]
        s_v_max = np.clip(1.0 - S_h_co - S_L_EPS, 0.0, 1.0)
        clipped = (S_v < 0.0) | (S_v > s_v_max)
        S_v = np.clip(S_v, 0.0, s_v_max)
        dS_v[:, clipped] = 0.0
        return S_v, dS_v

    def co2l_saturation_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        p, h, z_NaCl = thermodynamic_dependencies
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h, p)).T
        self.obl_sampler.sample_at(par_points)

        # Halite (solid) saturation
        S_h = self.obl_sampler.sampled_could.point_data["S_h"]
        dS_hdz = self.obl_sampler.sampled_could.point_data["grad_S_h"][:, 0]
        dS_hdH = self.obl_sampler.sampled_could.point_data["grad_S_h"][:, 1]
        dS_hdp = self.obl_sampler.sampled_could.point_data["grad_S_h"][:, 2]
        dS_h = np.vstack((dS_hdp, dS_hdH, dS_hdz))
        # Ceiling strictly below 1 so the fluid pore fraction 1 - S_h stays >= S_H_EPS:
        # the option-B liquid rel-perm divides by (1 - S_h) and scales absolute perm by
        # (1 - S_h)^2 (DriesnerModelConfiguration._liquid_relative_permeability / relative_
        # permeability); an inclusive [0, 1] clip lets S_h reach 1 -> 1/(1-S_h) and its
        # Jacobian blow up at a halite front and stall Newton. Mirrors weis_1d_solver's
        # pore = np.maximum(1 - s_h, 1e-12). Inert where S_h ~ 0 (Fig 4/5 pure water).
        S_H_EPS = 1.0e-6
        clipped = (S_h < 0.0) | (S_h > 1.0 - S_H_EPS)
        S_h = np.clip(S_h, 0.0, 1.0 - S_H_EPS)
        dS_h[:, clipped] = 0.0
        return S_h, dS_h

    def temperature_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        p, h, z_NaCl = thermodynamic_dependencies
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h, p)).T
        self.obl_sampler.sample_at(par_points)

        # Overall temperature
        T = self.obl_sampler.sampled_could.point_data["Temperature"]  # [K]
        dTdz = self.obl_sampler.sampled_could.point_data["grad_Temperature"][:, 0]
        dTdH = self.obl_sampler.sampled_could.point_data["grad_Temperature"][:, 1]
        dTdp = self.obl_sampler.sampled_could.point_data["grad_Temperature"][:, 2]
        dT = np.vstack((dTdp, dTdH, dTdz))
        return T, dT

    def CO2_liq_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        p, h, z_NaCl = thermodynamic_dependencies
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h, p)).T
        self.obl_sampler.sample_at(par_points)

        # Partial fraction of salt in liquid phase
        X_s = self.obl_sampler.sampled_could.point_data["Xl"]
        dX_sdz = self.obl_sampler.sampled_could.point_data["grad_Xl"][:, 0]
        dX_sdH = self.obl_sampler.sampled_could.point_data["grad_Xl"][:, 1]
        dX_sdp = self.obl_sampler.sampled_could.point_data["grad_Xl"][:, 2]
        dX_s = np.vstack((dX_sdp, dX_sdH, dX_sdz))
        clipped = (X_s < 0.0) | (X_s > 1.0)
        X_s = np.clip(X_s, 0.0, 1.0)
        dX_s[:, clipped] = 0.0
        return X_s, dX_s

    def CO2_gas_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        p, h, z_NaCl = thermodynamic_dependencies
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h, p)).T
        self.obl_sampler.sample_at(par_points)

        # Partial fraction of salt in vapor phase
        X_s = self.obl_sampler.sampled_could.point_data["Xv"]
        dX_sdz = self.obl_sampler.sampled_could.point_data["grad_Xv"][:, 0]
        dX_sdH = self.obl_sampler.sampled_could.point_data["grad_Xv"][:, 1]
        dX_sdp = self.obl_sampler.sampled_could.point_data["grad_Xv"][:, 2]
        dX_s = np.vstack((dX_sdp, dX_sdH, dX_sdz))
        clipped = (X_s < 0.0) | (X_s > 1.0)
        X_s = np.clip(X_s, 0.0, 1.0)
        dX_s[:, clipped] = 0.0
        return X_s, dX_s

    def CO2_co2l_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        p, h, z_NaCl = thermodynamic_dependencies
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h, p)).T
        self.obl_sampler.sample_at(par_points)

        # CO2 mass fraction in the CO2-liquid (h) slot
        X_s = self.obl_sampler.sampled_could.point_data["Xh"]
        dX_sdz = self.obl_sampler.sampled_could.point_data["grad_Xh"][:, 0]
        dX_sdH = self.obl_sampler.sampled_could.point_data["grad_Xh"][:, 1]
        dX_sdp = self.obl_sampler.sampled_could.point_data["grad_Xh"][:, 2]
        dX_s = np.vstack((dX_sdp, dX_sdH, dX_sdz))
        clipped = (X_s < 0.0) | (X_s > 1.0)
        X_s = np.clip(X_s, 0.0, 1.0)
        dX_s[:, clipped] = 0.0
        return X_s, dX_s

    def set_equations(self) -> None:
        super().set_equations()
        subdomains = self.mdg.subdomains()

        matrix = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        matrix_boundary = cast(
            pp.BoundaryGrid, self.mdg.subdomain_to_boundary_grid(matrix)
        )
        subdomains_and_matrix = subdomains + [matrix_boundary]

        chi_functions_map = {
            "CO2_liq": self.CO2_liq_func,
            "CO2_gas": self.CO2_gas_func,
            "CO2_co2l": self.CO2_co2l_func,
        }
        saturation_functions_map = {
            "gas": self.gas_saturation_func,
            "co2l": self.co2l_saturation_func,
        }

        ### Providing constitutive laws for the independent (non-reference) phase
        ### saturations based on the Driesner correlations
        rphase = self.fluid.reference_phase  # liquid phase
        independent_phases = [p for p in self.fluid.phases if p != rphase]

        for phase in independent_phases:
            self.eliminate_locally(
                phase.saturation,  # callable giving saturation on ``subdomains``
                self.dependencies_of_phase_properties(
                    phase
                ),  # callables giving primary variables on subdoains
                saturation_functions_map[phase.name],
                subdomains_and_matrix,  # all grids on which to eliminate the saturation
            )

        ### Providing constitutive laws for partial fractions based on correlations
        for phase in self.fluid.phases:
            for comp in phase:
                if self.has_independent_partial_fraction(comp, phase):
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
            self.temperature_func,
            subdomains,
        )
