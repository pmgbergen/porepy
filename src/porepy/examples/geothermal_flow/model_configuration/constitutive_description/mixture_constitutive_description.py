from __future__ import annotations

from typing import Callable, Sequence, Optional, cast

import numpy as np

import porepy as pp
from porepy.models.abstract_equations import LocalElimination

from ...vtk_sampler import VTKSampler
from enum import Enum


class ComponentSystem(str, Enum):
    WATER = "water"
    WATER_SALT = "water+NaCl"


class PhaseMode(str, Enum):
    SINGLE_PHASE = "single-phase"
    TWO_PHASE = "two-phase"
    THREE_PHASE = "three-phase"


class LiquidDriesnerCorrelations(pp.compositional.EquationOfState):
    """ Class implementing the calculation of thermodynamic properties of liquid phases
        using the Driesner Correlation
    """

    _vtk_sampler: 'VTKSampler'

    @property
    def vtk_sampler(self):
        return self._vtk_sampler

    @vtk_sampler.setter
    def vtk_sampler(self, vtk_sampler: VTKSampler):
        self._vtk_sampler = vtk_sampler

    def kappa(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:  # value, jacobian

        nc = len(thermodynamic_dependencies[0])
        vals = (2.0) * np.ones(nc)
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
        if not hasattr(self, "_vtk_sampler"):
            raise AttributeError(
                "Instance of the vtk_sampler attribute is not present."
            )
        
        if len(thermodynamic_input) == 3:
            p, h, z_NaCl = thermodynamic_input
        elif len(thermodynamic_input) == 2:
            p, h = thermodynamic_input
            z_NaCl = np.zeros_like(p)

        par_points = np.array((z_NaCl, h, p)).T
        self.vtk_sampler.sample_at(par_points)
        
        n = len(p)  # same for all input (number of cells)

        # Mass density of phase
        rho = self.vtk_sampler.sampled_cloud.point_data["Rho_l"]
        drhodz = self.vtk_sampler.sampled_cloud.point_data["grad_Rho_l"][:, 0]
        drhodH = self.vtk_sampler.sampled_cloud.point_data["grad_Rho_l"][:, 1]
        drhodp = self.vtk_sampler.sampled_cloud.point_data["grad_Rho_l"][:, 2]
        drho = [drhodp, drhodH]
        if len(thermodynamic_input) == 3:
            drho.append(drhodz)
        drho = np.vstack(drho)

        # specific enthalpy of phase
        h = self.vtk_sampler.sampled_cloud.point_data["H_l"]
        dhdz = self.vtk_sampler.sampled_cloud.point_data["grad_H_l"][:, 0]
        dhdH = self.vtk_sampler.sampled_cloud.point_data["grad_H_l"][:, 1]
        dhdp = self.vtk_sampler.sampled_cloud.point_data["grad_H_l"][:, 2]
        dh = [dhdp, dhdH]
        if len(thermodynamic_input) == 3:
            dh.append(dhdz)
        dh = np.vstack(dh)

        # dynamic viscosity of phase
        mu = self.vtk_sampler.sampled_cloud.point_data["mu_l"]
        dmudz = self.vtk_sampler.sampled_cloud.point_data["grad_mu_l"][:, 0]
        dmudH = self.vtk_sampler.sampled_cloud.point_data["grad_mu_l"][:, 1]
        dmudp = self.vtk_sampler.sampled_cloud.point_data["grad_mu_l"][:, 2]
        dmu = [dmudp, dmudH]
        if len(thermodynamic_input) == 3:
            dmu.append(dmudz)
        dmu = np.vstack(dmu)

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


class FluidMixture(pp.PorePyModel):

    """Mixture mixin creating the brine mixture with two components."""

    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    vtk_sampler: VTKSampler
    phase_mode: str

    """provided by :class:`~model_configuration.DriesnerBrineFlowModel´"""
  
    def get_components(self) -> Sequence[pp.FluidComponent]:
        """Setting H20 as first component in Sequence makes it the reference component.
        z_H20 will be eliminated."""
        if self.component_system == ComponentSystem.WATER:
            return pp.compositional.load_fluid_constants(["H2O"], "chemicals")
        return pp.compositional.load_fluid_constants(
            ["H2O", "NaCl"], "chemicals")

    def get_phase_configuration(
        self,
        components: Sequence[pp.Component]
    ) -> Sequence[
        tuple[
            pp.compositional.EquationOfState,
            pp.compositional.PhysicalState, str
        ]
    ]:
        # Phase EOS definitions
        eos_list = []
        phase_definitions = (LiquidDriesnerCorrelations, "liq")
        state = self.phase_mode
        eos_class, name = phase_definitions
        eos = eos_class(components)
        eos.vtk_sampler = self.vtk_sampler
        eos_list.append((eos, state, name))
        return eos_list

    def dependencies_of_phase_properties(
        self,
        phase: pp.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        z_NaCl = [
            comp.fraction
            for comp in self.fluid.components
            if comp != self.fluid.reference_component
        ]
        if len(self.fluid.components) == 1:
            return [self.pressure, self.enthalpy]
        return [self.pressure, self.enthalpy] + z_NaCl  # type:ignore[return-value]


class SecondaryEquations(LocalElimination):
    """Mixin to provide expressions for dangling variables.

    The CF framework has the following quantities always as independent variables:

    - independent phase saturations
    - partial fractions (independent since no equilibrium)
    - temperature (needs to be expressed through primary variables in this model, since
      no p-h equilibrium)

    """
    dependencies_of_phase_properties: Sequence[
        Callable[[pp.GridLikeSequence], pp.ad.Variable]
    ]
    """Defined in the Brine mixture mixin."""

    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""

    vtk_sampler: VTKSampler

    has_independent_partial_fraction: Callable[
        [pp.compositional.Component, pp.compositional.Phase], bool
    ]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""
    
    # Optional: allow this to be passed by the user for consistency with FluidMixture
    phase_mode: str  # Can be "gas", "liquid", or "two-phase"

    def temperature_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        if len(thermodynamic_dependencies) == 3:
            p, h, z_NaCl = thermodynamic_dependencies
        elif len(thermodynamic_dependencies) == 2:
            p, h = thermodynamic_dependencies
            z_NaCl = np.zeros_like(p)
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h, p)).T
        self.vtk_sampler.sample_at(par_points)

        # Overall temperature
        T = self.vtk_sampler.sampled_cloud.point_data["Temperature"] # [K]
        dTdz = self.vtk_sampler.sampled_cloud.point_data["grad_Temperature"][:, 0]
        dTdH = self.vtk_sampler.sampled_cloud.point_data["grad_Temperature"][:, 1]
        dTdp = self.vtk_sampler.sampled_cloud.point_data["grad_Temperature"][:, 2]
        dT = [dTdp, dTdH]
        if len(thermodynamic_dependencies) == 3:
            dT.append(dTdz)
        dT = np.vstack(dT)
        return T, dT

    def H2O_liq_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        if len(thermodynamic_dependencies) == 3:
            p, h, z_NaCl = thermodynamic_dependencies
        elif len(thermodynamic_dependencies) == 2:
            p, h = thermodynamic_dependencies
            z_NaCl = np.zeros_like(p)
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h, p)).T
        self.vtk_sampler.sample_at(par_points)

        # Partial fraction of water in liquid phase
        X_w = 1.0 - self.vtk_sampler.sampled_cloud.point_data["Xl"]
        dX_wdz = -self.vtk_sampler.sampled_cloud.point_data["grad_Xl"][:, 0]
        dX_wdH = -self.vtk_sampler.sampled_cloud.point_data["grad_Xl"][:, 1]
        dX_wdp = -self.vtk_sampler.sampled_cloud.point_data["grad_Xl"][:, 2]
        dX_w = [dX_wdp, dX_wdH]
        if len(thermodynamic_dependencies) == 3:
            dX_w.append(dX_wdz)
        dX_w = np.vstack(dX_w)
        return X_w, dX_w

    def H2O_gas_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(thermodynamic_dependencies) == 3:
            p, h, z_NaCl = thermodynamic_dependencies
        elif len(thermodynamic_dependencies) == 2:
            p, h = thermodynamic_dependencies
            z_NaCl = np.zeros_like(p)
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h, p)).T
        self.vtk_sampler.sample_at(par_points)

        # Partial fraction of water in gas phase
        X_w = 1.0 - self.vtk_sampler.sampled_cloud.point_data["Xv"]
        dX_wdz = -self.vtk_sampler.sampled_cloud.point_data["grad_Xv"][:, 0]
        dX_wdH = -self.vtk_sampler.sampled_cloud.point_data["grad_Xv"][:, 1]
        dX_wdp = -self.vtk_sampler.sampled_cloud.point_data["grad_Xv"][:, 2]
        dX_w = [dX_wdp, dX_wdH]
        if len(thermodynamic_dependencies) == 3:
            dX_w.append(dX_wdz)
        dX_w = np.vstack(dX_w)
        return X_w, dX_w

    def set_equations(self) -> None:
        super().set_equations()
        subdomains = self.mdg.subdomains()

        matrix = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        matrix_boundary = cast(
            pp.BoundaryGrid, self.mdg.subdomain_to_boundary_grid(matrix)
        )
        subdomains_and_matrix = subdomains + [matrix_boundary]

        chi_functions_map = {
            "H2O_liq": self.H2O_liq_func,
            "H2O_gas": self.H2O_gas_func,
        }

        # Partial fractions for any phase present in the model
        for phase in self.fluid.phases:
            # Only eliminate for components that are independent
            for comp in phase:
                if self.has_independent_partial_fraction(comp, phase):
                    func_key = f"{comp.name}_{phase.name}"
                    if func_key not in chi_functions_map:
                        raise KeyError(f"Missing constitutive law for {func_key}")
                    self.eliminate_locally(
                        phase.partial_fraction_of[comp],
                        self.dependencies_of_phase_properties(phase),
                        chi_functions_map[func_key],
                        subdomains_and_matrix,
                    )

        # Temperature: always needed, using reference phase
        ref_phase = self.fluid.reference_phase
        self.eliminate_locally(
            self.temperature,
            self.dependencies_of_phase_properties(ref_phase),
            self.temperature_func,
            subdomains,
        )
