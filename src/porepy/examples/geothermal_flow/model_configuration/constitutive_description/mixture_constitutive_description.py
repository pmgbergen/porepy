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
        rho = self.vtk_sampler.sampled_could.point_data["Rho_l"]
        drhodz = self.vtk_sampler.sampled_could.point_data["grad_Rho_l"][:, 0]
        drhodH = self.vtk_sampler.sampled_could.point_data["grad_Rho_l"][:, 1]
        drhodp = self.vtk_sampler.sampled_could.point_data["grad_Rho_l"][:, 2]
        drho = [drhodp, drhodH]
        if len(thermodynamic_input) == 3:
            drho.append(drhodz)
        drho = np.vstack(drho)

        # specific enthalpy of phase
        h = self.vtk_sampler.sampled_could.point_data["H_l"]
        dhdz = self.vtk_sampler.sampled_could.point_data["grad_H_l"][:, 0]
        dhdH = self.vtk_sampler.sampled_could.point_data["grad_H_l"][:, 1]
        dhdp = self.vtk_sampler.sampled_could.point_data["grad_H_l"][:, 2]
        dh = [dhdp, dhdH]
        if len(thermodynamic_input) == 3:
            dh.append(dhdz)
        dh = np.vstack(dh)

        # dynamic viscosity of phase
        mu = self.vtk_sampler.sampled_could.point_data["mu_l"]
        dmudz = self.vtk_sampler.sampled_could.point_data["grad_mu_l"][:, 0]
        dmudH = self.vtk_sampler.sampled_could.point_data["grad_mu_l"][:, 1]
        dmudp = self.vtk_sampler.sampled_could.point_data["grad_mu_l"][:, 2]
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
   

class GasDriesnerCorrelations(pp.compositional.EquationOfState):

    @property
    def vtk_sampler(self):
        return self._vtk_sampler

    @vtk_sampler.setter
    def vtk_sampler(self, vtk_sampler):
        self._vtk_sampler = vtk_sampler

    def kappa(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        # vals = (1.0e-2) * np.ones(nc)
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
        rho = self.vtk_sampler.sampled_could.point_data["Rho_v"]
        drhodz = self.vtk_sampler.sampled_could.point_data["grad_Rho_v"][:, 0]
        drhodH = self.vtk_sampler.sampled_could.point_data["grad_Rho_v"][:, 1]
        drhodp = self.vtk_sampler.sampled_could.point_data["grad_Rho_v"][:, 2]
        drho = [drhodp, drhodH]
        if len(thermodynamic_input) == 3:
            drho.append(drhodz)
        drho = np.vstack(drho)
        
        # specific enthalpy of phase
        h = self.vtk_sampler.sampled_could.point_data["H_v"]
        dhdz = self.vtk_sampler.sampled_could.point_data["grad_H_v"][:, 0]
        dhdH = self.vtk_sampler.sampled_could.point_data["grad_H_v"][:, 1]
        dhdp = self.vtk_sampler.sampled_could.point_data["grad_H_v"][:, 2]
        dh = [dhdp, dhdH]
        if len(thermodynamic_input) == 3:
            dh.append(dhdz)
        dh = np.vstack(dh)

        # dynamic viscosity of phase
        mu = self.vtk_sampler.sampled_could.point_data["mu_v"]
        dmudz = self.vtk_sampler.sampled_could.point_data["grad_mu_v"][:, 0]
        dmudH = self.vtk_sampler.sampled_could.point_data["grad_mu_v"][:, 1]
        dmudp = self.vtk_sampler.sampled_could.point_data["grad_mu_v"][:, 2]
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


class HaliteDriesnerCorrelations(pp.compositional.EquationOfState):
    """ Class implementing the calculation of thermodynamic properties of liquid phases
        using the DriesnerCorrelation
    """
    @property
    def vtk_sampler(self):
        return self._vtk_sampler

    @vtk_sampler.setter
    def vtk_sampler(self, vtk_sampler):
        self._vtk_sampler = vtk_sampler
    
    def kappa(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: # value, jacobian

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
        ``phase_type`` indicates the phsycal type (0 - liq, 1 - gas, 2 - halite).
        ``thermodynamic_dependencies`` are as defined by the user.
        """
        if not hasattr(self, "_vtk_sampler"):
            raise AttributeError(
                "Instance of the vtk_sampler attribute is not present."
            )

        p, h, z_NaCl = thermodynamic_input
        par_points = np.array((z_NaCl, h, p)).T
        self.vtk_sampler.sample_at(par_points)
        n = len(p)  # same for all input (number of cells)

        # Mass density of phase
        rho = self.vtk_sampler.sampled_could.point_data["Rho_h"]
        drhodz = self.vtk_sampler.sampled_could.point_data["grad_Rho_h"][:, 0]
        drhodH = self.vtk_sampler.sampled_could.point_data["grad_Rho_h"][:, 1]
        drhodp = self.vtk_sampler.sampled_could.point_data["grad_Rho_h"][:, 2]
        drho = np.vstack((drhodp, drhodH, drhodz))

        # specific enthalpy of phase
        h = self.vtk_sampler.sampled_could.point_data["H_h"]
        dhdz = self.vtk_sampler.sampled_could.point_data["grad_H_h"][:, 0]
        dhdH = self.vtk_sampler.sampled_could.point_data["grad_H_h"][:, 1]
        dhdp = self.vtk_sampler.sampled_could.point_data["grad_H_h"][:, 2]
        dh = np.vstack((dhdp, dhdH, dhdz))

        # dynamic viscosity of phase. No mu_h, I need to check this out.
        num_cells = len(thermodynamic_input[0])
        mu = 1.0*np.ones(num_cells)
        dmu = np.zeros((len(thermodynamic_input), num_cells))
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
        # Phase to EoS class and string label
        phase_definitions = {
            pp.compositional.PhysicalState.liquid: (LiquidDriesnerCorrelations, "liq"),
            pp.compositional.PhysicalState.gas: (GasDriesnerCorrelations, "gas"),
            pp.compositional.PhysicalState.halite: (HaliteDriesnerCorrelations, "halite"),
        }
        eos_list = []

        # Determine active phases
        if self.phase_mode == PhaseMode.TWO_PHASE:
            active_phases = [pp.compositional.PhysicalState.liquid, pp.compositional.PhysicalState.gas]
        elif self.phase_mode == PhaseMode.THREE_PHASE:
            active_phases = [
                pp.compositional.PhysicalState.liquid,
                pp.compositional.PhysicalState.gas,
                pp.compositional.PhysicalState.halite,
            ]
        else:  # Assume single-phase: phase_mode directly equals a PhysicalState
            active_phases = [self.phase_mode]

        for state in active_phases:
            eos_class, name = phase_definitions[state]
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

    def gas_saturation_func(
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

        # Gas saturation
        S_v = self.vtk_sampler.sampled_could.point_data["S_v"]
        dS_vdz = self.vtk_sampler.sampled_could.point_data["grad_S_v"][:, 0]
        dS_vdH = self.vtk_sampler.sampled_could.point_data["grad_S_v"][:, 1]
        dS_vdp = self.vtk_sampler.sampled_could.point_data["grad_S_v"][:, 2]
        dS_v = [dS_vdp, dS_vdH]
        if len(thermodynamic_dependencies) == 3:
            dS_v.append(dS_vdz)
        dS_v = np.vstack(dS_v)

        return S_v, dS_v

    def halite_saturation_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        p, h, z_NaCl = thermodynamic_dependencies
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h, p)).T
        # par_points = copy.deepcopy(par_points_ini)
        self.vtk_sampler.sample_at(par_points)
        # Halite saturation
        S_h = self.vtk_sampler.sampled_could.point_data["S_h"]
        dS_hdz = self.vtk_sampler.sampled_could.point_data["grad_S_h"][:, 0]
        dS_hdH = self.vtk_sampler.sampled_could.point_data["grad_S_h"][:, 1]
        dS_hdp = self.vtk_sampler.sampled_could.point_data["grad_S_h"][:, 2]
        dS_h = np.vstack((dS_hdp, dS_hdH, dS_hdz))
        return S_h, dS_h

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
        T = self.vtk_sampler.sampled_could.point_data["Temperature"] # [K]
        dTdz = self.vtk_sampler.sampled_could.point_data["grad_Temperature"][:, 0]
        dTdH = self.vtk_sampler.sampled_could.point_data["grad_Temperature"][:, 1]
        dTdp = self.vtk_sampler.sampled_could.point_data["grad_Temperature"][:, 2]
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
        X_w = 1.0 - self.vtk_sampler.sampled_could.point_data["Xl"]
        dX_wdz = -self.vtk_sampler.sampled_could.point_data["grad_Xl"][:, 0]
        dX_wdH = -self.vtk_sampler.sampled_could.point_data["grad_Xl"][:, 1]
        dX_wdp = -self.vtk_sampler.sampled_could.point_data["grad_Xl"][:, 2]
        dX_w = [dX_wdp, dX_wdH]
        if len(thermodynamic_dependencies) == 3:
            dX_w.append(dX_wdz)
        dX_w = np.vstack(dX_w)
        return X_w, dX_w

    def NaCl_liq_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        p, h, z_NaCl = thermodynamic_dependencies
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h, p)).T
        self.vtk_sampler.sample_at(par_points)

        # Partial fraction of salt in liquid phase
        X_s = self.vtk_sampler.sampled_could.point_data["Xl"]
        dX_sdz = self.vtk_sampler.sampled_could.point_data["grad_Xl"][:, 0]
        dX_sdH = self.vtk_sampler.sampled_could.point_data["grad_Xl"][:, 1]
        dX_sdp = self.vtk_sampler.sampled_could.point_data["grad_Xl"][:, 2]
        dX_s = [dX_sdp, dX_sdH]
        if len(thermodynamic_dependencies) == 3:
            dX_s.append(dX_sdz)
        dX_s = np.vstack(dX_s)
        return X_s, dX_s

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
        X_w = 1.0 - self.vtk_sampler.sampled_could.point_data["Xv"]
        dX_wdz = -self.vtk_sampler.sampled_could.point_data["grad_Xv"][:, 0]
        dX_wdH = -self.vtk_sampler.sampled_could.point_data["grad_Xv"][:, 1]
        dX_wdp = -self.vtk_sampler.sampled_could.point_data["grad_Xv"][:, 2]
        dX_w = [dX_wdp, dX_wdH]
        if len(thermodynamic_dependencies) == 3:
            dX_w.append(dX_wdz)
        dX_w = np.vstack(dX_w)
        return X_w, dX_w

    def NaCl_gas_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        p, h, z_NaCl = thermodynamic_dependencies
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h, p)).T
        self.vtk_sampler.sample_at(par_points)

        # Partial fraction of salt in gas phase
        X_s = self.vtk_sampler.sampled_could.point_data["Xv"]
        dX_sdz = self.vtk_sampler.sampled_could.point_data["grad_Xv"][:, 0]
        dX_sdH = self.vtk_sampler.sampled_could.point_data["grad_Xv"][:, 1]
        dX_sdp = self.vtk_sampler.sampled_could.point_data["grad_Xv"][:, 2]
        dX_s = [dX_sdp, dX_sdH]
        if len(thermodynamic_dependencies) == 3:
            dX_s.append(dX_sdz)
        dX_s = np.vstack(dX_s)
        return X_s, dX_s
    
    # Halite phase components
    def NaCl_halite_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        p, h, z_NaCl = thermodynamic_dependencies
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h, p)).T
        self.vtk_sampler.sample_at(par_points)

        # Partial fraction of salt in halite phase. 
        # There exists only a single component (Salt) in the halite phase.
        num_cells = len(thermodynamic_dependencies[0])
        X_h = 1.0*np.ones(num_cells)
        dX_hdz = np.zeros(num_cells)
        dX_hdH = np.zeros(num_cells)
        dX_hdp = np.zeros(num_cells)
        dX_h = np.vstack((dX_hdp, dX_hdH, dX_hdz))
        return X_h, dX_h

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
        if any(comp.name == "NaCl" for comp in self.fluid.components):
            chi_functions_map["NaCl_liq"] = self.NaCl_liq_func
            chi_functions_map["NaCl_gas"] = self.NaCl_gas_func
            chi_functions_map["NaCl_halite"] = self.NaCl_halite_func

        # Saturation: only eliminate for two-phase
        # In  Saturation is 1.0, if a single liquid, gas, or supercritical fluid is present!
        if self.phase_mode == PhaseMode.TWO_PHASE:
            ref_phase = self.fluid.reference_phase
            independent_phases = [p for p in self.fluid.phases if p != ref_phase]
            for phase in independent_phases:
                self.eliminate_locally(
                    phase.saturation,
                    self.dependencies_of_phase_properties(phase),
                    self.gas_saturation_func,
                    subdomains_and_matrix,
                )

        if self.phase_mode == PhaseMode.THREE_PHASE:
            ref_phase = self.fluid.reference_phase
            independent_phases = [p for p in self.fluid.phases if p != ref_phase]
            for phase in independent_phases:
                if phase.name == "halite":
                    saturation_func = self.halite_saturation_func
                else:
                    saturation_func = self.gas_saturation_func
                self.eliminate_locally(
                    phase.saturation,
                    self.dependencies_of_phase_properties(phase),
                    saturation_func,
                    subdomains_and_matrix,
                )

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
