from __future__ import annotations
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

from typing import Callable, Optional, Sequence, cast, Any
import numpy as np
import logging
import time
import scipy.sparse as sps
from porepy.fracs.fracture_network_3d import FractureNetwork3d
import porepy as pp
from porepy.models.abstract_equations import LocalElimination
from porepy.models.compositional_flow import (
    CompositionalFractionalFlowTemplate as FlowTemplate,
)
from abc import abstractmethod

# Configure logging to show info messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Ensure specific loggers are enabled for linear solver information
logging.getLogger('porepy.models.solution_strategy').setLevel(logging.INFO)
logging.getLogger('porepy').setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# test parameters
expected_order_loss = 3
mesh_2d_Q = True


residual_tolerance = 10.0 ** (-expected_order_loss)

# define constant phase densities
rho_l = 1000.0
rho_g = 500.0
h_l = 1.0
h_g = 2.0
to_Mega = 1.0e-6


# geometry description
class Geometry(pp.PorePyModel):
    @abstractmethod
    def dirichlet_facets(self, sd: pp.Grid | pp.BoundaryGrid) -> tuple[np.ndarray]:
        pass

    @staticmethod
    def harvest_sphere_members(xc, rc, x):
        dx = x - xc
        r = np.linalg.norm(dx, axis=1)
        return np.where(r < rc, True, False)


class ModelGeometry(Geometry):
    _sphere_radius: float = 0.025
    _sphere_centre: np.ndarray = np.array([2.5, 5.0, 0.0])

    def set_domain(self) -> None:
        x_length = self.units.convert_units(5.0, "m")
        y_length = self.units.convert_units(5.0, "m")
        box: dict[str, pp.number] = {"xmax": x_length, "ymax": y_length}
        self._domain = pp.Domain(box)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.025, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def set_fractures(self) -> None:
        points = np.array(
            [
                [1.0, 2.0],
                [4.0, 2.0],
                [1.0, 2.0],
                [1.0, 4.0],
                [4.0, 2.0],
                [4.0, 4.0],
                [2.0, 1.0],
                [2.0, 4.0],
                [3.0, 1.0],
                [3.0, 4.0],
            ]
        ).T
        fracs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]).T
        self._fractures = pp.frac_utils.pts_edges_to_linefractures(points, fracs)

    def dirichlet_facets(self, sd: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        if isinstance(sd, pp.Grid):
            face_centers = sd.face_centers.T
        elif isinstance(sd, pp.BoundaryGrid):
            face_centers = sd.cell_centers.T
        else:
            raise ValueError("Type not expected.")
        boundary_faces = self.domain_boundary_sides(sd)
        bf_indices = boundary_faces.all_bf

        def find_facets(center: np.ndarray) -> np.ndarray:
            logical = Geometry.harvest_sphere_members(
                center, self._sphere_radius, face_centers[bf_indices]
            )
            return bf_indices[logical]

        return find_facets(self._sphere_centre)


class ModelGeometry3D(Geometry):
    _sphere_radius: float = 0.5
    _sphere_centre: np.ndarray = np.array([2.5, 2.5, 5.0])

    def set_domain(self) -> None:
        x_length = self.units.convert_units(5.0, "m")
        y_length = self.units.convert_units(5.0, "m")
        z_length = self.units.convert_units(5.0, "m")
        box: dict[str, pp.number] = {
            "xmax": x_length,
            "ymax": y_length,
            "zmax": z_length,
        }
        self._domain = pp.Domain(box)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(1.0, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def dirichlet_facets(self, sd: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        if isinstance(sd, pp.Grid):
            face_centers = sd.face_centers.T
        elif isinstance(sd, pp.BoundaryGrid):
            face_centers = sd.cell_centers.T
        else:
            raise ValueError("Type not expected.")
        boundary_faces = self.domain_boundary_sides(sd)
        bf_indices = boundary_faces.all_bf

        def find_facets(center: np.ndarray) -> np.ndarray:
            logical = Geometry.harvest_sphere_members(
                center, self._sphere_radius, face_centers[bf_indices]
            )
            return bf_indices[logical]

        return find_facets(self._sphere_centre)

    def set_fractures(self) -> None:


        kind_1_square_u = np.array([1.0, 1.0, 4.0, 4.0])
        kind_1_square_v = np.array([1.0, 4.0, 4.0, 1.0])

        kind_2_square_u = np.array([2.0, 2.0, 4.0, 4.0])
        kind_2_square_v = np.array([2.0, 4.0, 4.0, 2.0])

        # normal along z from z = 2.0
        f1 = np.vstack([kind_1_square_u, kind_1_square_v, np.full(4, 2.0)])

        # normal along y from y = 1.0
        f2 = np.vstack([kind_1_square_u,  np.full(4, 1.0), kind_1_square_v])

        # normal along y from y = 4.0
        f3 = np.vstack([kind_1_square_u,  np.full(4, 4.0), kind_1_square_v])

        # normal along y from y = 3.0
        f4 = np.vstack([kind_1_square_u, np.full(4, 3.0), kind_1_square_v])

        # normal along x from x = 2.0
        f5 = np.vstack([np.full(4, 2.0), kind_2_square_u, kind_2_square_v])

        disjoint_set = [f1,f2,f3,f4,f5]
        self._fractures = [pp.PlaneFracture(p) for p in disjoint_set]

# constitutive description
def gas_saturation_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])
    vals = (z_CO2 * rho_l) / (z_CO2 * rho_l + rho_g - z_CO2 * rho_g)
    vals = np.clip(vals, 1.0e-16, 1.0)

    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = (rho_l * rho_g) / (
        (z_CO2 * (rho_l - rho_g) + rho_g) * (z_CO2 * (rho_l - rho_g) + rho_g)
    )
    return vals, diffs


def temperature_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])

    factor = 0.0
    vals = np.array(h) * factor
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[1, :] = 1.0 * factor
    return vals, diffs


def CO2_liq_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CO2)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


def CO2_gas_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])
    vals = np.ones_like(z_CO2)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


chi_functions_map = {
    "CO2_liq": CO2_liq_func,
    "CO2_gas": CO2_gas_func,
}


class LiquidEOS(pp.compositional.EquationOfState):
    def rho_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = rho_l * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def mu_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = (1.0e-3) * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def h(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = h_l * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def kappa(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = (2.0) * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def compute_phase_properties(
        self,
        phase_state: pp.compositional.PhysicalState,
        *thermodynamic_input: np.ndarray,
        params: Optional[Sequence[np.ndarray | float]] = None,
    ) -> pp.compositional.PhaseProperties:
        nc = len(thermodynamic_input[0])
        # mass density of phase
        rho, drho = self.rho_func(*thermodynamic_input)  # (n,), (3, n) array
        # specific enthalpy of phase
        h, dh = self.h(*thermodynamic_input)  # (n,), (3, n) array
        # dynamic viscosity of phase
        mu, dmu = self.mu_func(*thermodynamic_input)  # (n,), (3, n) array
        # thermal conductivity of phase
        kappa, dkappa = self.kappa(*thermodynamic_input)  # (n,), (3, n) array

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
            phis=np.empty((2, nc)),
            dphis=np.empty((2, 3, nc)),
        )


class GasEOS(LiquidEOS):
    def rho_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = rho_g * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def h(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = h_g * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))


class FluidMixture(pp.PorePyModel):
    def get_components(self) -> Sequence[pp.FluidComponent]:
        return pp.compositional.load_fluid_constants(["H2O", "CO2"], "chemicals")

    def get_phase_configuration(
        self, components: Sequence[pp.Component]
    ) -> Sequence[
        tuple[pp.compositional.EquationOfState, pp.compositional.PhysicalState, str]
    ]:
        eos_L = LiquidEOS(components)
        eos_G = GasEOS(components)
        configuration_L = (pp.compositional.PhysicalState.liquid, "liq", eos_L)
        configuration_G = (pp.compositional.PhysicalState.gas, "gas", eos_G)
        return [configuration_L, configuration_G]

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
    dependencies_of_phase_properties: Callable[
        ..., Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]
    ]
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]

    def set_equations(self) -> None:
        super().set_equations()
        subdomains = self.mdg.subdomains()

        matrix = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        matrix_boundary = cast(
            pp.BoundaryGrid, self.mdg.subdomain_to_boundary_grid(matrix)
        )
        subdomains_and_matrix = subdomains + [matrix_boundary]

        # liquid phase is dependent
        rphase = self.fluid.reference_phase
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


# model description
class BoundaryConditions(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self.dirichlet_facets(sd), "dir")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self.dirichlet_facets(sd), "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        # Match the value used in buoyancy_flow_model.py
        p_top = 10.0e6 * to_Mega
        return np.ones(boundary_grid.num_cells) * p_top

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        h_inlet = np.zeros(boundary_grid.num_cells)
        return h_inlet

    def bc_values_overall_fraction(
        self, component: pp.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        z_CO2 = np.zeros(boundary_grid.num_cells)
        return z_CO2

    def bc_values_fractional_flow_energy(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return self.bc_values_enthalpy(bg)

class InitialConditions(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def initial_condition(self) -> None:
        super().initial_condition()

        # set the values to be the custom functions
        liq, gas = self.fluid.phases
        for sd in self.mdg.subdomains():
            s_gas_val = self.ic_values_saturation(sd)
            x_CO2_liq_v = np.zeros_like(s_gas_val)
            x_CO2_gas_v = np.ones_like(s_gas_val)

            x_CO2_liq = liq.partial_fraction_of[self.fluid.components[1]]([sd])
            x_CO2_gas = gas.partial_fraction_of[self.fluid.components[1]]([sd])

            s_gas = gas.saturation([sd])
            self.equation_system.set_variable_values(s_gas_val, [s_gas], 0, 0)
            self.equation_system.set_variable_values(x_CO2_liq_v, [x_CO2_liq], 0, 0)
            self.equation_system.set_variable_values(x_CO2_gas_v, [x_CO2_gas], 0, 0)

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_init = 10.0e6 * to_Mega
        return np.ones(sd.num_cells) * p_init

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        # Compute initial enthalpy as a weighted average of phase enthalpies and densities
        ic_sg = self.ic_values_saturation(sd)
        ic_rho = rho_g * ic_sg + rho_l * (1.0 - ic_sg)
        h = (ic_sg * h_g * rho_g + (1.0 - ic_sg) * h_l * rho_l) / ic_rho
        return h

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        T = 0.0 * self.ic_values_enthalpy(sd)
        return T

    def ic_values_saturation(self, sd: pp.Grid) -> np.ndarray:
        # Match the logic from buoyancy_flow_model.py
        z_v = self.ic_values_overall_fraction(self.fluid.components[1], sd)
        return (z_v * rho_l) / (z_v * rho_l + rho_g - z_v * rho_g)

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        xc = sd.cell_centers.T
        # z = (
        #     np.where((xc[:, 1] >= 1.0) & (xc[:, 1] <= 2.0), 0.5, 0.0)
        #     + np.where((xc[:, 1] >= 3.0) & (xc[:, 1] <= 4.0), 0.5, 0.0)
        #     + np.where((xc[:, 0] >= 1.0) & (xc[:, 0] <= 2.0), 0.5, 0.0)
        #     + np.where((xc[:, 0] >= 3.0) & (xc[:, 0] <= 4.0), 0.5, 0.0)
        # )
        z = np.where((xc[:, 1] >= 0.0) & (xc[:, 1] <= 2.5), 0.33333333333333326, 0.33333333333333326)
        # z = np.where((xc[:, 2] >= 0.0) & (xc[:, 2] <= 2.5), 0.95, 0.05)
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)


class FlowModel(
    ModelGeometry if mesh_2d_Q else ModelGeometry3D,
    FluidMixture,
    InitialConditions,
    BoundaryConditions,
    SecondaryEquations,
    FlowTemplate,
):

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.TpfaAd:
        return pp.ad.TpfaAd(self.darcy_keyword, subdomains)

    def fourier_flux_discretization(
        self, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.TpfaAd:
        return pp.ad.TpfaAd(self.fourier_keyword, list(subdomains))

    def relative_permeability(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        return phase.saturation(domains)**2

    def assemble_linear_system(self) -> None:
        """Custom assemble linear system that updates Jacobian every 0, 3, 6, 9... Newton iterations.

        This method implements a dedicated solution strategy that:
        - Assembles the full linear system (Jacobian + residual) at iterations 0, 3, 6, 9, etc.
        - Updates only the residual part for other iterations (1, 2, 4, 5, 7, 8, etc.)
        """
        t_0 = time.time()

        # Get current Newton iteration number
        iteration_num = self.nonlinear_solver_statistics.num_iteration

        if iteration_num % 3 == 0 or iteration_num < 3:
            # Update both Jacobian and residual at iterations 0, 3, 6, 9, ...
            logger.info(f"Newton iteration {iteration_num}: Updating Jacobian and residual")
            self.linear_system = self.equation_system.assemble(evaluate_jacobian=True)
        else:
            # Update only residual at iterations 1, 2, 4, 5, 7, 8, ...
            logger.info(f"Newton iteration {iteration_num}: Updating residual only")
            if hasattr(self, 'linear_system') and self.linear_system is not None:
                # Keep the existing Jacobian, update only the residual
                new_residual = self.equation_system.assemble(evaluate_jacobian=False)
                # Update the residual part of the linear system (tuple format: (matrix, rhs))
                self.linear_system = (
                    self.linear_system[0],  # Keep existing Jacobian
                    -new_residual  # Update residual with new evaluation
                )
            else:
                # Fallback: if no previous linear system exists, assemble full system
                logger.warning("No previous linear system found, assembling full system")
                if self._apply_schur_complement_reduction():
                    assert self.schur_complement_primary_variables, (
                        "Primary column block for Schur technique not found."
                    )
                    assert self.schur_complement_primary_equations, (
                        "Primary row block for Schur technique not defined."
                    )
                    self.linear_system = self.equation_system.assemble_schur_complement_system(
                        self.schur_complement_primary_equations,
                        self.schur_complement_primary_variables,
                        inverter=cast(
                            Callable[[sps.spmatrix], sps.spmatrix],
                            self.params.get("schur_complement_inverter", None),
                        ),
                    )
                else:
                    self.linear_system = self.equation_system.assemble()

        t_1 = time.time()
        logger.debug(f"Assembled linear system in {t_1 - t_0:.2e} seconds.")

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()
        phases = list(self.fluid.phases)
        components = list(self.fluid.components)
        # Buoyancy reciprocity check
        flux_buoyancy_c0 = self.component_buoyancy(components[0], self.mdg.subdomains())
        flux_buoyancy_c1 = self.component_buoyancy(components[1], self.mdg.subdomains())
        b_c0 = self.equation_system.evaluate(flux_buoyancy_c0)
        b_c1 = self.equation_system.evaluate(flux_buoyancy_c1)
        assert np.all(np.isclose(b_c0 + b_c1, 0.0))

        # Reference and numerical integrals
        ref_rho = ref_rho_z = ref_energy = 0.0
        num_rho = num_rho_z = num_energy = 0.0
        for sd in self.mdg.subdomains():
            ic_sg_val = self.ic_values_saturation(sd)
            rho_l = phases[0].density([sd])
            rho_g_phase = phases[1].density([sd])
            ic_rho = pp.wrap_as_dense_ad_array(1.0 - ic_sg_val) * rho_l + pp.wrap_as_dense_ad_array(ic_sg_val) * rho_g_phase
            ref_rho += self.norm_vol_int(ic_rho, sd)

            ic_z_val = self.ic_values_overall_fraction(self.fluid.components[1], sd)
            ic_rho_z = ic_rho * pp.wrap_as_dense_ad_array(ic_z_val)
            ref_rho_z += self.norm_vol_int(ic_rho_z, sd)

            ic_p_val = self.ic_values_pressure(sd)
            ic_h_val = self.ic_values_enthalpy(sd)
            ic_energy = ic_rho * pp.wrap_as_dense_ad_array(ic_h_val) - pp.wrap_as_dense_ad_array(ic_p_val)
            ref_energy += self.norm_vol_int(ic_energy, sd)

            cur_rho = self.fluid.density([sd])
            num_rho += self.norm_vol_int(cur_rho, sd)

            cur_rho_z = cur_rho * components[1].fraction([sd])
            num_rho_z += self.norm_vol_int(cur_rho_z, sd)

            cur_energy = cur_rho * self.enthalpy([sd]) - self.pressure([sd])
            num_energy += self.norm_vol_int(cur_energy, sd)

        def order(loss: float) -> float:
            return np.inf if loss <= 0.0 else abs(np.floor(np.log10(loss)))

        expected = getattr(self, "expected_order_loss", expected_order_loss)
        mass_loss = abs(ref_rho - num_rho)
        z_mass_loss = abs(ref_rho_z - num_rho_z)
        energy_loss = abs(ref_energy - num_energy)
        # Diagnostic prints for conservation checks
        print("ref mass integral: ", ref_rho)
        print("num mass integral: ", num_rho)
        print("Order of mass loss: ", order(mass_loss))
        print("ref z-mass integral: ", ref_rho_z)
        print("num z-mass integral: ", num_rho_z)
        print("Order of z-mass loss: ", order(z_mass_loss))
        print("ref energy integral: ", ref_energy)
        print("num energy integral: ", num_energy)
        print("Order of energy loss: ", order(energy_loss))
        # Boolean checks with messages

        mass_conservative_Q = order(mass_loss) >= expected
        print("buoyancy discretization is mass conservative Q: ", mass_conservative_Q)
        z_mass_conservative_Q = order(z_mass_loss) >= expected
        print("buoyancy discretization is z mass conservative Q: ", z_mass_conservative_Q)
        energy_conservative_Q = order(energy_loss) >= expected
        print("buoyancy discretization is energy conservative Q: ", energy_conservative_Q)
        # assert mass_conservative_Q
        # assert z_mass_conservative_Q
        # assert energy_conservative_Q
        print("")
        print("")

    def norm_vol_int(self, op: pp.ad.Operator, sd: pp.Grid) -> float:
        # Global volume normalization (sum over all subdomains) as in buoyancy_flow_model.py
        total_volume = 0.0
        for g in self.mdg.subdomains():
            total_volume += np.sum(
                self.equation_system.evaluate(
                    self.volume_integral(pp.ad.Scalar(1), [g], dim=1)
                )
            )
        return (
            np.sum(
                self.equation_system.evaluate(
                    self.volume_integral(op, [sd], dim=1)
                )
            )
            / total_volume
        )

    def set_equations(self):
        super().set_equations()
        self.set_buoyancy_discretization_parameters()

    def set_nonlinear_discretizations(self) -> None:
        super().set_nonlinear_discretizations()
        self.set_nonlinear_buoyancy_discretization()

    def before_nonlinear_iteration(self) -> None:
        self.update_buoyancy_driven_fluxes()
        self.rediscretize()

    def gravity_field(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        g_constant = pp.GRAVITY_ACCELERATION
        val = self.units.convert_units(g_constant, "m*s^-2") * to_Mega
        size = np.sum([g.num_cells for g in subdomains]).astype(int)
        gravity_field = pp.wrap_as_dense_ad_array(val, size=size)
        gravity_field.set_name("gravity_field")
        return gravity_field

    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: Optional[np.ndarray],
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[bool, bool]:
        if self._is_nonlinear_problem():

            total_volume = 0.0
            for sd in model.mdg.subdomains():
                total_volume += np.sum(
                    self.equation_system.evaluate(self.volume_integral(pp.ad.Scalar(1), [sd], dim=1)))

            # nonlinear_increment based norm
            nonlinear_increment_norm = self.compute_nonlinear_increment_norm(
                nonlinear_increment
            )

            # Residual per subsystem
            residual_norm = np.linalg.norm(residual) * total_volume
            # Check convergence requiring both the increment and residual to be small.
            converged_inc = (
                nl_params["nl_convergence_tol"] is np.inf
                or nonlinear_increment_norm < nl_params["nl_convergence_tol"]
            )
            converged_res = (
                nl_params["nl_convergence_tol_res"] is np.inf
                or residual_norm < nl_params["nl_convergence_tol_res"]
            )
            converged = converged_inc and converged_res
            diverged = False
        else:
            raise ValueError(
                "Gravitational segregation is nonlinear in its simpler form."
            )
        print("residual norm: ", residual_norm)
        return converged, diverged

    def set_geometry(self) -> None:
        """Define geometry and create a mixed-dimensional grid.

        The default values provided in set_domain, set_fractures, grid_type and
        meshing_arguments produce a 2d unit square domain with no fractures and a
        four Cartesian cells.

        """
        # Create the geometry through domain amd fracture set.
        self.set_domain()
        self.set_fractures()
        # Create a fracture network and a mixed-dimensional grid.
        self.create_fracture_network()
        self.create_mdg()

        self.nd: int = self.mdg.dim_max()

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)

        self.set_well_network()
        if len(self.well_network.wells) > 0:
            # Compute intersections
            assert isinstance(self.fracture_network, FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            # Mesh wells and add fracture + intersection grids to mixed-dimensional
            # grid along with these grids' new interfaces to fractures.
            self.well_network.mesh(self.mdg)

        apply_distortion_Q = False
        if apply_distortion_Q:
            for grid in self.mdg.subdomains():
                xc = grid.nodes.T
                x1 = 0.25 * np.sin(2.0 * np.pi * xc[:, 0] / 5) * np.sin(2.0 * np.pi * xc[:, 1] / 5)
                x2 = 0.25 * np.sin(2.0 * np.pi * xc[:, 1] / 5) * np.sin(2.0 * np.pi * xc[:, 0] / 5)
                xc[:, 0] += x1
                xc[:, 1] += x2
                grid.compute_geometry()

day = 86400
t_scale = 1.0
tf = 100.0 * day
dt = 0.5 * day
time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    permeability=1.0e-14,
    porosity=0.1,
    thermal_conductivity=2.0 * to_Mega,
    density=2500.0,
    specific_heat_capacity=1000.0 * to_Mega,
)
material_constants = {"solid": solid_constants}
params = {
    "fractional_flow": True,
    "enable_buoyancy_effects": True,
    "material_constants": material_constants,
    "time_manager": time_manager,
    "prepare_simulation": False,
    "apply_schur_complement_reduction": False,
    "nl_convergence_tol": np.inf,
    "nl_convergence_tol_res": residual_tolerance,
    "flag_failure_as_diverged": False,
    "max_iterations": 100,
}

model = FlowModel(params)

model.prepare_simulation()

# Print number of cells and DOFs
total_cells = sum(sd.num_cells for sd in model.mdg.subdomains())
total_dofs = model.equation_system.num_dofs()
print(f"Number of cells: {total_cells}")
print(f"Number of DOFs: {total_dofs}")

pp.run_time_dependent_model(model, params)
