"""
Test for 3D gravitational segregation of two immiscible fluids.

This test simulates the gravitational segregation of two fluids in a 3D vertical
column, with gravity acting in the z-direction. The simulation is computationally
intensive and may take a significant amount of time to complete.

Plotting is controlled by an environment variable. To generate plots, run:
(Linux/macOS):
 export RUN_PLOTS=1
 pytest --run-skipped tests/functional/test_buoyancy_flow_benchmark.py
"""

from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any, Callable, Optional, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.interpolate import interp1d

import porepy as pp
from porepy.applications.test_utils import reference_arrays_buoyancy_discretization
from porepy.models.abstract_equations import LocalElimination
from porepy.models.compositional_flow import (
    CompositionalFlowTemplate,
    CompositionalFractionalFlowTemplate,
)
from tests.functional.setups.buoyancy_flow_model import (
    FullIterateCacheMixin,
    SchurEliminationDirectSolver,
)

_DAY = 86400.0  # seconds


def flow_template(fractional_flow: bool) -> type:
    """Return the CF flow template matching the ``fractional_flow`` flag.

    The fractional-flow template installs boundary conditions that require
    ``fractional_flow=True``, so template and flag must agree.
    """
    return (
        CompositionalFractionalFlowTemplate
        if fractional_flow
        else CompositionalFlowTemplate
    )


# Three density contrasts, each with its own Hayek et al. (2009) reference profile in
# the fixture. n_steps targets a worst-case gravity CFL of ~0.25 while dividing
# tf = 5 days exactly; the buoyant velocity is linear in delta_rho, hence the doubling.
_CASES = [
    # (rho_idx, delta_rho, n_steps)
    (0, 225.0, 304),
    (1, 450.0, 608),
    (2, 900.0, 1216),
]

# Per (rho_idx, fractional_flow): l2_tol bounds the relative L2 saturation error
# (~3% above the measured value); iters is the reference total Newton count, asserted
# as an upper bound with _ITER_MARGIN (None skips the iteration assertion).
_EXPECTED: dict[tuple[int, bool], dict[str, Any]] = {
    (0, True): {"l2_tol": 3.95e-02, "iters": 923},
    (0, False): {"l2_tol": 3.15e-02, "iters": 914},
    (1, True): {"l2_tol": 4.45e-02, "iters": 1842},
    (1, False): {"l2_tol": 3.70e-02, "iters": 1826},
    (2, True): {"l2_tol": 5.60e-02, "iters": 3698},
    (2, False): {"l2_tol": 4.95e-02, "iters": 3652},
}
_ITER_MARGIN = 1.05  # allowed slack above the reference iteration count.


def _run_buoyancy_case(
    rho_idx,
    delta_rho,
    epsilon_saturation,
    dt,
    fractional_flow,
    saturation_at_5_days,
):
    """Run one gravity-column case and compare the final saturation profile against
    the Hayek et al. (2009) analytical reference.

    Returns:
        Tuple ``(l2_norm, total_iterations)``: relative L2 saturation error and
        Newton iterations summed over all time steps.
    """
    # Check for environment variable to enable plotting
    run_plots = os.environ.get("RUN_PLOTS", "false").lower() in ("true", "1")

    # define constant phase densities
    rho_l = 1000.0
    rho_g = rho_l - delta_rho
    to_Mega = 1.0e-6

    FlowTemplate = flow_template(fractional_flow)
    # Tag identifying this run in plot filenames and printouts.
    config_tag = f"ff_{'true' if fractional_flow else 'false'}_hybrid"

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
        _sphere_radius: float = 0.005
        _sphere_centre: np.ndarray = np.array([0.0025, 0.0025, 7.0])

        def set_domain(self) -> None:
            x_length = self.units.convert_units(0.005, "m")
            y_length = self.units.convert_units(0.005, "m")
            z_length = self.units.convert_units(7.0, "m")
            box: dict[str, pp.number] = {
                "xmax": x_length,
                "ymax": y_length,
                "zmax": z_length,
            }
            self._domain = pp.Domain(box)

        def grid_type(self) -> str:
            return self.params.get("grid_type", "cartesian")

        def meshing_arguments(self) -> dict:
            cell_size = self.units.convert_units(0.005, "m")
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

    class BaseEOS(pp.compositional.EquationOfState):
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
            vals = (1.0) * np.ones(nc)
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
            rho, drho = self.rho_func(*thermodynamic_input)
            # specific enthalpy of phase
            h, dh = self.h(*thermodynamic_input)
            # dynamic viscosity of phase
            mu, dmu = self.mu_func(*thermodynamic_input)
            # thermal conductivity of phase
            kappa, dkappa = self.kappa(*thermodynamic_input)

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

    class LiquidEOS(BaseEOS):
        def rho_func(
            self,
            *thermodynamic_dependencies: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            nc = len(thermodynamic_dependencies[0])
            vals = rho_l * np.ones(nc)
            return vals, np.zeros((len(thermodynamic_dependencies), nc))

    class GasEOS(LiquidEOS):
        def rho_func(
            self,
            *thermodynamic_dependencies: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            nc = len(thermodynamic_dependencies[0])
            vals = rho_g * np.ones(nc)
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
            return [self.pressure, self.enthalpy] + z_CO2

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

            rphase = self.fluid.reference_phase
            independent_phases = [p for p in self.fluid.phases if p != rphase]

            for phase in independent_phases:
                self.eliminate_locally(
                    phase.saturation,
                    self.dependencies_of_phase_properties(phase),
                    gas_saturation_func,
                    subdomains_and_matrix,
                )

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

            self.eliminate_locally(
                self.temperature,
                self.dependencies_of_phase_properties(rphase),
                temperature_func,
                subdomains_and_matrix,
            )

    class BoundaryConditions(pp.PorePyModel):
        get_inlet_outlet_sides: Callable[
            [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
        ]

        def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
            return pp.BoundaryCondition(sd, self.dirichlet_facets(sd), "dir")

        def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
            return pp.BoundaryCondition(sd, self.dirichlet_facets(sd), "dir")

        def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
            p_top = 10.0e6 * to_Mega
            p = p_top * np.ones(boundary_grid.num_cells)
            return p

        def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
            h = np.ones(boundary_grid.num_cells)
            return h

        def bc_values_overall_fraction(
            self, component: pp.Component, boundary_grid: pp.BoundaryGrid
        ) -> np.ndarray:
            z_CO2 = np.zeros(boundary_grid.num_cells)
            return z_CO2

        def bc_values_fractional_flow_energy(self, bg: pp.BoundaryGrid) -> np.ndarray:
            return self.bc_values_enthalpy(bg)

    class InitialConditions(pp.PorePyModel):
        def initial_condition(self) -> None:
            super().initial_condition()
            liq, gas = self.fluid.phases
            for sd in self.mdg.subdomains():
                s_gas_val = self.ic_values_staturation(sd)
                x_CO2_liq_v = np.zeros_like(s_gas_val)
                x_CO2_gas_v = np.ones_like(s_gas_val)

                x_CO2_liq = liq.partial_fraction_of[self.fluid.components[1]]([sd])
                x_CO2_gas = gas.partial_fraction_of[self.fluid.components[1]]([sd])

                s_gas = gas.saturation([sd])
                self.equation_system.set_variable_values(s_gas_val, [s_gas], 0, 0)
                self.equation_system.set_variable_values(x_CO2_liq_v, [x_CO2_liq], 0, 0)
                self.equation_system.set_variable_values(x_CO2_gas_v, [x_CO2_gas], 0, 0)

        def ic_values_staturation(self, sd: pp.Grid) -> np.ndarray:
            z_v = self.ic_values_overall_fraction(self.fluid.components[1], sd)
            return (z_v * rho_l) / (z_v * rho_l + rho_g - z_v * rho_g)

        def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
            # Hydrostatic initial pressure in equilibrium with the initial mixture
            # density, anchored to p_top at the Dirichlet boundary. A uniform IC
            # would leave the gravity gradient unbalanced and stall the first step.
            p_top = 10.0e6 * to_Mega
            g_val = (
                self.units.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2") * to_Mega
            )
            zc = sd.cell_centers[2]
            n = zc.size
            if n == 0:
                return np.zeros(0)
            s_gas = self.ic_values_staturation(sd)
            rho = s_gas * rho_g + (1.0 - s_gas) * rho_l
            # Integrate rho * g downward from the top boundary to each cell centre.
            order = np.argsort(-zc)
            zc_s = zc[order]
            dz = (zc_s[0] - zc_s[-1]) / (n - 1) if n > 1 else 0.0
            seg = np.empty(n)
            seg[0] = 0.5 * dz  # top boundary to first cell centre
            seg[1:] = zc_s[:-1] - zc_s[1:]
            integ = np.cumsum(rho[order] * seg)
            p = np.empty(n)
            p[order] = p_top + g_val * integ
            return p

        def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
            h = 1.0 + self.ic_values_overall_fraction(self.fluid.components[1], sd)
            return h

        def ic_values_overall_fraction(
            self, component: pp.Component, sd: pp.Grid
        ) -> np.ndarray:
            s_init_val = 0.8
            z_init_val = (rho_g * s_init_val) / (
                rho_l * (1 - s_init_val) + rho_g * (s_init_val)
            )
            xc = sd.cell_centers.T
            z = np.where((xc[:, 2] >= 1.0) & (xc[:, 2] <= 5.0), z_init_val, 0.0)
            return z * np.ones(sd.num_cells)

    class FlowModel(
        ModelGeometry,
        FluidMixture,
        InitialConditions,
        BoundaryConditions,
        SecondaryEquations,
        FullIterateCacheMixin,
        FlowTemplate,
    ):
        _total_iterations = 0  # Newton iterations accumulated over all time steps.

        def after_nonlinear_convergence(self) -> None:
            super().after_nonlinear_convergence()
            self._total_iterations += self.nonlinear_solver_statistics.num_iterations

        def relative_permeability(
            self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
        ) -> pp.ad.Operator:
            return phase.saturation(domains) ** 2

        def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.TpfaAd:
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)

        def fourier_flux_discretization(
            self, subdomains: list[pp.Grid]
        ) -> pp.ad.TpfaAd:
            return pp.ad.TpfaAd(self.fourier_keyword, subdomains)

        def set_equations(self):
            super().set_equations()
            self.set_buoyancy_discretization_parameters()

        def set_nonlinear_discretizations(self) -> None:
            super().set_nonlinear_discretizations()
            self.set_nonlinear_buoyancy_discretization()

        def before_nonlinear_iteration(self) -> None:
            self.update_buoyancy_driven_fluxes()
            self.rediscretize()

        def gravity_field(
            self, subdomains: pp.SubdomainsOrBoundaries
        ) -> pp.ad.Operator:
            g_constant = pp.GRAVITY_ACCELERATION
            val = self.units.convert_units(g_constant, "m*s^-2") * to_Mega
            size = np.sum([g.num_cells for g in subdomains]).astype(int)
            gravity_field = pp.wrap_as_dense_ad_array(val, size=size)
            gravity_field.set_name("gravity_field")
            return gravity_field

    # --- Simulation setup and execution ---

    day = 86400
    tf = 5.0 * day
    # dt comes from the parametrization, see _CASES.
    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
        iter_max=50,
        print_info=True,
    )

    solid_constants = pp.SolidConstants(
        permeability=1.0e-13,
        porosity=0.1,
        thermal_conductivity=2.0 * to_Mega,
        density=2500.0,
        specific_heat_capacity=1000.0 * to_Mega,
    )
    material_constants = {"solid": solid_constants}
    model_params = {
        "fractional_flow": fractional_flow,
        "enable_buoyancy_effects": True,
        "material_constants": material_constants,
        "time_manager": time_manager,
        "prepare_simulation": False,
    }

    model = FlowModel(model_params)
    model.prepare_simulation()

    _, gas = model.fluid.phases
    initial_saturation = model.equation_system.evaluate(
        gas.saturation(model.mdg.subdomains())
    )

    # A Lebesgue metric strictly bounds the residual error in the mass conservation
    # equations.
    solver_params = {
        # The model was prepared manually above.
        "prepare_simulation": False,
        "nl_convergence_criteria": {
            "res_abs": pp.solvers.ResidualBasedAbsoluteCriterion(
                tol=1.0e-5, metric=pp.EquationBasedLebesgueMetric(model)
            ),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.solvers.MaxIterationsCriterion(max_iterations=50),
        },
    }

    # The local eliminations form a cell-local block; Schur-eliminating them leaves
    # the direct solve with only the primary unknowns (p, h, z).
    nonlinear_solver = pp.solvers.NewtonSolver(
        params=solver_params,
        linear_solver=SchurEliminationDirectSolver(),
    )

    print("\nTotal number of DoF: ", model.equation_system.num_dofs())
    pp.ModelRunner(model, solver_params, nonlinear_solver=nonlinear_solver).run()

    # retrieve data from fixture
    data = saturation_at_5_days[rho_idx]

    # Parse columns
    distance = data[:, 0]
    reference_saturation = data[:, 1]

    grid = model.mdg.subdomains()[0]
    distance_h = grid.cell_centers.T[:, 2]
    numerical_saturation = model.equation_system.evaluate(
        gas.saturation(model.mdg.subdomains())
    )

    # Interpolator for reference saturation at numerical distances
    interp_func = interp1d(
        distance, reference_saturation, kind="linear", fill_value="extrapolate"
    )
    ref_interp = interp_func(distance_h)

    # Compute L2 norm of the difference in saturation
    l2_norm = np.linalg.norm(ref_interp - numerical_saturation) / np.linalg.norm(
        ref_interp
    )

    if run_plots:
        print(f"Plotting enabled. Saving plot for case rho_idx={rho_idx}...")
        plt.figure(figsize=(8, 5))
        plt.plot(
            distance_h,
            initial_saturation,
            label="Initial Saturation",
            color="grey",
            linestyle="-",
            linewidth=2,
        )
        plt.plot(
            distance_h,
            ref_interp,
            label="Reference Saturation",
            color="blue",
            linewidth=2,
        )
        plt.plot(
            distance_h,
            numerical_saturation,
            label="Numerical Saturation",
            color="red",
            linestyle="--",
            linewidth=2,
        )

        plt.xlabel("Distance [m]")
        plt.ylabel("Saturation [-]")
        plt.title(
            f"Saturation Profile at $t = 5$ days for "
            f"$\\Delta \\rho = {delta_rho}$ kg/m$^3$\n"
            f"({config_tag}, {model._total_iterations} Newton iterations in total)"
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"hayek_test_comparison_case_{rho_idx}_{config_tag}.png", dpi=300)
        plt.close()
        print("Plot saved.")

    n_steps = round(tf / dt)
    total_iterations = model._total_iterations
    print(
        f"[buoyancy drho={delta_rho:.0f} {config_tag} n={n_steps} dt={dt / day:.4f}d] "
        f"L2={l2_norm:.4e}  tol={epsilon_saturation:.2e}  iters={total_iterations}  "
        f"{'PASS' if l2_norm < epsilon_saturation else 'FAIL'}"
    )
    return l2_norm, total_iterations


@pytest.mark.skipped  # reason: slow
@pytest.mark.parametrize("fractional_flow", [True, False])
@pytest.mark.parametrize("rho_idx, delta_rho, n_steps", _CASES)
def test_buoyancy_flow_benchmark(
    rho_idx, delta_rho, n_steps, fractional_flow, saturation_at_5_days
):
    """Run one density contrast for one formulation (hybrid upwinding).

    Asserts the relative L2 saturation error against the Hayek reference and, where a
    reference is pinned, the total Newton iteration count.
    """
    expected = _EXPECTED[(rho_idx, fractional_flow)]
    tag = f"case{rho_idx}_ff_{'true' if fractional_flow else 'false'}_hybrid"
    dt = 5.0 * _DAY / n_steps
    l2_norm, total_iterations = _run_buoyancy_case(
        rho_idx,
        delta_rho,
        expected["l2_tol"],
        dt,
        fractional_flow,
        saturation_at_5_days,
    )

    print(
        f"\n===== buoyancy benchmark (case {rho_idx}, delta_rho={delta_rho:.0f}) "
        "=====\n"
        f"  {tag}: L2={l2_norm:.4e} (tol {expected['l2_tol']:.2e})  "
        f"iters={int(total_iterations)} (ref {expected['iters']})"
    )

    failures: list[str] = []
    if not l2_norm < expected["l2_tol"]:
        failures.append(f"{tag}: L2={l2_norm:.4e} >= tol={expected['l2_tol']:.2e}")
    if expected["iters"] is not None:
        iter_bound = int(expected["iters"] * _ITER_MARGIN)
        if total_iterations > iter_bound:
            failures.append(
                f"{tag}: iters={int(total_iterations)} > bound={iter_bound} "
                f"(ref {expected['iters']})"
            )
    assert not failures, "buoyancy benchmark metrics out of range:\n" + "\n".join(
        failures
    )


@pytest.fixture
def saturation_at_5_days():
    return (
        reference_arrays_buoyancy_discretization.reference_values_buoyancy_benchmark()
    )
