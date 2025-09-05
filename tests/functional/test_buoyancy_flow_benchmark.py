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
import pytest
from typing import Callable, Optional, Sequence, cast, Any
import numpy as np
import porepy as pp
from scipy.interpolate import interp1d
from porepy.models.abstract_equations import LocalElimination
from porepy.models.compositional_flow import (
    CompositionalFractionalFlowTemplate as FlowTemplate,
)
from abc import abstractmethod
import matplotlib.pyplot as plt
from porepy.applications.test_utils import reference_arrays_buoyancy_discretization


@pytest.mark.skipped  # reason: slow
@pytest.mark.parametrize(
    "rho_idx, delta_rho, epsilon_saturation",
    [
        (0, 225.0, 3.85e-02),
        (1, 450.0, 4.39e-02),
        (2, 900.0, 5.61e-02),
    ],
)
def test_buoyancy_flow_benchmark(
    rho_idx, delta_rho, epsilon_saturation, saturation_at_5_days
):
    """
    Tests the gravitational segregation of two immiscible fluids.

    This test runs a 3D simulation of two-phase flow in a vertical column
    under the effect of gravity. It is parameterized to run for three different
    density contrasts between the two fluids.

    The test validates the numerical results by comparing the final saturation
    profile against a reference analytical solution from Hayek, M., et al. (2009),
    "An analytical solution for one-dimensional, two-phase, immiscible flow in a
    porous medium," published in Advances in Water Resources.

    An assertion is made on the relative L2-norm of the difference between the
    numerical and reference solutions.

    Args:
        rho_idx (int): The index for the current test case.
        delta_rho (float): The density difference between the light and heavy fluids.
        epsilon_saturation (float): The tolerance for the L2-norm assertion.
    """
    # Check for environment variable to enable plotting
    run_plots = os.environ.get("RUN_PLOTS", "false").lower() in ("true", "1")

    # define constant phase densities
    rho_l = 1000.0
    rho_g = rho_l - delta_rho
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
        _sphere_radius: float = 0.005
        _sphere_centre: np.ndarray = np.array([0.0025, 0.0025, 10.0])

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
            p_init = 10.0e6 * to_Mega
            return np.ones(sd.num_cells) * p_init

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
        FlowTemplate,
    ):
        def relative_permeability(
            self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
        ) -> pp.ad.Operator:
            return phase.saturation(domains) ** 2

        def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.TpfaAd:
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)

        def after_nonlinear_convergence(self) -> None:
            super().after_nonlinear_convergence()
            print(
                "\nNumber of iterations: ",
                self.nonlinear_solver_statistics.num_iteration,
            )
            print("Time value: ", self.time_manager.time)
            print("Time index: ", self.time_manager.time_index)
            print("")

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

        def check_convergence(
            self,
            nonlinear_increment: np.ndarray,
            residual: Optional[np.ndarray],
            reference_residual: np.ndarray,
            nl_params: dict[str, Any],
        ) -> tuple[bool, bool]:
            if self._is_nonlinear_problem():
                total_volume = 0.0
                for sd in self.mdg.subdomains():
                    total_volume += np.sum(
                        self.equation_system.evaluate(
                            self.volume_integral(pp.ad.Scalar(1), [sd], dim=1)
                        )
                    )

                nonlinear_increment_norm = self.compute_nonlinear_increment_norm(
                    nonlinear_increment
                )
                residual_norm = np.linalg.norm(residual) * total_volume
                print("Residual norm: ", residual_norm)
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
            return converged, diverged

    # --- Simulation setup and execution ---

    day = 86400
    tf = 5.0 * day
    dt = 0.02 * day
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
    params = {
        "fractional_flow": True,
        "enable_buoyancy_effects": True,
        "material_constants": material_constants,
        "time_manager": time_manager,
        "prepare_simulation": False,
        "reduce_linear_system": False,
        "nl_convergence_tol": np.inf,
        "nl_convergence_tol_res": 1.0e-8,
        "max_iterations": 25,
    }

    model = FlowModel(params)
    model.prepare_simulation()

    _, gas = model.fluid.phases
    initial_saturation = model.equation_system.evaluate(
        gas.saturation(model.mdg.subdomains())
    )

    print("\nTotal number of DoF: ", model.equation_system.num_dofs())
    pp.run_time_dependent_model(model, params)

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

    print(f"\nRelative L2 norm of saturation difference: {l2_norm:.4e}")
    assert l2_norm < epsilon_saturation, (
        f"L2 norm {l2_norm:.4e} exceeds tolerance {epsilon_saturation:.4e} "
        f"for case rho_idx={rho_idx}"
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
            f"$\\Delta \\rho = {delta_rho}$ kg/m$^3$"
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"hayek_test_comparison_case_{rho_idx}.png", dpi=300)
        plt.close()
        print("Plot saved.")


@pytest.fixture
def saturation_at_5_days():
    return (
        reference_arrays_buoyancy_discretization.reference_values_buoyancy_benchmark()
    )
