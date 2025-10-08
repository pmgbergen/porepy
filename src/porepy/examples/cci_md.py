"""Cold CO2 injection in 3D fractured setting."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

# os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np

import porepy as pp
import porepy.models.compositional_flow as cf
import porepy.models.compositional_flow_with_equilibrium as cfle
from porepy.applications.material_values.solid_values import basalt
from porepy.examples.cold_co2_injection.model import (
    AdjustedPointWellModel,
    BoundaryConditions,
    BuoyancyModel,
    FluidMixture,
    InitialConditions,
    PointWells,
    SolutionStrategy,
)
from porepy.examples.cold_co2_injection.solver import NewtonArmijoAndersonSolver
from porepy.examples.flow_benchmark_3d_case_4 import Geometry


class GeometryWithPointWells(PointWells, Geometry):
    """Combining Pointwell (0d-nd coupling) with benchmark geometry."""


class BoundaryConditionsBenchmark(BoundaryConditions):
    """Circle on bottom boundary plane of matrix is heated."""

    def _xy_plane_circle(self, sd: pp.Grid) -> tuple[float, float]:
        """The face centers of the subdomains which are within a certain circle
        in the x-y- plane"""

        x0 = 0.0
        y0 = 800.0
        r = 200

        x = sd.face_centers[0]
        y = sd.face_centers[0]

        return (x - x0) ** 2 + (y - y0) ** 2 <= r**2

    def _heated_boundary_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define heated boundary with D-type conditions for conductive flux."""
        sides = self.domain_boundary_sides(sd)

        heated = np.zeros(sd.num_faces, dtype=bool)
        heated[sides.bottom] = True
        circle = self._xy_plane_circle(sd)
        heated &= circle

        return heated

    def bc_values_fractional_flow_component(
        self, component: pp.Component, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        return np.zeros(bg.num_cells)

    def bc_values_fractional_flow_energy(self, bg: pp.BoundaryGrid) -> np.ndarray:

        return np.zeros(bg.num_cells)


class Permeability(pp.PorePyModel):
    """Custom permeability allowing to define fracture permeability and setting the
    permeability of any domain with Dimension nd-2 or lower to 1.."""

    total_mass_mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return pp.constitutive_laws.DimensionDependentPermeability.permeability(
            self,  # type:ignore[arg-type]
            subdomains,
        )

    def matrix_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Matrix permeability with a higher permeability with factor 1e3 around the
        wells in the matrix."""

        assert len(subdomains) <= 1, "Expecting at most 1 grid as matrix."
        if len(subdomains) == 1:
            assert subdomains[0].dim == self.nd, "Expecting only matrix as input."

        K_vals: list[np.ndarray] = [np.zeros((0,))]

        for sd in subdomains:
            k = np.ones(sd.num_cells)
            k *= self.solid.permeability
            K_vals.append(k)

        K_: pp.ad.Operator = pp.wrap_as_dense_ad_array(
            np.concatenate(K_vals), name="base_matrix_permeability"
        )

        if cf.is_fractional_flow(self):
            K_ *= self.total_mass_mobility(subdomains)

        K = self.isotropic_second_order_tensor(subdomains, K_)
        K.set_name("matrix_permeability")
        return K

    def fracture_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Set with model parameter 'fracture_permeability'."""
        N = sum([sd.num_cells for sd in subdomains])
        K_val = self.params["fracture_permeability"]
        K_: pp.ad.Operator = pp.wrap_as_dense_ad_array(
            float(K_val), size=N, name="base_well_permeability"
        )

        if cf.is_fractional_flow(self):
            K_ *= self.total_mass_mobility(subdomains)

        K = self.isotropic_second_order_tensor(subdomains, K_)
        K.set_name("well_permeability")
        return K

    def intersection_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Base permeability of wells is 1."""
        N = sum([sd.num_cells for sd in subdomains])
        K_: pp.ad.Operator = pp.wrap_as_dense_ad_array(
            1.0, size=N, name="base_well_permeability"
        )

        if cf.is_fractional_flow(self):
            K_ *= self.total_mass_mobility(subdomains)

        K = self.isotropic_second_order_tensor(subdomains, K_)
        K.set_name("well_permeability")
        return K


class BenchmarkModel(  #type:ignore
    Permeability,
    BuoyancyModel,
    GeometryWithPointWells,
    AdjustedPointWellModel,
    FluidMixture,
    InitialConditions,
    BoundaryConditionsBenchmark,
    SolutionStrategy,
    cfle.EnthalpyBasedCFFLETemplate,
):
    """'Yippee Ki Yay, motherfucker' - John McClane, Die Hard."""

    _p_INIT: float = 20e6
    _T_INIT: float = 450.0
    _z_INIT: dict[str, float] = {"H2O": 0.995, "CO2": 0.005}

    # In- and outflow values.
    _T_HEATED: float = 640.0
    _T_IN: float = 300.0
    _z_IN: dict[str, float] = {"H2O": 0.9, "CO2": 0.1}

    _p_OUT: float = _p_INIT - 2e6

    # Value obtained from a p-T flash with values defined above.
    # Divide by 3600 to obtain an injection of unit per hour
    # Multiplied by some number for how many units per hour
    _TOTAL_INJECTED_MASS: float = 5 * 27430.998956110157 / (60 * 60)  # mol / m^3
    # _TOTAL_INJECTED_MASS: float = 10 * 21202.860945350567 / (60 * 60)  # mol / m^3

    # Injection model configuration
    _T_INJECTION: dict[int, float] = {0: _T_IN}
    _p_PRODUCTION: dict[int, float] = {0: _p_OUT}

    _INJECTED_MASS: dict[str, dict[int, float]] = {
        "H2O": {0: _TOTAL_INJECTED_MASS * _z_IN["H2O"]},
        "CO2": {0: _TOTAL_INJECTED_MASS * _z_IN["CO2"]},
    }

    # Coordinates of injection and production wells in meters
    _INJECTION_POINTS: list[np.ndarray] = [np.array([200.0, 200.0, 0.0])]
    _PRODUCTION_POINTS: list[np.ndarray] = [np.array([-400, 1300.0, 400.0])]


max_iterations = 40
iter_range = (21, 35)
newton_tol = 1e-5
newton_tol_increment = 1e-5
T_end_months = 24

time_schedule = [i * 30 * pp.DAY for i in range(T_end_months + 1)]
dt_init = pp.HOUR
dt_min = 10 * pp.MINUTE
dt_max = 30 * pp.DAY

time_manager = pp.TimeManager(
    schedule=time_schedule,
    dt_init=dt_init,
    dt_min_max=(dt_min, dt_max),
    iter_max=max_iterations,
    iter_optimal_range=iter_range,
    iter_relax_factors=(0.75, 2),
    recomp_factor=0.6,
    recomp_max=10,
    print_info=True,
    rtol=0.0,
)

phase_property_params = {
    "phase_property_params": [0.0],
}

basalt_ = basalt.copy()
# basalt_["permeability"] = 1e-14
well_surrounding_permeability = 1e-13
material_params = {"solid": pp.SolidConstants(**basalt_)}

flash_params: dict[Any, Any] = {
    "mode": "parallel",
    "solver": "npipm",
    "solver_params": {
        "tolerance": 1e-3,
        "max_iterations": 80,  # 150
        "armijo_rho": 0.99,
        "armijo_kappa": 0.4,
        "armijo_max_iterations": 30,
        "npipm_u1": 10,
        "npipm_u2": 10,
        "npipm_eta": 0.5,
    },
    "global_iteration_stride": 3,
    "fallback_to_iterate": True,
}
flash_params.update(phase_property_params)

solver_params = {
    "max_iterations": max_iterations,
    "nl_convergence_tol": newton_tol_increment,
    "nl_convergence_tol_res": newton_tol,
    "apply_schur_complement_reduction": True,
    "linear_solver": "scipy_sparse",
    "nonlinear_solver": NewtonArmijoAndersonSolver,
    "armijo_line_search": True,
    "armijo_line_search_weight": 0.95,
    "armijo_line_search_incline": 0.2,
    "armijo_line_search_max_iterations": 20,
    "armijo_stop_after_residual_reaches": 1e-1,
    "appplyard_chop": 0.2,
    "anderson_acceleration": False,
    "anderson_acceleration_depth": 3,
    "anderson_acceleration_constrained": False,
    "anderson_acceleration_regularization_parameter": 1e-3,
    "anderson_start_after_residual_reaches": 1e2,
    "solver_statistics_file_name": "solver_statistics.json",
    "flag_failure_as_diverged": True,
}

model_params: dict[str, Any] = {
    "equilibrium_condition": "unified-p-h",
    "eliminate_reference_phase": True,
    "eliminate_reference_component": True,
    "flash_params": flash_params,
    "fractional_flow": True,
    "material_constants": material_params,
    "time_manager": time_manager,
    "prepare_simulation": False,
    "enable_buoyancy_effects": True,
    "compile": True,
    "flash_compiler_args": ("p-T", "p-h"),
    "_lbc_viscosity": True,
    "fracture_permeability": 1e-2,
}

model_params.update(phase_property_params)
model_params.update(solver_params)
model_params["_well_surrounding_permeability"] = well_surrounding_permeability

if __name__ == "__main__":
    model = BenchmarkModel(model_params)

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("porepy").setLevel(logging.DEBUG)
    t_0 = time.time()
    model.prepare_simulation()
    prep_sim_time = time.time() - t_0
    logging.getLogger("porepy").setLevel(logging.INFO)

    model_params["anderson_acceleration_dimension"] = model.equation_system.num_dofs()

    # Defining sub system for Schur complement reduction.
    primary_equations = cfle.cf.get_primary_equations_cf(model)
    primary_equations += [
        eq for eq in model.equation_system.equations.keys() if "flux" in eq
    ]
    primary_equations += [
        "production_pressure_constraint",
        "injection_temperature_constraint",
    ]
    primary_variables = cfle.cf.get_primary_variables_cf(model)
    primary_variables += list(
        set([v.name for v in model.equation_system.variables if "flux" in v.name])
    )

    model.schur_complement_primary_equations = primary_equations
    model.schur_complement_primary_variables = primary_variables

    t_0 = time.time()
    SIMULATION_SUCCESS: bool = True
    try:
        pp.run_time_dependent_model(model, model_params)
    except Exception as err:
        SIMULATION_SUCCESS = False
        print(f"\nSIMULATION FAILED: {err}")
        model.save_data_time_step()
    sim_time = time.time() - t_0
