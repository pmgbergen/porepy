""" "Run script for the CO2 injection model."""

from __future__ import annotations

REFINEMENT_LEVEL: int = 3
"""Chose mesh size with h = 4 * 0.5 ** i, with i being the refinement level."""
EQUILIBRIUM_CONDITION: Literal["unified-p-T", "unified-p-h"] = "unified-p-h"
"""Define the equilibrium condition to determin the flash type used in the solution
procedure."""
FLASH_TOL_CASE: int = 2
"""Define the flash tolerance used in the solution procedure."""
LOCAL_SOLVER_STRIDE: int = 3
"""Ïnteger determining every which global iteration to start the local solver."""
NUM_MONTHS: int = 24
""""Number of months (30 days) for which to run the simulation."""
REL_PERM: Literal["quadratic", "linear"] = "linear"
"""Chocie between quadratic and linear relative permeabilities."""
RUN_WITH_SCHEDULE: bool = False
"""Ïf running without schedule for the time steps, there is no maximum admissible time
step size and no time to be hit between start and end of simulation.

This option is for obtaining the uninfluenced results of the time stepping scheme.

If True, a schedule is used at specified times for plotting purposes.

"""
DISABLE_COMPILATION: bool = False
"""For disabling numba compilation and faster start of simulation. Intended for
debugging."""
BUOYANCY_ON: bool = False
"""Turn on buoyancy. NOTE: This is still under development."""
FRACTIONAL_FLOW: bool = False
"""Use the fractional flow formulation without upwinding in the diffusive fluxes."""

MESH_SIZES: dict[int, float] = {
    0: 4.0,  # 308 cells
    1: 2.0,  # 1204 cells
    2: 1.0,  # 4636 cells
    3: 0.5,  # 18,464 cells
    # 4: 0.25,  # 73,748 cells
}
"""Tested mesh sizes in meters."""

FLASH_TOLERANCES: dict[int, float] = {
    0: 1e-1,
    1: 1e-2,
    2: 1e-3,
    3: 1e-4,
    4: 1e-5,
    5: 1e-6,
    6: 1e-7,
    7: 1e-8,
}
"""Tested flash tolerances."""

LOCAL_STRIDES: list[int] = [-1, 1, 2, 3, 4, 5]
""""Tested iteration strides for applying local solver."""

# FOLDER: str = "src/porepy/examples/cold_co2_injection/"
FOLDER: str = ""
"""For storing simulation results."""

import argparse
import json
import logging
import os
import pathlib
import time
import warnings
from typing import Any, Literal, Optional, cast

import numpy as np

if DISABLE_COMPILATION:
    os.environ["NUMBA_DISABLE_JIT"] = "1"

import porepy as pp
import porepy.models.compositional_flow_with_equilibrium as cfle
from porepy.applications.material_values.solid_values import basalt
from porepy.applications.test_utils.models import add_mixin
from porepy.examples.cold_co2_injection.model import (
    ColdCO2InjectionModel,
    ColdCO2InjectionModelFF,
)
from porepy.examples.cold_co2_injection.solver import NewtonArmijoAndersonSolver

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# mypy: ignore-errors


def get_file_name(
    condition: str,
    refinement: int,
    flash_tol_case: int = 7,
    flash_stride: int = 3,
    rel_perm: Literal["quadratic", "linear"] = "linear",
    num_months: int = 24,
) -> str:
    return (
        f"{condition}_h{refinement}_ftol{flash_tol_case}"
        f"_fstride{flash_stride}_{rel_perm}_m{num_months}"
    )


def get_path(
    condition: str,
    refinement: int,
    flash_tol_case: int = 7,
    flash_stride: int = 3,
    rel_perm: Literal["quadratic", "linear"] = "linear",
    num_months: int = 24,
    file_name: str | None = None,
) -> pathlib.Path:
    """ "Returns path to result data for a simulation case."""
    if file_name is None:
        file_name = get_file_name(
            condition, refinement, flash_tol_case, flash_stride, rel_perm, num_months
        )
        file_name = f"stats_{file_name}.json"
    return pathlib.Path(f"{FOLDER}{file_name}")


class DataCollectionMixin(pp.PorePyModel):
    """ "Collects data required for running the plot script."""

    def __init__(self, params: dict | None = None):
        super().__init__(params)

        # Data saving for plotting for paper.
        self._time_steps: list[float] = []
        self._time_step_sizes: list[float] = []
        self._time_tracker: dict[
            Literal["flash", "assembly", "linsolve"], list[float]
        ] = {
            "flash": [],
            "assembly": [],
            "linsolve": [],
        }
        self._recomputations: list[int] = []
        """Number of recomputations of dt at a time due to convergence failure."""
        self._num_global_iter: list[int] = []
        """Number of global iterations per successful time step."""
        self._num_cell_averaged_flash_iter: list[float | int] = []
        """Number of cell-averaged flash iterations per successful time step."""
        self._num_linesearch_iter: list[int] = []
        """Number of linesearch iterations per successful time step."""

        self._flash_iter_counter: int = 0
        """Counter for cell-averaged flash iterations per time step."""

        self._cum_flash_iter_per_grid: dict[pp.Grid, list[np.ndarray]] = {}
        self._flash_iter_for_cell_mean: list[np.ndarray] = []

        self._total_num_time_steps: int = 0
        self._total_num_global_iter: int = 0
        self._total_num_flash_iter: int = 0
        self.nonlinear_solver_statistics.num_iteration_armijo = 0

    def data_to_export(self):
        data: list = super().data_to_export()

        for sd in self.mdg.subdomains():
            if sd in self._cum_flash_iter_per_grid:
                ni = self._cum_flash_iter_per_grid[sd]
                n = np.array(sum(ni), dtype=int)
            else:
                n = np.zeros(sd.num_cells, dtype=int)

            data.append((sd, "cumulative flash iterations", n))

        return data

    def assemble_linear_system(self) -> None:
        start = time.time()
        super().assemble_linear_system()
        self._time_tracker["assembly"].append(time.time() - start)

    def solve_linear_system(self) -> np.ndarray:
        start = time.time()
        sol = super().solve_linear_system()
        self._time_tracker["linsolve"].append(time.time() - start)
        return sol

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()
        self._cum_flash_iter_per_grid.clear()
        self._flash_iter_counter = 0
        self.nonlinear_solver_statistics.num_iteration_armijo = 0

    def after_nonlinear_convergence(self):
        # Get data before reset and recomputation in super-call.
        self._num_global_iter.append(self.nonlinear_solver_statistics.num_iteration)
        self._num_cell_averaged_flash_iter.append(self._flash_iter_counter)
        self._num_linesearch_iter.append(
            self.nonlinear_solver_statistics.num_iteration_armijo
        )

        self._recomputations.append(self.time_manager._recomp_num)
        self._time_step_sizes.append(self.time_manager.dt)
        # NOTE the time manager always returns the time at the end of the time step,
        # The one for which we solve.
        self._time_steps.append(self.time_manager.time - self.time_manager.dt)

        self._total_num_time_steps += 1
        self._total_num_global_iter += self.nonlinear_solver_statistics.num_iteration
        self._total_num_flash_iter += sum(
            [sum(vals).sum() for vals in self._cum_flash_iter_per_grid.values()]
        )

        return super().after_nonlinear_convergence()

    def after_nonlinear_failure(self):
        # Do not include clock times of failed attempts.
        n = self.nonlinear_solver_statistics.num_iteration
        self._time_tracker["linsolve"] = self._time_tracker["linsolve"][:-n]
        self._time_tracker["assembly"] = self._time_tracker["assembly"][:-n]
        self._time_tracker["flash"] = self._time_tracker["flash"][:-n]
        self._total_num_time_steps += 1
        self._total_num_global_iter += n
        self._total_num_flash_iter += sum(
            [sum(vals).sum() for vals in self._cum_flash_iter_per_grid.values()]
        )
        return super().after_nonlinear_failure()

    def update_thermodynamic_properties_of_phases(
        self, state: Optional[np.ndarray] = None
    ) -> None:
        start = time.time()
        super().update_thermodynamic_properties_of_phases(state=state)
        self._time_tracker["flash"].append(time.time() - start)
        if self._flash_iter_for_cell_mean:
            ni = np.concatenate(self._flash_iter_for_cell_mean)
            self._flash_iter_counter += float(ni.mean())
        self._flash_iter_for_cell_mean.clear()

    def local_equilibrium(
        self,
        sd: pp.Grid,
        state: Optional[np.ndarray] = None,
        equilibrium_specs: Optional[cfle.IsobaricEquilibriumSpecs] = None,
        initial_guess_from_current_state: bool = True,
        update_secondary_variables: bool = True,
        return_num_iter: bool = False,
    ) -> None | np.ndarray:
        nfi = super().local_equilibrium(
            sd=sd,
            state=state,
            equilibrium_specs=equilibrium_specs,
            initial_guess_from_current_state=initial_guess_from_current_state,
            update_secondary_variables=update_secondary_variables,
            return_num_iter=True,
        )

        self._flash_iter_for_cell_mean.append(nfi)

        if sd not in self._cum_flash_iter_per_grid:
            self._cum_flash_iter_per_grid[sd] = []
        self._cum_flash_iter_per_grid[sd].append(nfi)

        if return_num_iter:
            return nfi
        else:
            return None


class BuoyancyModel(pp.PorePyModel):
    def initial_condition(self):
        super().initial_condition()
        self.set_buoyancy_discretization_parameters()

    def update_flux_values(self):
        super().update_flux_values()
        self.update_buoyancy_driven_fluxes()

    def set_nonlinear_discretizations(self):
        super().set_nonlinear_discretizations()
        self.set_nonlinear_buoyancy_discretization()

    def gravity_field(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        g_constant = pp.GRAVITY_ACCELERATION
        val = self.units.convert_units(g_constant, "m*s^-2")
        size = np.sum([g.num_cells for g in subdomains]).astype(int)
        gravity_field = pp.wrap_as_dense_ad_array(val, size=size)
        gravity_field.set_name("gravity_field")
        return gravity_field


class QuadraticRelPerm(pp.PorePyModel):
    """ "Contains the quadratic relative permeability law."""

    def relative_permeability(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Quadratic relative permeability model."""
        return phase.saturation(domains) ** pp.ad.Scalar(2)


if __name__ == "__main__":
    # region Argparsing
    parser = argparse.ArgumentParser(prog="Cold CO2 injection run script")
    parser.add_argument(
        "-e",
        "--equilibrium",
        nargs=1,
        default="ph",
        choices=["pT", "ph"],
        help="Local equilibrium condition. Defaults to ph",
    )
    parser.add_argument(
        "-r",
        "--refinement",
        nargs=1,
        default=REFINEMENT_LEVEL,
        choices=[_ for _ in MESH_SIZES.keys()],
        type=int,
        help="Level of mesh refinement h=4 * 0.5 * *r. Defaults to 0.",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        nargs=1,
        default=FLASH_TOL_CASE,
        choices=[_ for _ in FLASH_TOLERANCES.keys()],
        type=int,
        help="Case for tolerance of local solver. Defaults to minimal tolerance.",
    )
    parser.add_argument(
        "-s",
        "--stride",
        nargs=1,
        default=LOCAL_SOLVER_STRIDE,
        choices=LOCAL_STRIDES,
        type=int,
        help=(
            "Apply every s-th global iteration the local solver. Set to -1 to disable "
            "local solver. Defaults to 3"
        ),
    )
    parser.add_argument(
        "-m",
        "--months",
        nargs=1,
        default=NUM_MONTHS,
        type=int,
        help="Run simulation for m months (30 days). Defaults to 24.",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Run simulation with settings for 2D plot, including quadratic rel-perms.",
    )

    args = parser.parse_args()

    if args.equilibrium:
        if isinstance(args.equilibrium, list):
            a = args.equilibrium[0]
        else:
            a = args.equilibrium
        if a == "pT":
            EQUILIBRIUM_CONDITION = "unified-p-T"
        elif a == "ph":
            EQUILIBRIUM_CONDITION = "unified-p-h"
        else:
            raise ValueError(f"Unknown equilibrium condition -e {a}")
    if args.refinement:
        if isinstance(args.refinement, list):
            a = args.refinement[0]
        else:
            a = args.refinement
        REFINEMENT_LEVEL = int(a)
        assert REFINEMENT_LEVEL in list(MESH_SIZES.keys()), (
            f"Uncovered refinement level -r {a}."
        )
    if args.tolerance:
        if isinstance(args.tolerance, list):
            a = args.tolerance[0]
        else:
            a = args.tolerance
        FLASH_TOL_CASE = int(a)
        assert FLASH_TOL_CASE in list(FLASH_TOLERANCES.keys()), (
            f"Uncovered local tol case -t {a}."
        )
    if args.stride:
        if isinstance(args.stride, list):
            a = args.stride[0]
        else:
            a = args.stride
        user_stride = int(a)
        assert user_stride in LOCAL_STRIDES, f"Uncovered stride value -s {a}."
        if user_stride > 0:
            LOCAL_SOLVER_STRIDE = user_stride
        elif user_stride < 0:
            LOCAL_SOLVER_STRIDE = None
        else:
            raise ValueError(
                "Local solver stride must be positive, or -1 to disable it."
            )
    if args.months:
        if isinstance(args.months, list):
            a = args.months[0]
        else:
            a = args.months
        NUM_MONTHS = int(a)
        assert NUM_MONTHS > 0, "Number of months to simulate must be positive."

    file_name: str | None = None

    if args.plot:
        print("Configuring simulation for 2D plot.\n")
        EQUILIBRIUM_CONDITION = "unified-p-h"
        REFINEMENT_LEVEL = 3
        FLASH_TOL_CASE = 2
        LOCAL_SOLVER_STRIDE = 3
        NUM_MONTHS = 24
        REL_PERM = "linear"
        RUN_WITH_SCHEDULE = True
        file_name = "stats_ph_scheduled.json"
    else:
        print("--- start of run script ---\n")

    print(
        f"Equilibrium condition: {EQUILIBRIUM_CONDITION}\n"
        f"Refinement level: {REFINEMENT_LEVEL}\n"
        f"Local tolerance case: {FLASH_TOL_CASE}\n"
        f"Local iteration stride: {LOCAL_SOLVER_STRIDE}\n"
        f"Number of months: {NUM_MONTHS}\n"
        f"Relative permeability: {REL_PERM}\n"
        f"Time schedule: {RUN_WITH_SCHEDULE}"
    )
    print(
        f"Results stored in: {
            get_path(
                condition=EQUILIBRIUM_CONDITION,
                refinement=REFINEMENT_LEVEL,
                flash_tol_case=FLASH_TOL_CASE,
                flash_stride=LOCAL_SOLVER_STRIDE,
                rel_perm=REL_PERM,
                num_months=NUM_MONTHS,
                file_name=file_name,
            ).resolve()
        }\n"
    )

    # endregion

    h_MESH = MESH_SIZES[REFINEMENT_LEVEL]
    tol_flash = FLASH_TOLERANCES[FLASH_TOL_CASE]
    max_iterations = 30
    iter_range = (15, 25)
    if FRACTIONAL_FLOW:
        max_iterations = 40
        iter_range = (21, 35)
    if REL_PERM == "quadratic":
        max_iterations = 50
        iter_range = (36, 45)

    newton_tol = 1e-7
    newton_tol_increment = 5e-6
    dt_init = pp.DAY / 2.0

    if RUN_WITH_SCHEDULE:
        time_schedule = [i * 30 * pp.DAY for i in range(NUM_MONTHS + 1)]
        dt_max = 30 * pp.DAY
    else:
        time_schedule = [0, NUM_MONTHS * 30 * pp.DAY]
        dt_max = time_schedule[-1]

    time_manager = pp.TimeManager(
        schedule=time_schedule,
        dt_init=dt_init,
        dt_min_max=(10 * pp.MINUTE, dt_max),
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
    well_surrounding_permeability = 1e-13
    material_params = {"solid": pp.SolidConstants(**basalt_)}

    flash_params: dict[Any, Any] = {
        "mode": "parallel",
        "solver": "npipm",
        "solver_params": {
            "tolerance": tol_flash,
            "max_iterations": 80,  # 150
            "armijo_rho": 0.99,
            "armijo_kappa": 0.4,
            "armijo_max_iterations": 50 if "p-T" in EQUILIBRIUM_CONDITION else 30,
            "npipm_u1": 10,
            "npipm_u2": 10,
            "npipm_eta": 0.5,
        },
        "global_iteration_stride": LOCAL_SOLVER_STRIDE,
        "fallback_to_iterate": True,
    }
    flash_params.update(phase_property_params)

    restart_params = {
        "restart_options": {
            "restart": False,
            "pvd_file": pathlib.Path(".\\visualization\\data.pvd").resolve(),
            "is_mdg_pvd": False,
            "vtu_files": None,
            "times_file": pathlib.Path(".\\visualization\\times.json").resolve(),
        },
    }

    meshing_params = {
        "grid_type": "simplex",
        "meshing_arguments": {
            "cell_size": h_MESH,
            "cell_size_fracture": 5e-1,
        },
    }

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
        "armijo_line_search_max_iterations": 10 if REL_PERM == "linear" else 20,
        "armijo_stop_after_residual_reaches": 1e0 if REL_PERM == "linear" else 1e-2,
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
        "equilibrium_condition": EQUILIBRIUM_CONDITION,
        "eliminate_reference_phase": True,
        "eliminate_reference_component": True,
        "flash_params": flash_params,
        "fractional_flow": FRACTIONAL_FLOW,
        "material_constants": material_params,
        "time_manager": time_manager,
        "prepare_simulation": False,
        "enable_buoyancy_effects": BUOYANCY_ON,
        "compile": True,
        "flash_compiler_args": ("p-T", "p-h"),
    }

    model_params.update(phase_property_params)
    model_params.update(restart_params)
    model_params.update(meshing_params)
    model_params.update(solver_params)
    model_params["_well_surrounding_permeability"] = well_surrounding_permeability
    # Storing simulation results in individual folder.

    if file_name is None:
        data_path = get_file_name(
            condition=EQUILIBRIUM_CONDITION,
            refinement=REFINEMENT_LEVEL,
            flash_tol_case=FLASH_TOL_CASE,
            flash_stride=LOCAL_SOLVER_STRIDE,
            rel_perm=REL_PERM,
            num_months=NUM_MONTHS,
        )
    else:
        data_path = "ph_scheduled"
    model_params["folder_name"] = f"visualization/{data_path}"

    if FRACTIONAL_FLOW:
        model_class = ColdCO2InjectionModelFF
    else:
        model_class = ColdCO2InjectionModel

    model_class = add_mixin(DataCollectionMixin, model_class)

    if BUOYANCY_ON:
        model_class = add_mixin(BuoyancyModel, model_class)

    if REL_PERM == "quadratic":
        model_class = add_mixin(QuadraticRelPerm, model_class)

    model = cast(DataCollectionMixin, model_class(model_params))

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
        print(f"SIMULATION FAILED: {err}")
        n = model.nonlinear_solver_statistics.num_iteration
        model._time_tracker["linsolve"] = model._time_tracker["linsolve"][:-n]
        model._time_tracker["assembly"] = model._time_tracker["assembly"][:-n]
        model._time_tracker["flash"] = model._time_tracker["flash"][:-n]
        model._total_num_time_steps += 1
        model._total_num_global_iter += n
        model._total_num_flash_iter += sum(
            [sum(vals).sum() for vals in model._cum_flash_iter_per_grid.values()]
        )
    sim_time = time.time() - t_0

    # Dump simulation data for visualization.
    data = {
        "simulation_success": SIMULATION_SUCCESS,
        "refinement_level": int(REFINEMENT_LEVEL),
        "equilibrium_condition": str(EQUILIBRIUM_CONDITION),
        "tol_flash_case": int(FLASH_TOL_CASE),
        "num_cells": int(model.mdg.num_subdomain_cells()),
        "local_stride": int(-1 if LOCAL_SOLVER_STRIDE is None else LOCAL_SOLVER_STRIDE),
        "t": [float(_) for _ in model._time_steps],
        "dt": [float(_) for _ in model._time_step_sizes],
        "recomputations": [int(_) for _ in model._recomputations],
        "num_global_iter": [int(_) for _ in model._num_global_iter],
        "num_flash_iter": [float(_) for _ in model._num_cell_averaged_flash_iter],
        "num_linesearch_iter": [int(_) for _ in model._num_linesearch_iter],
        "clock_time_global_solver": (
            float(np.mean(model._time_tracker["linsolve"])),
            float(np.sum(model._time_tracker["linsolve"])),
        ),
        "clock_time_assembly": (
            float(np.mean(model._time_tracker["assembly"])),
            float(np.sum(model._time_tracker["assembly"])),
        ),
        "clock_time_flash_solver": (
            float(np.mean(model._time_tracker["flash"])),
            float(np.sum(model._time_tracker["flash"])),
        ),
        "setup_time": float(prep_sim_time),
        "simulation_time": float(sim_time),
        "total_num_time_steps": int(model._total_num_time_steps),
        "total_num_global_iter": int(model._total_num_global_iter),
        "total_num_flash_iter": int(model._total_num_flash_iter),
    }

    # NOTE To avoid accidently overwriting data, we do not export the one we intend
    # to visualize in 2D.
    path = get_path(
        condition=EQUILIBRIUM_CONDITION,
        refinement=REFINEMENT_LEVEL,
        flash_tol_case=FLASH_TOL_CASE,
        flash_stride=LOCAL_SOLVER_STRIDE,
        rel_perm=REL_PERM,
        num_months=NUM_MONTHS,
        file_name=file_name,
    )
    with open(
        path.resolve(),
        "w",
    ) as result_file:
        json.dump(data, result_file)
    print(f"Results saved at {str(path.resolve())}")
    print("\n--- end of run script ---")
