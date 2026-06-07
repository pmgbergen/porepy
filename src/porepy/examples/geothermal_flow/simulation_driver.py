"""Simulation driver for the numerical benchmark and fractured-reservoir examples
in the paper titled: Mathematical Modeling of Salt Precipitation and Multi-Phase Flow
in High Enthalpy Fractured Geothermal Systems.

This module provides the command-line entry point used to run the simulation
cases. It loads a YAML configuration, constructs the appropriate PorePy model, attaches
the correlated H2O--NaCl thermodynamic samplers, runs the time-dependent
simulation, and writes the standard PorePy visualization output.

The driver supports two classes of cases:

1. The 1D benchmark case used for comparison with CSMP++ reference results.
2. The 2D fractured-reservoir examples used to study halite precipitation,
   permeability reduction, fracture clogging, and production performance.

Typical usage from ``src/porepy/examples`` is

    python -m geothermal_flow.simulation_driver --config geothermal_flow/configs/benchmark.yaml
    python -m geothermal_flow.simulation_driver --config geothermal_flow/configs/example1.yaml
    python -m geothermal_flow.simulation_driver --config geothermal_flow/configs/example2.yaml
    python -m geothermal_flow.simulation_driver --config geothermal_flow/configs/example3.yaml

For each case, visualization files are written to

    visualization/<case_name>/

The generated ``.pvd`` files are subsequently consumed by
``geothermal_flow.make_figures`` together with ParaView state files, extraction
scripts, and Matplotlib plotting scripts to reproduce the paper figures.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import porepy as pp
from porepy import compositional_flow as cf

# from porepy.applications.test_utils.models import add_mixin
from porepy.examples.geothermal_flow.solver_configuration.line_search_armijo import (
    NewtonAndersonArmijoSolver,
)

from .benchmark.flow_model import BenchmarkThreePhaseFlowModel
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler

### This requires PETSC
# import pp_solvers
# from pp_solvers.preconditioners import cf_factory_well_inj, cf_factory_no_well

from .reservoir_domain import ConnectedFracturedDomain2D, DisconnectedFracturedDomain2D
from .io_utils import as_float, load_config, is_benchmark_config
from .model import (
    FractureSolidConstants,
    NoFlowAdiabaticBoundary,
    ThreePhaseFlowModelConfiguration2D,
    UniformInitialConditions,
)

# Geometry registry for the generic 2D fractured-reservoir examples.
# The 1D benchmark is not listed here because it uses the fully assembled
# BenchmarkThreePhaseFlowModel, which already includes its own geometry,
# boundary conditions, initial conditions, and flow-model configuration.
GEOMETRIES = {
    "disconnected": DisconnectedFracturedDomain2D,
    "connected": ConnectedFracturedDomain2D,
}

VARIABLE_BLOCK_ALIASES = {
    "overall_fraction": ["z_NaCl"],
    "fraction_in_phase": ["x_NaCl_liq", "x_NaCl_halite", "x_NaCl_gas"],
    "saturation": ["s_gas", "s_halite"],
}


def build_time_manager(config: dict[str, Any]) -> pp.TimeManager:
    """Construct a PorePy TimeManager from YAML configuration."""

    t = config["time"]
    t_end = as_float(t["end"])
    return pp.TimeManager(
        schedule=[0.0, t_end],
        dt_init=as_float(t["dt_init"]),
        dt_min_max=(as_float(t["dt_min"]), as_float(t["dt_max"])),
        constant_dt=bool(t.get("constant_dt", False)),
        iter_max=int(t.get("max_iterations", 100)),
        iter_optimal_range=tuple(t.get("iter_optimal", [5, 15])),
        iter_relax_factors=tuple(t.get("iter_relax", [0.75, 1.5])),
        recomp_factor=as_float(t.get("recompute_factor", 0.5)),
        recomp_max=int(t.get("recompute_max", 10)),
        print_info=bool(t.get("print_info", True)),
        rtol=as_float(t.get("rtol", 0.0)),
    )


def build_material_constants(config: dict[str, Any]) -> dict[str, Any]:
    """Build solid material constants used by the selected geothermal model."""

    m = config["material"]

    if is_benchmark_config(config):
        solid = pp.SolidConstants(
            permeability=as_float(m["permeability"]),
            porosity=as_float(m["porosity"]),
            thermal_conductivity=as_float(m["thermal_conductivity"]),
            density=as_float(m["density"]),
            specific_heat_capacity=as_float(m["specific_heat_capacity"]),
        )
        return {"solid": solid}

    solid = FractureSolidConstants(
        residual_aperture=as_float(m["residual_aperture"]),
        permeability=as_float(m["permeability"]),
        normal_permeability=as_float(m["normal_permeability"]),
        fracture_permeability=as_float(m["fracture_permeability"]),
        porosity=as_float(m["porosity"]),
        thermal_conductivity=as_float(m["thermal_conductivity"]),
        density=as_float(m["density"]),
        specific_heat_capacity=as_float(m["specific_heat_capacity"]),
    )
    return {"solid": solid}


def create_model_class(config: dict[str, Any]) -> type[pp.PorePyModel]:
    """Create the case-specific model class from geometry, BC, IC, and physics settings."""

    if is_benchmark_config(config):
        return BenchmarkThreePhaseFlowModel

    geometry_cls = GEOMETRIES[config["geometry"]]
    solver_cfg = config["solver"]
    well_cfg = config["well"]
    physics_cfg = config["physics"]

    class GeothermalSimulationModel(
        geometry_cls,
        NoFlowAdiabaticBoundary,
        UniformInitialConditions,
        ThreePhaseFlowModelConfiguration2D,
    ):
        """Configured model assembled from geometry, BC, IC, and physics mixins."""

        _p_INIT: float = as_float(well_cfg["p_init"])
        _T_INIT: float = as_float(well_cfg["t_init"])
        _z_INIT: dict[str, float] = {
            "H2O": as_float(well_cfg["z_init"]["H2O"]),
            "NaCl": as_float(well_cfg["z_init"]["NaCl"]),
        }
        _T_INJ: float = as_float(well_cfg["t_inj"])
        _z_INJ: dict[str, float] = {
            "H2O": as_float(well_cfg["z_inj"]["H2O"]),
            "NaCl": as_float(well_cfg["z_inj"]["NaCl"]),
        }
        _INJECTION_FRACTION: float = as_float(physics_cfg["injection_fraction"])
        _p_OUT: float = as_float(well_cfg["p_out"])
        _well_radius: float = as_float(well_cfg["well_radius"])
        _fracture_aperture: float = as_float(physics_cfg["reference_aperture"])
        _aperture_clogging_exponent: float = as_float(physics_cfg["clogging_exponent"])
        _minimum_aperture: float = as_float(physics_cfg.get("minimum_aperture", 1.0e-4))

        _T_INJECTION: dict[int, float] = {0: _T_INJ}
        _p_PRODUCTION: dict[int, float] = {0: _p_OUT}
        _p_INJECTION: dict[int, float] = {0: _p_INIT}

        def after_nonlinear_convergence(self) -> None:
            super().after_nonlinear_convergence()
            if solver_cfg.get("print_nonlinear_statistics", True):
                print(
                    f"Number of iterations: {self.nonlinear_solver_statistics.num_iteration}"
                )
                print(f"Time value (days): {self.time_manager.time / pp.DAY:.4f}")
                print(f"Time index: {self.time_manager.time_index}\n")

        def get_variable_block_indices(self, var_name: str | list[str]) -> np.ndarray:
            if not isinstance(var_name, list):
                var_name = [var_name]
            if len(var_name) == 1 and var_name[0] in VARIABLE_BLOCK_ALIASES:
                var_name = VARIABLE_BLOCK_ALIASES[var_name[0]]
            return self.equation_system.dofs_of(var_name)

        def after_simulation(self) -> None:
            super().after_simulation()
            if config.get("visualization", {}).get("write_pvd", True):
                self.exporter.write_pvd()

        # def solve_linear_system(self):
        #     if solver_cfg.get("print_residual_blocks", True):
        #         if not solver_cfg.get("use_schur_complement", False):
        #             _, residual = self.linear_system
        #         else:
        #             _, residual = self.equation_system.assemble()

        #         print("Overall residual norm at x_k:", np.linalg.norm(residual))
        #         for name in [
        #             "pressure",
        #             "enthalpy",
        #             "overall_fraction",
        #             "temperature",
        #             "fraction_in_phase",
        #             "saturation",
        #             "well_flux",
        #             "well_enthalpy_flux",
        #         ]:
        #             block = self.get_variable_block_indices(name)
        #             if block.size:
        #                 print(f"Residual norm for {name} equation: {np.linalg.norm(residual[block]):.3e}")
        #         print(" ")
        #     return super().solve_linear_system()

    return GeothermalSimulationModel


def build_solver_params(config: dict[str, Any]) -> dict[str, Any]:
    """Translate YAML solver options into the parameter names expected by PorePy."""

    s = config["solver"]

    if bool(s.get("use_preconditioner", False)):
        raise NotImplementedError(
            "The Docker reproducibility workflow does not include the optional "
            "iterative preconditioner dependency. Set solver.use_preconditioner "
            "to false in the YAML configuration."
        )

    return {
        "max_iterations": int(config["time"].get("max_iterations", 100)),
        "nl_convergence_tol": np.inf,
        "nl_convergence_tol_res": as_float(s.get("nonlinear_tolerance", 1.0e-3)),
        "Global_line_search": bool(s.get("use_line_search", False)),
        "armijo_line_search": bool(s.get("use_line_search", False)),
        "nonlinear_solver": NewtonAndersonArmijoSolver,
        "flag_failure_as_diverged": bool(s.get("flag_failure_as_diverged", True)),
        "armijo_line_search_weight": as_float(s.get("armijo_weight", 0.8)),
        "armijo_line_search_incline": as_float(s.get("armijo_incline", 1.0e-2)),
        "armijo_line_search_max_iterations": int(s.get("armijo_max_iterations", 10)),
        "Anderson_acceleration": bool(s.get("use_anderson", True)),
        "anderson_acceleration_depth": int(s.get("anderson_acceleration_depth", 3)),
        "anderson_acceleration_constrained": bool(
            s.get("anderson_acceleration_constrained", False)
        ),
        "anderson_acceleration_regularization_parameter": as_float(
            s.get("anderson_acceleration_regularization_parameter", 1.0e-3)
        ),
        "appleyard_chop": bool(s.get("use_appleyard_chop", False)),
        "use_appleyard_chop": bool(s.get("use_appleyard_chop", False)),
        "appleyard_chop_value": as_float(s.get("appleyard_chop_value", 0.2)),
        "solver_statistics_file_name": str(
            s.get("solver_statistics_file_name", "solver_statistics.json")
        ),
        "use_preconditioner": False,
        "linear_solver": "pypardiso",  # preconditioner_options if use_preconditioner else s.get("linear_solver", "pypardiso"),
    }


def build_model_params(config: dict[str, Any]) -> dict[str, Any]:
    """Assemble model-construction parameters passed to the PorePy model instance."""

    params = {
        "has_time_dependent_boundary_equilibrium": False,
        "eliminate_reference_phase": True,
        "eliminate_reference_component": True,
        "apply_schur_complement_reduction": bool(
            config["solver"].get("use_schur_complement", False)
        ),
        "material_constants": build_material_constants(config),
        "enable_buoyancy_effects": False,
        "fractional_flow": bool(config["model"].get("fractional_flow", False)),
        "time_manager": build_time_manager(config),
        "prepare_simulation": False,
        "folder_name": str(visualization_dir(config)),
        "file_name": str(config["case_name"]),
    }

    if is_benchmark_config(config):
        params["boundary_conditions"] = config["boundary_conditions"]
        params["initial_conditions"] = config["initial_conditions"]
        params["relative_permeability"] = config.get("relative_permeability", {})

    return params


def visualization_dir(config: dict[str, Any]) -> Path:
    """Return the visualization output directory for the selected simulation case."""

    root = Path(config.get("visualization", {}).get("root", "visualization"))
    return root / str(config["case_name"])


def attach_vtk_samplers(model: pp.PorePyModel, config: dict) -> None:
    """Attach PHZ and PTZ VTK thermodynamic samplers to the model."""

    vtk_cfg = config["vtk"]

    base_dir = Path(__file__).resolve().parent

    vtk_dir = Path(
        vtk_cfg.get(
            "directory",
            "model_configuration/constitutive_description/driesner_vtk_files",
        )
    )

    if not vtk_dir.is_absolute():
        vtk_dir = base_dir / vtk_dir

    phz_path = vtk_dir / vtk_cfg["phz_file"]
    ptz_path = vtk_dir / vtk_cfg["ptz_file"]

    if not phz_path.exists():
        raise FileNotFoundError(f"PHZ VTK file not found: {phz_path}")

    if not ptz_path.exists():
        raise FileNotFoundError(f"PTZ VTK file not found: {ptz_path}")

    brine_vtk_sampler_phz = VTKSampler(str(phz_path))
    brine_vtk_sampler_ptz = VTKSampler(str(ptz_path))

    # Match the original example_1.py thermodynamic table scaling exactly.
    brine_vtk_sampler_phz.conversion_factors = (1.0, 1.0e-3, 1.0e-5)
    brine_vtk_sampler_ptz.conversion_factors = (1.0, 1.0, 1.0e-5)
    brine_vtk_sampler_ptz.translation_factors = (0.0, -273.15, 0.0)

    model.vtk_sampler = brine_vtk_sampler_phz
    model.vtk_sampler_ptz = brine_vtk_sampler_ptz


def configure_schur_if_requested(model: pp.PorePyModel, config: dict[str, Any]) -> None:
    """Configure Schur-complement equation and variable groups when enabled."""

    if not config["solver"].get("use_schur_complement", False):
        return

    primary_equations = cf.get_primary_equations_cf(model)
    primary_equations += [
        eq for eq in model.equation_system.equations.keys() if "flux" in eq
    ]

    if not is_benchmark_config(config):
        primary_equations += ["injection_temperature_constraint"]

    primary_variables = cf.get_primary_variables_cf(model)
    primary_variables += list(
        {v.name for v in model.equation_system.variables if "flux" in v.name}
    )

    model.schur_complement_primary_equations = primary_equations
    model.schur_complement_primary_variables = primary_variables


def print_flux_diagnostics(model: pp.PorePyModel) -> None:
    """Print inlet and outlet Darcy-flux diagnostics without affecting the simulation."""

    try:
        grid = model.mdg.subdomains()[0]
        darcy_flux = model.darcy_flux(model.mdg.subdomains()).value(
            model.equation_system
        )
        inlet_idx, outlet_idx = model.get_inlet_outlet_sides(grid)
        print(f"Inflow values: {darcy_flux[inlet_idx]}")
        print(f"Outflow values: {darcy_flux[outlet_idx]}")
    except Exception as exc:  # diagnostics must never hide a successful simulation
        print(f"Flux diagnostics skipped: {exc}")


def run(
    config_path: str | Path, defaults_path: str | Path | None = None
) -> pp.PorePyModel:
    """Run one geothermal-flow case from YAML and return the completed model."""

    config = load_config(config_path, defaults_path=defaults_path)
    vis_dir = visualization_dir(config)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning geothermal-flow case: {config['case_name']}")
    print(f"Geometry: {config['geometry']}")
    print(f"Visualization directory: {vis_dir}\n")

    model_cls = create_model_class(config)
    # if config["solver"].get("use_preconditioner", False):
    #     model_cls = add_mixin(pp_solvers.IterativeSolverMixin, model_cls)

    params = {**build_model_params(config), **build_solver_params(config)}
    model = model_cls(params)
    attach_vtk_samplers(model, config)

    start = time.time()
    model.prepare_simulation()
    params["anderson_acceleration_dimension"] = model.equation_system.num_dofs()
    configure_schur_if_requested(model, config)

    print(f"Elapsed time for preparation: {time.time() - start:.2f} seconds")
    print(f"Simulation prepared for total DoFs: {model.equation_system.num_dofs()}")
    print(f"Grid info: {model.mdg}")

    start = time.time()
    pp.run_time_dependent_model(model, params)
    print(f"Elapsed time for simulation: {time.time() - start:.2f} seconds")
    print(f"Total DoFs: {model.equation_system.num_dofs()}")
    print(f"Grid info: {model.mdg}")

    print_flux_diagnostics(model)

    return model


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for the unified driver."""

    parser = argparse.ArgumentParser(
        description="Run one geothermal-flow example from YAML."
    )
    parser.add_argument("--config", required=True, help="Path to example YAML file.")
    parser.add_argument(
        "--defaults",
        default=None,
        help="Optional path to defaults YAML. If omitted, package configs/defaults.yaml is used.",
    )
    return parser


def main() -> None:
    """Parse command-line arguments and launch the selected simulation case."""
    args = build_parser().parse_args()
    run(args.config, defaults_path=args.defaults)


if __name__ == "__main__":
    main()
