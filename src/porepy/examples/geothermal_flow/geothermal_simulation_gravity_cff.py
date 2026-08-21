"""
Python script runs and visualizes 1D high-enthalpy geothermal compositional
flow simulations using PorePy.

Simulation cases include:
  - Single-phase (high, moderate, low pressure)
  - Two-phase (high, low pressure)
  - Three-phase (low pressure)

This script:
  - Creates geothermal models with appropriate BC and IC
  - Loads precomputed thermodynamic data on a discrete parametric phz- and pTz-spaces from VTK files
  - Runs time-dependent simulations using a unified compositional flow  model in Porepy
  - Generates and saves simulation results compared with CSMP++ reference data (Weis et al., DOI: 10.1111/gfl.12080).

"""

from __future__ import annotations
import time
import numpy as np
import porepy as pp
import os
from pathlib import Path
from typing import Callable, Optional
import scipy.sparse as sps
from porepy import compositional_flow as cf
# solvers and preconditioners
import pp_solvers
from pp_solvers.preconditioners import cf_factory_no_well
from porepy.applications.test_utils.models import add_mixin
from porepy.examples.geothermal_flow.solver_configuration.line_search_armijo import NewtonAndersonArmijoSolver
# import scipy.sparse as sps

# Import model configurations
from porepy.examples.geothermal_flow.model_configuration.flow_model_configuration_cff import (
    TwoPhaseFlowModelConfiguration as TwoPhaseFlowModel,
)

from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler

# Import geometric setup for the model domain
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import SimpleGeometryVertical as ModelGeometry

# Boundary & Initial Conditions
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (
    BCTwoPhaseHighPressure_Gravity_CFF,
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (
    ICTwoPhaseHighPressure,
)
use_schur_technique = False
use_line_search = False
use_anderson = False
use_preconditioner = False
use_anderson_acceleration = False
appleyard_chop = False
appleyard_chop_value = 0.2

BASE_DIR = Path(__file__).resolve().parent  # This gives the path of this script's folder
VTK_DIR = BASE_DIR / "model_configuration" / "constitutive_description" / "driesner_vtk_files"

# Simulation configurations
SIMULATION_CASES = {
    "two_phase_HP": {  # Low-pressure two-phase (Figure 4)
        "BC": BCTwoPhaseHighPressure_Gravity_CFF,
        "IC": ICTwoPhaseHighPressure,
        "FlowModel": TwoPhaseFlowModel,
        "tf": 1000.0 * 365.0 * 86400,
        "dt": 365.0 * 86400,
    },
}

# Define material properties
solid_constants = pp.SolidConstants(
    permeability=1.0e-15,  # m^2
    porosity=0.1,  # dimensionless
    thermal_conductivity=2.0,  # W/(m.K)
    density=2700.0,  # kg/m^3
    specific_heat_capacity=880.0,  # J/(kg.K)
)

material_constants = {"solid": solid_constants}

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

def create_dynamic_model(BC, IC, FlowModel):
    """Create a geothermal model class with specific BC, IC, and Flow Model."""
    class GeothermalSimulationFlowModel(BuoyancyModel, ModelGeometry, BC, IC, FlowModel):
        def after_nonlinear_convergence(self) -> None:
            """Print solver statistics after each nonlinear iteration."""

            super().after_nonlinear_convergence()

            print(f"Number of iterations: {self.nonlinear_solver_statistics.num_iterations}")
            print(f"Time value (years): {self.time_manager.time / (365 * 86400):.2f}")
            print(f"Time index: {self.time_manager.time_index}\n")

        def after_simulation(self):
            """Export results after the simulation."""
            self.exporter.write_pvd()
        
    return GeothermalSimulationFlowModel


def run_simulation(
    case_name: str,
    config: dict[str, any],
    correl_vtk_phz: str,
    correl_vtk_ptz: Optional[str] = None,
):

    """
    Run a simulation based on the provided configuration.

    Args:
        case_name (str): Name of the simulation case.
        config (dict): Dictionary containing BC, IC, Flow Model, and simulation time settings.
        correl_vtk_phz (str): Path to the VTK file for phase/fluid mixture thermodynamic property sampling.

    The function loads the model, prepares the simulation, 
    runs it, and plot the results, which are then saved in the same directory as the script.
    """
    print(f"\n Running simulation: {case_name}")  
    BC, IC, FlowModel = config["BC"], config["IC"], config["FlowModel"]
    tf, dt = config["tf"], config["dt"]

    # Create dynamic model
    GeothermalModel = create_dynamic_model(BC, IC, FlowModel)

    # Add iterative solver mixin to the model for linear solving capabilities
    if use_preconditioner:
        GeothermalModel = add_mixin(
            pp_solvers.IterativeSolverMixin, GeothermalModel
        )
    
    # options for linear solver and preconditioner
    preconditioner_options = {
        "preconditioner_factory": cf_factory_no_well,
        "options": {
            "gmres": {
                "ksp_max_it": 300,
                "ksp_gmres_restart": 100,
                "ksp_monitor": None,
            },
            "cpr_stage1_ilu": {
                "pc_type": "hypre",
                "pc_hypre_type": "ilu",
                "pc_hypre_ilu_level": 2,
                "pc_hypre_ilu_maxiter": 10,
            },
            "cpr_stage0_identity": {
                "pc_type": "jacobi",
            },
            "cpr_stage0_amg": {
                "pc_hypre_boomeramg_strong_threshold": 0.25,
                "pc_hypre_boomeramg_relax_type_all": "Chebyshev",
            },
        },
    }
    
    # Simulation time settings
    time_manager = pp.TimeManager(
        schedule=[0.0, tf],  # Schedule for time steps (start and end)
        dt_init=dt,          # Start with your current dt
        dt_min_max=(50.0*pp.DAY,  dt),  # 1 second to 30 days
        constant_dt=False,              # CRITICAL: enable adaptive stepping
        iter_max=100,
        iter_optimal_range= (20, 25),  #(5, 15),     # Target range for Newton iterations
        iter_relax_factors=(0.75, 2.0), # Conservative growth
        recomp_factor=0.5,              # Halve dt on failure
        recomp_max=10,                  # Allow retries
        print_info=True,
        rtol=0.0,
    )

    params = {
        "material_constants": material_constants,
        "has_time_dependent_boundary_equilibrium": False,
        "eliminate_reference_phase": True,
        "eliminate_reference_component": True,
        "fractional_flow": True,
        "time_manager": time_manager,
        "prepare_simulation": False,
        "enable_buoyancy_effects": True,  # NOTE: This must always be disabled!!!
        "apply_schur_complement_reduction": use_schur_technique,
        "rediscretize_darcy_flux": True,
        "nl_convergence_inc_atol": 1.0e8,
        "nl_convergence_res_atol": 1.0e-3,
        "nl_convergence_inc_rtol": 1.0e8,
        "nl_divergence_inc_atol": 1e8,
        "nl_divergence_res_atol": 1e8,
        "nl_max_iterations": 100,
        "use_preconditioner": use_preconditioner,
        "linear_solver":  preconditioner_options if use_preconditioner else "pypardiso",
        # Solver settings
        "Global_line_search": use_line_search,
        "armijo_line_search": use_line_search,
        "nonlinear_solver": NewtonAndersonArmijoSolver,
        "flag_failure_as_diverged": True,
        "armijo_line_search_weight": 0.8, # 0.95,
        "armijo_line_search_incline": 1.0e-2, # 0.2,
        "appleyard_chop": appleyard_chop,
        "appleyard_chop_value":appleyard_chop_value,
        "armijo_line_search_max_iterations": 10,
        "Anderson_acceleration": use_anderson, # was use_anderson,
        "anderson_acceleration_depth": 3,
        "anderson_acceleration_constrained": use_anderson_acceleration, # was use_anderson,
        "anderson_acceleration_regularization_parameter": 1e-3,
        "solver_statistics_file_name": "solver_statistics.json",
    }

    # Initialize model
    model = GeothermalModel(params)
    # Load VTK files
    if correl_vtk_ptz is None:
        correl_vtk_ptz = VTK_DIR / "XTP_l2_original.vtk"
    brine_vtk_sampler_phz = VTKSampler(correl_vtk_phz)
    brine_vtk_sampler_phz.conversion_factors = (1.0, 1.0e-3, 1.0e-5) # (z,h,p)
    model.vtk_sampler = brine_vtk_sampler_phz
    brine_vtk_sampler_ptz = VTKSampler(correl_vtk_ptz)
    brine_vtk_sampler_ptz.conversion_factors = (1.0, 1.0, 1.0e-5)  # (z,t,p)
    brine_vtk_sampler_ptz.translation_factors = (0.0, -273.15, 0.0)  # (z,t,p)
    model.vtk_sampler_ptz = brine_vtk_sampler_ptz

    # Prepare and run simulation
    start_time = time.time()
    model.prepare_simulation()
    params["anderson_acceleration_dimension"] = model.equation_system.num_dofs()
    print(f"Elapsed time for preparation: {time.time() - start_time:.2f} seconds")
    print(f"Simulation prepared for total DoFs: {model.equation_system.num_dofs()}")
    print(f"Grid info: {model.mdg}")
    
    # Defining sub system for Schur complement reduction.
    primary_equations = cf.get_primary_equations_cf(model)
    primary_variables = cf.get_primary_variables_cf(model)

    primary_equations += [
        eq for eq in model.equation_system.equations.keys() if "flux" in eq
    ]
    primary_variables += list(
        set([v.name for v in model.equation_system.variables if "flux" in v.name])
    )
    model.schur_complement_primary_equations = primary_equations
    model.schur_complement_primary_variables = primary_variables

    # Export geometry
    # model.exporter.write_vtu()
    start_time = time.time()

    # Run the simulation
    pp.run_time_dependent_model(model, params)
    # pp.ModelRunner(
    #     model,
    #     params
    # ).run()

    print(f"Elapsed time for simulation: {time.time() - start_time:.2f} seconds")
    print(f"Total DoFs: {model.equation_system.num_dofs()}")
    print(f"Grid info: {model.mdg}")

    # Retrieve grid and boundary info
    grid = model.mdg.subdomains()[0]

    # Compute mass flux
    darcy_flux = model.darcy_flux(model.mdg.subdomains()).value(model.equation_system)
    inlet_idx, outlet_idx = model.get_inlet_outlet_sides(grid)
    print(f"Inflow values: {darcy_flux[inlet_idx]}")
    print(f"Outflow values: {darcy_flux[outlet_idx]}")

# ------------------------------------------------------
# Run Simulations for All Configured Cases
# ------------------------------------------------------


# Define file paths for VTK files used for thermodynamic property sampling
correl_vtk_ptz_salt = VTK_DIR / "XTP_l2_original_salt_new.vtk"
correl_vtk_phz_salt = VTK_DIR / "XHP_l2_original_salt_new.vtk"
correl_vtk_phz_3 = VTK_DIR / "XHP_l2_original.vtk"

for case_name, config in SIMULATION_CASES.items():
    if case_name == "two_phase_HP":
        run_simulation(
            case_name,
            config,
            correl_vtk_phz=correl_vtk_phz_3,
            correl_vtk_ptz=correl_vtk_ptz_salt
        )