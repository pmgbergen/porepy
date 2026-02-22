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
# import scipy.sparse as sps

# Import model configurations
from porepy.examples.geothermal_flow.model_configuration.flow_model_configuration import (
    SinglePhaseFlowModelConfigurationVapor as VaporPhaseModel,
    SinglePhaseFlowModelConfigurationLiquid as LiquidPhaseModel,
    TwoPhaseFlowModelConfiguration as TwoPhaseFlowModel,
    ThreePhaseFlowModelConfiguration as ThreePhaseFlowModel,
)

from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler
import porepy.examples.geothermal_flow.data_extractor_util as data_util

# Import geometric setup for the model domain
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import SimpleGeometryHorizontal as ModelGeometry

# Boundary & Initial Conditions
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (
    BCSinglePhaseHighPressure, 
    BCSinglePhaseModeratePressure,
    BCSinglePhaseLowPressure,
    BCTwoPhaseHighPressure, 
    BCTwoPhaseLowPressure,
    BCThreePhaseLowPressure,
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (
    ICSinglePhaseHighPressure,
    ICSinglePhaseModeratePressure,
    ICSinglePhaseLowPressure,
    ICTwoPhaseHighPressure,
    ICTwoPhaseLowPressure,
    ICThreePhaseLowPressure,
)

use_schur_technique = False
use_preconditioner = False

BASE_DIR = Path(__file__).resolve().parent  # This gives the path of this script's folder
VTK_DIR = BASE_DIR / "model_configuration" / "constitutive_description" / "driesner_vtk_files"

# Simulation configurations
SIMULATION_CASES = {
    # "single_phase_HP": {    # High-pressure single-phase (Figure 2, Case 1)
    #     "BC": BCSinglePhaseHighPressure,
    #     "IC": ICSinglePhaseHighPressure,
    #     "FlowModel": LiquidPhaseModel,
    #     "tf": 250 * 365 * 86400,
    #     "dt": 365 * 86400,
    # },
    # "single_phase_MP": {  # Moderate-pressure (Supercritical) single-phase (Figure 2, Case 2) # for now works with 1.0e-3
    #     "BC": BCSinglePhaseModeratePressure,
    #     "IC": ICSinglePhaseModeratePressure,
    #     "FlowModel": TwoPhaseFlowModel, # Works with TwophaseFlowModel, and 1e-4 instead of 1e-3 before
    #     "tf": 120 * 365 * 86400,  # 120 years
    #     "dt": 365 * 86400,  # 1 years 
    # },
    # "single_phase_LP": {  # Low-pressure single-phase (Figure 2, Case 3)
    #     "BC": BCSinglePhaseLowPressure,
    #     "IC": ICSinglePhaseLowPressure,
    #     "FlowModel": VaporPhaseModel,
    #     "tf": 1500 * 365 * 86400,
    #     "dt": 365 * 86400,
    # },
    # "two_phase_HP": {  # Low-pressure two-phase (Figure 4)
    #     "BC": BCTwoPhaseHighPressure,
    #     "IC": ICTwoPhaseHighPressure,
    #     "FlowModel": TwoPhaseFlowModel,
    #     "tf": 200.0 * 365.0 * 86400,
    #     "dt": 200.0 * 86400,
    # },
    # "two_phase_LP": {  # Low-pressure two-phase (Figure 4)
    #     "BC": BCTwoPhaseLowPressure,
    #     "IC": ICTwoPhaseLowPressure,
    #     "FlowModel": TwoPhaseFlowModel,
    #     "tf": 2000.0 * 365.0 * 86400,
    #     "dt": 365.0 * 86400,
    # },
    "three_phase_LP": {  # Low-pressure two-phase (Figure 4)
        "BC": BCThreePhaseLowPressure,
        "IC": ICThreePhaseLowPressure,
        "FlowModel": ThreePhaseFlowModel,
        "tf": 2000.0 * 365.0 * 86400,
        "dt": 200.0 * 86400,
    }
}

# Define material properties
solid_constants = pp.SolidConstants(
    permeability=1.0e-15,  # m^2
    porosity=0.1,  # dimensionless
    thermal_conductivity=1.9,  # W/(m.K)
    density=2700.0,  # kg/m^3
    specific_heat_capacity=880.0,  # J/(kg.K)
)

material_constants = {"solid": solid_constants}


def create_dynamic_model(BC, IC, FlowModel):
    """Create a geothermal model class with specific BC, IC, and Flow Model."""
    class GeothermalSimulationFlowModel(ModelGeometry, BC, IC, FlowModel):
        def compute_residual_norm_old(
            self, residual: Optional[np.ndarray], reference_residual: np.ndarray
        ) -> float:
            if residual is None:
                return np.nan
            residual_norm = np.linalg.norm(residual)
            return float(residual_norm)

        def after_nonlinear_convergence(self) -> None:
            """Print solver statistics after each nonlinear iteration."""
            super().after_nonlinear_convergence()
            print(f"Number of iterations: {self.nonlinear_solver_statistics.num_iteration}")
            print(f"Time value (years): {self.time_manager.time / (365 * 86400):.2f}")
            print(f"Time index: {self.time_manager.time_index}\n")

        def after_simulation(self):
            """Export results after the simulation."""
            self.exporter.write_pvd()

        def solve_linear_system(self):
            # TODO: The mismatch in the mapping of the equations to block indices 
            # is due to bug in the equation system. There is a quick fix to this
            # but I am lazy to that, hence the reason for the manual selection.
            # return super().solve_linear_system()
            if use_schur_technique:
                return super().solve_linear_system()
            eq_idx_map = self.equation_system.assembled_equation_indices
            eq_p_dof_idx = eq_idx_map['mass_balance_equation']
            eq_T_dof_idx = eq_idx_map['component_mass_balance_equation_NaCl']
            eq_h_dof_idx = eq_idx_map['energy_balance_equation']
            eq_t_dof_idx = eq_idx_map['elimination_of_temperature_on_grids_[0]']
            eq_s_dof_idx = eq_idx_map['elimination_of_s_gas_on_grids_[0]']
            eq_xs_v_dof_idx = eq_idx_map['elimination_of_x_NaCl_liq_on_grids_[0]']
            eq_xs_l_dof_idx = eq_idx_map['elimination_of_x_NaCl_gas_on_grids_[0]']
            eq_z_dof_idx = eq_idx_map['elimination_of_s_halite_on_grids_[0]']

            jac_g, res_g = self.linear_system
            print("Overall residual norm at x_k: ", np.linalg.norm(res_g))
            print("Pressure residual norm: ", np.linalg.norm(res_g[eq_p_dof_idx]))
            print("Composition residual norm: ", np.linalg.norm(res_g[eq_z_dof_idx]))
            print("Enthalpy residual norm: ", np.linalg.norm(res_g[eq_h_dof_idx]))
            print("Temperature residual norm: ", np.linalg.norm(res_g[eq_T_dof_idx]))
            print("Halite Saturation residual norm: ", np.linalg.norm(res_g[eq_s_dof_idx]))
            print("Xs_v residual norm: ", np.linalg.norm(res_g[eq_xs_v_dof_idx]))
            print("Xs_l residual norm: ", np.linalg.norm(res_g[eq_xs_l_dof_idx]))
            print(" ")
            return super().solve_linear_system()
    
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
    # time_manager = pp.TimeManager(
    #     schedule=[0.0, tf],
    #     dt_init=dt,
    #     constant_dt=True,
    #     iter_max=100,
    #     print_info=True
    # )
    
    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=100*pp.DAY,          # Start with your current dt
        dt_min_max=(1.0*pp.HOUR,  100 * pp.DAY),  # 1 second to 30 days
        constant_dt=False,              # CRITICAL: enable adaptive stepping
        iter_max=100,
        iter_optimal_range=(5, 15),     # Target range for Newton iterations
        iter_relax_factors=(0.75, 1.5), # Conservative growth
        recomp_factor=0.5,              # Halve dt on failure
        recomp_max=10,                  # Allow retries
        print_info=True,
        rtol=0.0,
    )

    params = {
        "material_constants": material_constants,
        "eliminate_reference_phase": True,
        "eliminate_reference_component": True,
        "time_manager": time_manager,
        "prepare_simulation": False,
        "enable_buoyancy_effects": False,  # NOTE: This must always be disabled!!!
        "apply_schur_complement_reduction": use_schur_technique,
        "flag_failure_as_diverged": False,
        "rediscretize_darcy_flux": True,
        "nl_convergence_tol": np.inf,
        "nl_convergence_tol_res": 1.0e-3,
        "max_iterations": 100,
        "use_preconditioner": use_preconditioner,
        "linear_solver": preconditioner_options if use_preconditioner else "pypardiso",
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
    print(f"Elapsed time for preparation: {time.time() - start_time:.2f} seconds")
    print(f"Simulation prepared for total DoFs: {model.equation_system.num_dofs()}")
    print(f"Grid info: {model.mdg}")
    
    # Defining sub system for Schur complement reduction.
    primary_equations = cf.get_primary_equations_cf(model)
    primary_variables = cf.get_primary_variables_cf(model)

    primary_equations += [
        eq for eq in model.equation_system.equations.keys() if "flux" in eq
    ]
    # primary_equations += [
    #     "production_pressure_constraint",
    #     "injection_temperature_constraint",
    # ]
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
correl_vtk_phz_1 = VTK_DIR / "XHP_l2_original_sc.vtk"
correl_vtk_phz_2 = VTK_DIR / "XHP_l2_original_all.vtk"
correl_vtk_phz_3 = VTK_DIR / "XHP_l2_original.vtk"
correl_vtk_ptz_salt = VTK_DIR / "XTP_l2_original_salt_new.vtk"
correl_vtk_phz_salt = VTK_DIR / "XHP_l2_original_salt_new.vtk"

for case_name, config in SIMULATION_CASES.items():
    if case_name in {'single_phase_MP'}:
        run_simulation(case_name, config, correl_vtk_phz=correl_vtk_phz_1)
    elif case_name in {'two_phase_HP'}:
        run_simulation(case_name, config, correl_vtk_phz=correl_vtk_phz_3)
    elif case_name in {'three_phase_LP'}:
        run_simulation(
            case_name,
            config,
            correl_vtk_phz=correl_vtk_phz_salt,
            correl_vtk_ptz=correl_vtk_ptz_salt
        )
    else:
        run_simulation(case_name, config, correl_vtk_phz=correl_vtk_phz_2)









### Some useful hooks for debugging and ensuring physical consistency during the nonlinear iterations.
# def after_nonlinear_iteration(self, solution_increment):
#     super().after_nonlinear_iteration(solution_increment)
    
#     # Clamp saturations to [0, 1]
#     for sat_name in ["s_gas", "s_halite", "s_liquid"]:  # adjust names as needed
#         try:
#             s = self.equation_system.get_variable_values(sat_name)
#             s_clamped = np.clip(s, 0.0, 1.0)
#             if not np.allclose(s, s_clamped):
#                 print(f"Clamped {sat_name}: [{s.min():.2e}, {s.max():.2e}] → [0, 1]")
#                 self.equation_system.set_variable_values(s_clamped, sat_name)
#         except KeyError:
#             pass
    
#     # Clamp composition to [0, 1]
#     for z_name in ["z_NaCl", "z_H2O"]:  # adjust names as needed
#         try:
#             z = self.equation_system.get_variable_values(z_name)
#             z_clamped = np.clip(z, 0.0, 1.0)
#             if not np.allclose(z, z_clamped):
#                 print(f"Clamped {z_name}: [{z.min():.2e}, {z.max():.2e}] → [0, 1]")
#                 self.equation_system.set_variable_values(z_clamped, z_name)
#         except KeyError:
#             pass