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
# from typing import Callable, cast
import time
import numpy as np

from pathlib import Path
from typing import Optional
# import scipy.sparse as sps

import porepy as pp
# from porepy import matrix_operations
from porepy import compositional_flow as cf
# solvers and preconditioners
import pp_solvers
from pp_solvers.preconditioners import cf_factory_well_inj
from porepy.applications.test_utils.models import add_mixin


# Import model configurations: flow_model_config_pointwells_heter_frac_salt_2D
from porepy.examples.geothermal_flow.model_configuration.new_flow_model_well_flux_dir_left import (
    LiquidPhaseFlowModelConfiguration2D as LiquidPhaseFlowModel2D,
    FractureSolidConstants,
)
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler

# Import geometric setup for the model domain
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import DomainFractureWellLeftDirichlet2D as ModelGeometry2D

from porepy.examples.geothermal_flow.solver_configuration.line_search_armijo import NewtonAndersonArmijoSolver
# Boundary & Initial Conditions
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (
    BCLiquidPhaseLowPressure_Well_Flux_Dir_Left,
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (
    ICLiquidPhaseLowPressure_Well_Flux_Dir_Left,
)
# from porepy.examples.geothermal_flow.function_trace_infr import CallHierarchyTracer

use_schur_technique = False
use_line_search = False
use_anderson = True
use_preconditioner = False

BASE_DIR = Path(__file__).resolve().parent  # This gives the path of this script's folder
VTK_DIR = BASE_DIR / "model_configuration" / "constitutive_description" / "driesner_vtk_files"


# Simulation configurations
SIMULATION_CASES = {
    "three_phase_LP_2D": {  # Low-pressure two-phase (Figure 4)
        "BC": BCLiquidPhaseLowPressure_Well_Flux_Dir_Left,
        "IC": ICLiquidPhaseLowPressure_Well_Flux_Dir_Left,
        "FlowModel": LiquidPhaseFlowModel2D,
        "tf": 100.0 * pp.YEAR,
        # "tf": 1 * 0.01 * 86400,
        "dt": 0.02 * pp.DAY,
    },
}

solid_constants = FractureSolidConstants(
    residual_aperture=1.0e-1,  # m
    permeability=1.0e-15,  # m^2
    normal_permeability=1.0e-13,  # m^2
    fracture_permeability=1.0e-13,  # m^2
    porosity=0.1,  # dimensionless
    thermal_conductivity=2.0,  # W/(m.K)
    density=2700.0,  # kg/m^3
    specific_heat_capacity=880.0,  # J/(kg.K)
)
# Define material properties
# solid_constants = pp.SolidConstants(
#     permeability=1.0e-15,  # m^2
#     porosity=0.1,  # dimensionless
#     thermal_conductivity=1.9,  # W/(m.K)
#     density=2700.0,  # kg/m^3
#     specific_heat_capacity=880.0,  # J/(kg.K)
# )
material_constants = {"solid": solid_constants}


def create_dynamic_model(BC, IC, FlowModel):
    """Create a geothermal model class with specific BC, IC, and Flow Model."""
    class GeothermalSimulationFlowModel(ModelGeometry2D, BC, IC, FlowModel):

        # def schur_complement_inverter(self) -> Callable[[sps.spmatrix], sps.spmatrix]:
        #     """Parallelized block diagonal inverter for local equilibrium equations,
        #     assuming they are defined on all subdomains in each cell."""

        #     def inverter(A: sps.csr_matrix) -> sps.csr_matrix:
        #         row_perm, col_perm, block_sizes = matrix_operations.generate_permutation_to_block_diag_matrix(A)
  
        #         return matrix_operations.invert_permuted_block_diag_matrix(
        #             A, row_perm, col_perm, block_sizes
        #         )
        #     return inverter

        def after_nonlinear_convergence(self) -> None:
            """Print solver statistics after each nonlinear iteration."""
            super().after_nonlinear_convergence()
            # self.initial_condition() # Bugg!!
            # alpha = self.ramp_factor()
            # t_now = self.time_manager.time
            # print(f"t_now: {t_now/86400:.1f} days, alpha: {alpha:.3f}\n")
            print(f"Number of iterations: {self.nonlinear_solver_statistics.num_iteration}")
            print(f"Time value (years): {self.time_manager.time / (365 * 86400):.2f}")
            print(f"Time index: {self.time_manager.time_index}\n")
        
        # def before_nonlinear_iteration(self):
        #     super().before_nonlinear_iteration()
        #     # Quick source check
        #     subdomains = list(self.mdg.subdomains())
            
        #     fluid_src = self.fluid_source(subdomains).value(self.equation_system)
        #     print(f"Fluid source sum: {fluid_src.sum():.6e} (should be ~0 if no injection)")
            
        #     for comp in self.fluid.components:
        #         comp_src = self.component_source(comp, subdomains).value(self.equation_system)
        #         print(f"Component source ({comp.name}) sum: {comp_src.sum():.6e}")
        
        def get_variable_block_indices(
            self: pp.PorePyModel,
            var_name: str
        ) -> np.ndarray:
            """Return DOF indices for one or more variable names.

            Parameters
            ----------
            var_name : str or list[str]
                The name(s) of variables defined as attributes of the model,
                e.g. 'pressure_variable' or ['pressure_variable', 'temperature_variable'].

            Returns
            -------
            np.ndarray
                The global DOF indices for the given variable(s).
                Returns an empty array if none of the names exist.
            """
            # if hasattr(self, var_name):
            if not isinstance(var_name, list):
                var_name = [var_name]
            if var_name == ["overall_fraction"]:
                var_name = ['z_NaCl']
            if var_name == ["fraction_in_phase"]:
                var_name = ['x_NaCl_liq', 'x_NaCl_halite', 'x_NaCl_gas']
            if var_name == ["saturation"]:
                var_name = ['s_gas', 's_halite']
            return self.equation_system.dofs_of(var_name)
            # return []

        def after_simulation(self):
            """Export results after the simulation."""
            self.exporter.write_pvd()

        def solve_linear_system(self):
            # TODO: -------------Debugging---------------
            _names = [
                    "pressure", "enthalpy",
                    "overall_fraction",
                    "temperature",
                    "fraction_in_phase",
                    "saturation",
                    "well_flux", 
                    "well_enthalpy_flux",
                ]
            if not use_schur_technique:
                _, res_g = self.linear_system
                print("Overall residual norm at x_k: ", np.linalg.norm(res_g))
            else:
                _, res_g = self.equation_system.assemble()
                print("Overall residual norm at x_k: ", np.linalg.norm(res_g))

            for name in _names:
                block_indx = self.get_variable_block_indices(var_name=name)
                rn = np.linalg.norm(res_g[block_indx])
                print(f"Residual norm for {name} equation: {rn:.3e}")
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

    # Simulation time settings
    # time_manager = pp.TimeManager(
    #     schedule=[0.0, tf],
    #     dt_init=dt,
    #     constant_dt=True,
    #     iter_max=100,
    #     print_info=True
    # )

    time_manager = pp.TimeManager(
        schedule=[0.0, 1.0*pp.YEAR],
        dt_init=0.01*pp.DAY,          # Start with your current dt
        dt_min_max=(0.1*pp.SECOND,  0.02*pp.DAY),  # 1 second to 30 days
        constant_dt=False,              # CRITICAL: enable adaptive stepping
        iter_max=100,
        iter_optimal_range=(5, 15),     # Target range for Newton iterations
        iter_relax_factors=(0.75, 1.5), # Conservative growth
        recomp_factor=0.5,              # Halve dt on failure
        recomp_max=10,                  # Allow retries
        print_info=True,
        rtol=0.0,
    )

    # options for linear solver and preconditioner
    preconditioner_options = {
            "preconditioner_factory": cf_factory_well_inj,  # TODO: check this out
            "options": {
                "gmres": {
                    "ksp_max_it": 300,          # 300
                    "ksp_gmres_restart": 100,   # was 100
                    "ksp_monitor": None,
                },
                "cpr_stage1_ilu": {
                    "pc_type": "hypre",
                    "pc_hypre_type": "ilu",
                    "pc_hypre_ilu_level": 2,      # was 2
                    "pc_hypre_ilu_maxiter": 10,
                },
                "cpr_stage0_identity": {
                    "pc_type": "jacobi",
                },
                "cpr_stage0_amg": {
                    "pc_hypre_boomeramg_strong_threshold": 0.5,  # was 0.25
                    "pc_hypre_boomeramg_relax_type_all": "Chebyshev",
                },
            },
        }

    model_params = {
        "has_time_dependent_boundary_equilibrium": False,
        "eliminate_reference_phase": True,
        "eliminate_reference_component": True,
        "apply_schur_complement_reduction": use_schur_technique,
        "material_constants": material_constants,
        "enable_buoyancy_effects": False,
        "time_manager": time_manager,
        "prepare_simulation": False,
    }
    solver_params = {
        "max_iterations": 100,
        "nl_convergence_tol": np.inf,
        "nl_convergence_tol_res": 1.0e-3,
        # "linear_solver": "pypardiso",
        "Global_line_search": use_line_search,
        "nonlinear_solver": NewtonAndersonArmijoSolver,
        "flag_failure_as_diverged": True,
        "armijo_line_search_weight": 0.8, # 0.95,
        "armijo_line_search_incline": 1.0e-2, # 0.2,
        "armijo_line_search_max_iterations": 10,
        "Anderson_acceleration": use_anderson,
        "anderson_acceleration_depth": 3,
        "anderson_acceleration_constrained": use_anderson,
        "anderson_acceleration_regularization_parameter": 1e-3,
        "solver_statistics_file_name": "solver_statistics.json",
        "use_preconditioner": use_preconditioner,
        "linear_solver": preconditioner_options if use_preconditioner else "pypardiso"
    }
    # solver_params = {
    #     "max_iterations": 100,
    #     "nl_convergence_tol": np.inf,
    #     "nl_convergence_tol_res": 1.0e-3,
    #     # "linear_solver": "scipy_sparse",
    #     "linear_solver": "pypardiso",
    #     "Global_line_search": use_line_search,
    #     "nonlinear_solver": NewtonAndersonArmijoSolver,
    #     "armijo_line_search": True,
    #     "armijo_line_search_weight": 0.95,
    #     "armijo_line_search_incline": 0.2,
    #     "armijo_line_search_max_iterations": 20,
    #     "armijo_stop_after_residual_reaches": 1e0,
    #     "appplyard_chop": 0.2,
    #     "Anderson_acceleration": True,
    #     "anderson_acceleration_depth": 2,
    #     "anderson_acceleration_constrained": True,
    #     "anderson_acceleration_regularization_parameter": 1e-3,
    #     "solver_statistics_file_name": "solver_statistics.json",
    #     "flag_failure_as_diverged": True,
    # }
    params = {**model_params, **solver_params}
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

    # total dofs
    DOFs = []
    eq_system = model.equation_system
    for name, _ in eq_system.equations.items():
        # Each equation has associated subdomains — check their total DOFs
        image_info = eq_system._equation_image_space_composition[name]
        total_dofs = sum(len(indices) for indices in image_info.values())
        DOFs.append(total_dofs)
    tot = sum(DOFs)
    print(f"Total DOFs calculated from equation system: {tot}")
    
    params["anderson_acceleration_dimension"] = model.equation_system.num_dofs()
    print(f"Elapsed time for preparation: {time.time() - start_time:.2f} seconds")
    print(f"Simulation prepared for total DoFs: {model.equation_system.num_dofs()}")
    print(f"Grid info: {model.mdg}")

    if use_schur_technique:
        primary_equations = cf.get_primary_equations_cf(model)
        primary_equations += [
            eq for eq in model.equation_system.equations.keys() if "flux" in eq
        ]
        # primary_equations += [
        #     "production_pressure_constraint"
        # ]
        primary_equations += [
            "injection_temperature_constraint"
        ]
        primary_variables = cf.get_primary_variables_cf(model)
        primary_variables += list(
            set([v.name for v in model.equation_system.variables if "flux" in v.name])
        )
        # model.primary_equations = primary_equations
        # model.primary_variables = primary_variables
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

    # Get the last time step's solution data
    # pvd_file = "./visualization/data.pvd"
    # mesh = data_util.get_last_mesh_from_pvd(pvd_file)

# ------------------------------------------------------
# Run Simulations for All Configured Cases
# ------------------------------------------------------


# Define file paths for VTK files used for thermodynamic property sampling
correl_vtk_phz_1 = VTK_DIR / "XHP_l2_original_sc.vtk"
correl_vtk_phz_2 = VTK_DIR / "XHP_l2_original_all.vtk"
correl_vtk_phz_3 = VTK_DIR / "XHP_l2_original.vtk"
correl_vtk_ptz_salt = VTK_DIR / "XTP_l2_original_salt_new.vtk"
correl_vtk_phz_salt = VTK_DIR / "XHP_l2_original_salt_new.vtk"  # Note: "XHP_l2_original_salt_new.vtk" is my main vtk!

for case_name, config in SIMULATION_CASES.items():
    run_simulation(
        case_name,
        config,
        correl_vtk_phz=correl_vtk_phz_salt,
        correl_vtk_ptz=correl_vtk_ptz_salt
    )
