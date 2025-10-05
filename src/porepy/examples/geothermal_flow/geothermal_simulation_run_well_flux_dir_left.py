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
from typing import Callable, cast
import time
import numpy as np

from pathlib import Path
from typing import Optional
import scipy.sparse as sps

import porepy as pp
from porepy import matrix_operations
from porepy import compositional_flow as cf


# Import model configurations: flow_model_config_pointwells_heter_frac_salt_2D
from porepy.examples.geothermal_flow.model_configuration.flow_model_config_well_flux_dir_left import (
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
use_line_search = True 

BASE_DIR = Path(__file__).resolve().parent  # This gives the path of this script's folder
VTK_DIR = BASE_DIR / "model_configuration" / "constitutive_description" / "driesner_vtk_files"


# Simulation configurations
SIMULATION_CASES = {
    "three_phase_LP_2D": {  # Low-pressure two-phase (Figure 4)
        "BC": BCLiquidPhaseLowPressure_Well_Flux_Dir_Left,
        "IC": ICLiquidPhaseLowPressure_Well_Flux_Dir_Left,
        "FlowModel": LiquidPhaseFlowModel2D,
        "tf": 100.0 * 365.0 * 86400,
        # "tf": 1 * 0.01 * 86400,
        "dt": 0.02 * 86400,
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

        def schur_complement_inverter(self) -> Callable[[sps.spmatrix], sps.spmatrix]:
            """Parallelized block diagonal inverter for local equilibrium equations,
            assuming they are defined on all subdomains in each cell."""

            def inverter(A: sps.csr_matrix) -> sps.csr_matrix:
                row_perm, col_perm, block_sizes = matrix_operations.generate_permutation_to_block_diag_matrix(A)
              
                return matrix_operations.invert_permuted_block_diag_matrix(
                    A, row_perm, col_perm, block_sizes
                )

            return inverter
        
        def after_nonlinear_convergence_main(self) -> None:
            """Print solver statistics after each nonlinear iteration."""
            super().after_nonlinear_convergence()
            print(f"Number of iterations: {self.nonlinear_solver_statistics.num_iteration}")
            print(f"Time value (years): {self.time_manager.time / (365 * 86400):.2f}")
            print(f"Time index: {self.time_manager.time_index}\n")
        
        def after_nonlinear_convergence(self):
            """Adjust timestep near breakthrough based on residuals or time."""
            super().after_nonlinear_convergence()

            # p_val = self.adaptive_production_pressure()
            # print(f"Adaptive production pressure: {p_val/1e6:.2f} MPa")
            subdomains = self.mdg.subdomains()
            well_tags = {"injection_well", "production_well"}
            fractures = [sd for sd in subdomains if sd.dim < self.nd and not well_tags.intersection(sd.tags)]
            halite_phase = [p for p in self.fluid.phases if p.name == "halite"][0]
            s_h_new_f = halite_phase.saturation(fractures).value(self.equation_system)
            # Store for next Newton iteration
            self._s_halite_prev = s_h_new_f.copy()

            # wells = [sd for sd in subdomains if well_tags.intersection(sd.tags)]
            # ap = self.aperture(fractures).value(self.equation_system)
            # poro_f = self.porosity_fracture_and_intersection(fractures).value(self.equation_system)
            # poro_w = self.porosity_wells(wells).value(self.equation_system)
            # print(f"Fracture_halite: {s_h_new_f}")
            # print(f"Fracture porosities: {poro_f}")
            # print(f"Well porosities: {poro_w}")
            # print(f"Fracture apertures: {ap}")
            print(f"Number of iterations: {self.nonlinear_solver_statistics.num_iteration}")
            print(f"Time value (years): {self.time_manager.time / (365 * 86400):.2f}")
            print(f"Time index: {self.time_manager.time_index}\n")

        def after_simulation(self):
            """Export results after the simulation."""
            self.exporter.write_pvd()

        # def compute_residual_norm(
        #     self, 
        #     residual: Optional[np.ndarray],
        #     reference_residual: np.ndarray = np.array([])
        # ) -> float:
        #     if residual is None:
        #         return np.nan
        #     residual_norm = np.linalg.norm(residual)
        #     return float(residual_norm)

        def solve_linear_system(self):
            #TODO: -------------Debugging---------------
            # print(f"Component Source for H2O: {self.component_source(self.fluid.components[0],self.mdg.subdomains()).value(self.equation_system)}\n")
            # print(f"Component Source for NaCl: {self.component_source(self.fluid.components[1],self.mdg.subdomains()).value(self.equation_system)}\n")
            # print(f"Energy Source: {self.energy_source(self.mdg.subdomains()[:1] + [self.mdg.subdomains()[2]]).value(self.equation_system)}\n")
            # TODO: The mismatch in the mapping of the equations to block indices 
            # is due to bug in the equation system. There is a quick fix to this
            # but I am lazy to that, hence the reason for the manual selection.
            # return super().solve_linear_system()
            if not use_schur_technique:
                # eq_idx_map = self.equation_system.assembled_equation_indices
                # eq_p_dof_idx = eq_idx_map['mass_balance_equation']
                # eq_T_dof_idx = eq_idx_map['elimination_of_temperature_on_grids_[0]']
                # eq_h_dof_idx = eq_idx_map['energy_balance_equation']
                # eq_p_well_index = eq_idx_map['production_pressure_constraint']
                # eq_h_injection_index = eq_idx_map['injection_enthalpy_constraint']

                _, res_g = self.linear_system
                print("Overall residual norm at x_k: ", np.linalg.norm(res_g))
                for name, eq in self.equation_system.equations.items():
                    if name not in [
                        "mass_balance_equation", 
                        "energy_balance_equation",
                        "well_enthalpy_flux_equation",
                        "component_mass_balance_equation_NaCl",
                        "elimination_of_temperature_on_grids_[0]",
                        # "well_flux_equation"
                    ]:
                        continue
                    rn = self.compute_residual_norm(
                        cast(np.ndarray, self.equation_system.evaluate(eq)),
                        reference_residual=np.array([])
                    )
                    print(f"Residual norm for {name}: {rn:.3e}")
                    # if name == "well_flux_equation":
                    #     wv = self.equation_system.evaluate(eq)
                    #     print(f"Well_fluxes: {wv}")
                print(" ")
                # if self.nonlinear_solver_statistics.num_iteration >= 98:
                #     res_energy = self.compute_residual_norm(
                #         cast(np.ndarray, self.equation_system.evaluate(
                #             self.equation_system.equations["well_enthalpy_flux_equation"]
                #         )),
                #         reference_residual=np.array([])
                #     )

                #     if res_energy > 1e-2: #and not self.convergence_status:
                #         old_dt = self.time_manager.dt
                #         new_dt = max(0.5 * old_dt, 200)
                #         if new_dt < old_dt:
                #             print(f"Reducing timestep from {old_dt:.2e} to {new_dt:.2e}")
                #             self.time_manager.dt = new_dt
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
    
    # Simulation time settings
    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
        iter_max=100,
        print_info=True
    )

    # params = {
    #     # "material_constants": material_constants,
    #     # "eliminate_reference_phase": True,
    #     # "eliminate_reference_component": True,
    #     # "time_manager": time_manager,
    #     # "prepare_simulation": False,
    #     # "reduce_linear_system": use_schur_technique,
    #     "rediscretize_darcy_flux": True,
    #     "nl_convergence_tol": np.inf,
    #     "nl_convergence_tol_res": 1.0e-3,
    #     "max_iterations": 100,
    # }
    model_params = {
        "has_time_dependent_boundary_equilibrium": False,
        "eliminate_reference_phase": True,
        "eliminate_reference_component": True,
        "reduce_linear_system": use_schur_technique,
        "material_constants": material_constants,
        "time_manager": time_manager,
        "prepare_simulation": False,
    }
    solver_params = {
        "max_iterations": 100,
        "nl_convergence_tol": np.inf,
        "nl_convergence_tol_res": 1.0e-3,
        # "linear_solver": "scipy_sparse",
        "linear_solver": "pypardiso",
        "Global_line_search": use_line_search,
        "nonlinear_solver": NewtonAndersonArmijoSolver,
        "armijo_line_search_weight": 0.95,
        "armijo_line_search_incline": 0.2,
        "armijo_line_search_max_iterations": 10,
        "Anderson_acceleration": True,
        "anderson_acceleration_depth": 3,
        "anderson_acceleration_constrained": True,
        "anderson_acceleration_regularization_parameter": 1e-3,
        "solver_statistics_file_name": "solver_statistics.json"
    }
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
    params["anderson_acceleration_dimension"] = model.equation_system.num_dofs()
    print(f"Elapsed time for preparation: {time.time() - start_time:.2f} seconds")
    print(f"Simulation prepared for total DoFs: {model.equation_system.num_dofs()}")
    print(f"Grid info: {model.mdg}")

    if use_schur_technique:
        primary_equations = cf.get_primary_equations_cf(model)
        primary_equations += [
            eq for eq in model.equation_system.equations.keys() if "flux" in eq
        ]
        primary_equations += [
            "production_pressure_constraint"
        ]
        primary_equations += [
            "injection_enthalpy_constraint"
        ]
        primary_variables = cf.get_primary_variables_cf(model)
        primary_variables += list(
            set([v.name for v in model.equation_system.variables if "flux" in v.name])
        )
        model.primary_equations = primary_equations
        model.primary_variables = primary_variables

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
correl_vtk_phz_salt = VTK_DIR / "XHP_l2_original_salt_new.vtk" # Note: "XHP_l2_original_salt_new.vtk" is my main vtk!

for case_name, config in SIMULATION_CASES.items():
    # with CallHierarchyTracer("get_variable_values", "porepy"):
    run_simulation(
        case_name,
        config,
        correl_vtk_phz=correl_vtk_phz_salt,
        correl_vtk_ptz=correl_vtk_ptz_salt
    )
