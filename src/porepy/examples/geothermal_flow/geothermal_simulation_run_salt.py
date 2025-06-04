"""
Python script runs and visualizes 1D high-enthalpy geothermal compositional
flow simulations using PorePy.

Simulation cases include:
  - Single-phase (high, moderate, low pressure)
  - Two-phase (high, low pressure)

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
# import scipy.sparse as sps

# Import model configurations
from porepy.examples.geothermal_flow.model_configuration.flow_model_configuration import (
    ThreePhaseFlowModelConfiguration as ThreePhaseFlowModel,
)
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler
import porepy.examples.geothermal_flow.data_extractor_util as data_util

# Import geometric setup for the model domain
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import SimpleGeometryHorizontal as ModelGeometry

# Boundary & Initial Conditions
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (
    BCThreePhaseLowPressure,
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (
    ICThreePhaseLowPressure,
)

use_schur_technique = False

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # This gives the path of this script's folder
VTK_DIR = BASE_DIR / "model_configuration" / "constitutive_description" / "driesner_vtk_files"

# Simulation configurations
SIMULATION_CASES = {
    "three_phase_LP": {  # Low-pressure two-phase (Figure 4)
        "BC": BCThreePhaseLowPressure,
        "IC": ICThreePhaseLowPressure,
        "FlowModel": ThreePhaseFlowModel,
        "tf": 2000.0 * 365.0 * 86400,
        "dt": 200.0 * 86400,
    },
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
# pressure_increment = []
# energy_increment = []
# current_pressure = []
# current_energy = []


def create_dynamic_model(BC, IC, FlowModel):
    """Create a geothermal model class with specific BC, IC, and Flow Model."""
    class GeothermalSimulationFlowModel(ModelGeometry, BC, IC, FlowModel):
        def compute_residual_norm(
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
            # return super().solve_linear_system()
            eq_idx_map = self.equation_system.assembled_equation_indices
            eq_p_dof_idx = eq_idx_map['mass_balance_equation']
            eq_T_dof_idx = eq_idx_map['component_mass_balance_equation_NaCl']
            eq_h_dof_idx = eq_idx_map['energy_balance_equation']
            eq_t_dof_idx = eq_idx_map['elimination_of_temperature_on_grids_[0]']
            eq_s_dof_idx = eq_idx_map['elimination_of_s_gas_on_grids_[0]']
            eq_xs_v_dof_idx = eq_idx_map['elimination_of_x_NaCl_liq_on_grids_[0]']
            eq_xs_l_dof_idx = eq_idx_map['elimination_of_x_NaCl_gas_on_grids_[0]']
            eq_z_dof_idx = eq_idx_map['elimination_of_s_halite_on_grids_[0]']

            # A, b = self.linear_system
            # x = sps.linalg.spsolve(A, b)
            # pressure_sol = x[eq_p_dof_idx]
            # pressure_increment.append(pressure_sol.copy())
            # energy_sol = x[eq_h_dof_idx]
            # energy_increment.append(energy_sol.copy())

            # current_solution =  self.equation_system.get_variable_values(iterate_index=0)
            # current_pressure.append(current_solution[eq_p_dof_idx])
            # current_energy.append(current_solution[eq_h_dof_idx])

            # # Convert to arrays
            # pressure_increment_array = np.array(pressure_increment)  # Shape (num_iter, num_pressure_dofs)
            # energy_increment_array = np.array(energy_increment)      # Shape (num_iter, num_energy_dofs)
            # current_pressure_array = np.array(current_pressure)  # Shape (num_iter, num_pressure_dofs)
            # current_energy_array = np.array(current_energy)

            # # Save
            # output_folder = "newton_outputs"
            # os.makedirs(output_folder, exist_ok=True)

            # np.save(os.path.join(output_folder, "pressure_increment_iterations.npy"), pressure_increment_array)
            # np.save(os.path.join(output_folder, "energy_increment_iterations.npy"), energy_increment_array)
            # np.save(os.path.join(output_folder, "current_pressure_iterations.npy"), current_pressure_array)
            # np.save(os.path.join(output_folder, "current_energy_iterations.npy"), current_energy_array)

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
        
        def solve_linear_system_v1(self):
            """Custom Newton solver with backtracking line search."""
            # Get current Newton iterate and system
            x_k = self.equation_system.get_variable_values(iterate_index=0)
            A, b = self.linear_system

            # Solve Newton step: A * delta_x = b
            delta_x = sps.linalg.spsolve(A, b)

            # Residual norm before update
            res_norm_k = np.linalg.norm(b)

            # Line search parameters
            alpha = 1.0
            c = 1e-4
            rho = 0.5
            max_ls_iter = 10

            for i in range(max_ls_iter):
                x_new = x_k - alpha * delta_x
                self.equation_system.set_variable_values(x_new, iterate_index=0)
                # self.equation_system.update_equations()
                self.assemble_linear_system()
                rhs_new = self.equation_system.assemble(evaluate_jacobian=False)
                res_norm_new = np.linalg.norm(rhs_new)

                if res_norm_new <= (1.0 - c * alpha) * res_norm_k:
                    print(f"Line search success at alpha={alpha:.3f}, residual norm: {res_norm_new:.2e}")
                    break
                else:
                    print(f"Line search alpha={alpha:.3f} failed, residual norm: {res_norm_new:.2e}")
                    alpha *= rho
            else:
                print("Line search failed to find sufficient decrease.")

            # Accept step and update solution
            self.equation_system.set_variable_values(x_k - alpha * delta_x, iterate_index=0)

            # Optional: print residual breakdown
            res_full = self.equation_system.assemble(evaluate_jacobian=False)
            eq_idx_map = self.equation_system.assembled_equation_indices
            print("Residual norm breakdown:")
            for eq_name, idx in eq_idx_map.items():
                print(f"  {eq_name}: {np.linalg.norm(res_full[idx]):.2e}")

            return delta_x

        def solve_linear_system_v2(self):
            """Robust Newton solver with backtracking line search and physical clipping."""
            # Current solution vector
            x_k = self.equation_system.get_variable_values(iterate_index=0)
            A, b = self.linear_system

            # Solve for Newton direction
            delta_x = sps.linalg.spsolve(A, b)
            res_norm_k = np.linalg.norm(b)

            # Backtracking line search parameters
            alpha = 1.0
            c = 1e-4
            rho = 0.5
            max_ls_iter = 10

            def safe_update(x_old, delta, alpha):
                """Apply delta with damping and clipping."""
                x_trial = x_old - alpha * delta

                # === Clip dangerous variables ===
                try:
                    T_idx = self.equation_system.assembled_equation_indices['component_mass_balance_equation_NaCl'] # TODO: Equation system is rendering indices correctly!
                    x_trial[T_idx] = np.clip(x_trial[T_idx], 0.0, 2500.0)
                except Exception:
                    pass

                try:
                    H_idx = self.equation_system.assembled_equation_indices['energy_balance_equation']
                    x_trial[H_idx] = np.clip(x_trial[H_idx], 0.0, 4.0e6)
                except Exception:
                    pass

                # Global safety clamp (emergency)
                # x_trial = np.clip(x_trial, -1e8, 1e8)

                return x_trial

            for i in range(max_ls_iter):
                x_new = safe_update(x_k, delta_x, alpha)
                self.equation_system.set_variable_values(x_new, iterate_index=0)
                self.assemble_linear_system()
                rhs_new = self.equation_system.assemble(evaluate_jacobian=False)
                res_norm_new = np.linalg.norm(rhs_new)

                if not np.isfinite(res_norm_new):
                    print(f"Line search alpha={alpha:.3f} gave non-finite residual. Reducing step.")
                    alpha *= rho
                    continue

                if res_norm_new <= (1.0 - c * alpha) * res_norm_k:
                    print(f"Line search success at alpha={alpha:.3f}, residual norm: {res_norm_new:.2e}")
                    break
                else:
                    print(f"Line search alpha={alpha:.3f} failed, residual norm: {res_norm_new:.2e}")
                    alpha *= rho
            else:
                print("Line search failed to find sufficient decrease.")

            # Accept and set new solution
            x_new = safe_update(x_k, delta_x, alpha)
            self.equation_system.set_variable_values(x_new, iterate_index=0)

            # Optional: detailed residual diagnostics
            res_full = self.equation_system.assemble(evaluate_jacobian=False)
            eq_idx_map = self.equation_system.assembled_equation_indices
            print("Residual norm breakdown:")
            for eq_name, idx in eq_idx_map.items():
                print(f"  {eq_name}: {np.linalg.norm(res_full[idx]):.2e}")

            return delta_x
        
        def solve_linear_system_v(self):
            """Improved Newton solver with robust line search and physical safety checks."""
            from scipy.sparse.linalg import spsolve
            import numpy as np

            # Get current Newton iterate and system
            x_k = self.equation_system.get_variable_values(iterate_index=0)
            A, b = self.linear_system

            # Newton step
            try:
                delta_x = spsolve(A, b)
            except Exception as e:
                raise RuntimeError("Linear solve failed.") from e

            res_norm_k = np.linalg.norm(b)

            # Line search parameters
            alpha = 1.0
            c = 1e-4
            rho = 0.5
            max_ls_iter = 10

            # Helper: safe update + clipping
            def safe_update(x_old, delta, alpha):
                x_trial = x_old - alpha * delta

                # Clip dangerous or unbounded variables
                try:
                    H_idx = self.equation_system.assembled_equation_indices['energy_balance_equation']
                    # x_trial[H_idx] = np.clip(x_trial[H_idx], 1.0e3, 3.5e6)  # Enthalpy
                except Exception:
                    pass

                try:
                    T_idx = self.equation_system.assembled_equation_indices['component_mass_balance_equation_NaCl']
                    x_trial[T_idx] = np.clip(x_trial[T_idx], 423.15, 2500.0)  # Temperature in Kelvin
                except Exception:
                    pass

                # Optional: safety clamp for all variables
                # x_trial = np.clip(x_trial, -1e8, 1e8)
                return x_trial

            # Backtracking line search
            for i in range(max_ls_iter):
                x_new = safe_update(x_k, delta_x, alpha)
                self.equation_system.set_variable_values(x_new, iterate_index=0)
                self.assemble_linear_system()
                rhs_new = self.equation_system.assemble(evaluate_jacobian=False)
                res_norm_new = np.linalg.norm(rhs_new)

                if not np.isfinite(res_norm_new):
                    print(f"Line search alpha={alpha:.3f} → non-finite residual. Reducing alpha.")
                    alpha *= rho
                    continue

                if res_norm_new <= (1.0 - c * alpha) * res_norm_k:
                    print(f"Line search success at alpha={alpha:.3f}, residual norm: {res_norm_new:.2e}")
                    break
                else:
                    print(f"Line search alpha={alpha:.3f} failed, residual norm: {res_norm_new:.2e}")
                    alpha *= rho
            else:
                print("Line search failed to find sufficient decrease. Proceeding with damped step.")

            # Accept final update
            x_new = safe_update(x_k, delta_x, alpha)
            self.equation_system.set_variable_values(x_new, iterate_index=0)

            # Final residual diagnostics
            res_full = self.equation_system.assemble(evaluate_jacobian=False)
            eq_idx_map = self.equation_system.assembled_equation_indices
            print("Residual norm breakdown:")
            for eq_name, idx in eq_idx_map.items():
                print(f"  {eq_name}: {np.linalg.norm(res_full[idx]):.2e}")

            return delta_x

    # Return the dynamically created model class    
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

    # Model parameters
    params = {
        "material_constants": material_constants,
        "eliminate_reference_phase": True,
        "eliminate_reference_component": True,
        "time_manager": time_manager,
        "prepare_simulation": False,
        "reduce_linear_system": use_schur_technique,
        "rediscretize_darcy_flux": True,
        "nl_convergence_tol": np.inf,
        "nl_convergence_tol_res": 1.0e-3,
        "max_iterations": 100,
    }
    params["use_line_search"] = True

    # Initialize model
    model = GeothermalModel(params)
    # Load VTK files
    # /Users/michealoguntola/Desktop/Porepy_Velkjo/porepy/src/porepy/examples/geothermal_flow
    # path_file_ptz = "/Users/michealoguntola/Desktop/Porepy_Velkjo/porepy/src/porepy/examples/geothermal_flow/"
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


# ------------------------------------------------------
# Run Simulations for All Configured Cases
# ------------------------------------------------------

# Define file paths for VTK files used for thermodynamic property sampling
correl_vtk_ptz_salt = VTK_DIR / "XTP_l2_original_salt.vtk"
correl_vtk_phz_salt = VTK_DIR / "XHP_l2_original_salt.vtk"

for case_name, config in SIMULATION_CASES.items():
    run_simulation(
        case_name,
        config,
        correl_vtk_phz=correl_vtk_phz_salt,
        correl_vtk_ptz=correl_vtk_ptz_salt
    )
