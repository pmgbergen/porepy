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
from pathlib import Path
from typing import Optional

# Import model configurations
# from porepy.examples.geothermal_flow.model_configuration.flow_model_configuration_iso import (
#     SinglePhaseFlowModelConfigurationLiquid as LiquidPhaseModel
# )
from porepy.examples.geothermal_flow.model_configuration.flow_model_configuration_iso import (
    TwoPhaseFlowModelConfiguration as TwoPhasePhaseModel
)
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler

# Import geometric setup for the model domain
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import (
    SimpleGeometryHorizontal as ModelGeometry
)

# Boundary & Initial Conditions
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (
    BCSinglePhaseHighPressure,
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (
    ICSinglePhaseHighPressure,
)

use_schur_technique = False

BASE_DIR = Path(__file__).resolve().parent  # This gives the path of this script's folder
VTK_DIR = BASE_DIR / "model_configuration" / "constitutive_description" / "driesner_vtk_files"

# Simulation configurations
SIMULATION_CASES = {
    "single_phase_HP": {    # High-pressure single-phase (Figure 2, Case 1)
        "BC": BCSinglePhaseHighPressure,
        "IC": ICSinglePhaseHighPressure,
        "FlowModel": TwoPhasePhaseModel,
        "tf": 200 * 365 * 86400,
        "dt": 365 * 86400
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

        # def solve_linear_system(self):
        #     eq_idx_map = self.equation_system.assembled_equation_indices
        #     eq_p_dof_idx = eq_idx_map['mass_balance_equation']
        #     eq_T_dof_idx = eq_idx_map['component_mass_balance_equation_NaCl']
        #     eq_h_dof_idx = eq_idx_map['energy_balance_equation']
        #     eq_t_dof_idx = eq_idx_map['elimination_of_temperature_on_grids_[0]']
        #     eq_s_dof_idx = eq_idx_map['elimination_of_s_gas_on_grids_[0]']
        #     eq_xs_v_dof_idx = eq_idx_map['elimination_of_x_NaCl_liq_on_grids_[0]']
        #     eq_xs_l_dof_idx = eq_idx_map['elimination_of_x_NaCl_gas_on_grids_[0]']
        #     eq_z_dof_idx = eq_idx_map['elimination_of_s_halite_on_grids_[0]']

        #     jac_g, res_g = self.linear_system
        #     print("Overall residual norm at x_k: ", np.linalg.norm(res_g))
        #     print("Pressure residual norm: ", np.linalg.norm(res_g[eq_p_dof_idx]))
        #     print("Composition residual norm: ", np.linalg.norm(res_g[eq_z_dof_idx]))
        #     print("Enthalpy residual norm: ", np.linalg.norm(res_g[eq_h_dof_idx]))
        #     print("Temperature residual norm: ", np.linalg.norm(res_g[eq_T_dof_idx]))
        #     print("Halite Saturation residual norm: ", np.linalg.norm(res_g[eq_s_dof_idx]))
        #     print("Xs_v residual norm: ", np.linalg.norm(res_g[eq_xs_v_dof_idx]))
        #     print("Xs_l residual norm: ", np.linalg.norm(res_g[eq_xs_l_dof_idx]))
        #     print(" ")
        #     return super().solve_linear_system()
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
correl_vtk_phz_2 = VTK_DIR / "XHP_l2_original.vtk"

for case_name, config in SIMULATION_CASES.items():
    run_simulation(case_name, config, correl_vtk_phz=correl_vtk_phz_2)
