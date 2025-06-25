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
import scipy.sparse as sps


from porepy import matrix_operations as mo
from porepy import compositional_flow as cf

# Import model configurations
from porepy.examples.geothermal_flow.model_configuration.flow_model_config_2D import (
    BrineFlowModelConfiguration2D as BrineFlowModel2D,
)
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler

# Import geometric setup for the model domain
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import DomainPointGridWellsFracture2D as ModelGeometry2D

# Boundary & Initial Conditions
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (
    BCBrineSystem2D,
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (
    ICBrineSystem2D,
)

use_schur_technique = False

BASE_DIR = Path(__file__).resolve().parent  # This gives the path of this script's folder
VTK_DIR = BASE_DIR / "model_configuration" / "constitutive_description" / "driesner_vtk_files"

# Simulation configurations
SIMULATION_CASES = {
    "three_phase_LP_2D": {  # Low-pressure two-phase (Figure 4)
        "BC": BCBrineSystem2D,
        "IC": ICBrineSystem2D,
        "FlowModel": BrineFlowModel2D,
        "tf": 100.0 * 365.0 * 86400,
        "dt": 0.5 * 86400,
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


def create_dynamic_model(BC, IC, FlowModel):
    """Create a geothermal model class with specific BC, IC, and Flow Model."""
    class GeothermalSimulationFlowModel(ModelGeometry2D, BC, IC, FlowModel):

        def schur_complement_inverter(self) -> Callable[[sps.spmatrix], sps.spmatrix]:
            """Parallelized block diagonal inverter for local equilibrium equations,
            assuming they are defined on all subdomains in each cell."""

            def inverter(A: sps.csr_matrix) -> sps.csr_matrix:
                row_perm, col_perm, block_sizes = mo.generate_permutation_to_block_diag_matrix(A)
              
                return mo.invert_permuted_block_diag_matrix(
                    A, row_perm, col_perm, block_sizes
                )

            return inverter
        
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
            # TODO: -------------Debugging---------------
            eq_idx_map = self.equation_system.assembled_equation_indices
            eq_p_dof_idx = eq_idx_map['mass_balance_equation']
            eq_h_dof_idx = eq_idx_map['energy_balance_equation']

            _, res_g = self.linear_system
            print("Overall residual norm at x_k: ", np.linalg.norm(res_g))
            print("Pressure residual norm: ", np.linalg.norm(res_g[eq_p_dof_idx]))
            print("Enthalpy residual norm: ", np.linalg.norm(res_g[eq_h_dof_idx]))
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
    # grid = model.mdg.subdomains()[0]
    
    # Compute mass flux
    # darcy_flux = model.darcy_flux(model.mdg.subdomains()).value(model.equation_system)
    # inlet_idx, outlet_idx = model.get_inlet_outlet_sides(grid)
    # print(f"Inflow values: {darcy_flux[inlet_idx]}")
    # print(f"Outflow values: {darcy_flux[outlet_idx]}")

    # Get the last time step's solution data
    # pvd_file = "./visualization/data.pvd"
    # mesh = data_util.get_last_mesh_from_pvd(pvd_file)

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
