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
        "dt": 365.0 * 86400,
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
    class GeothermalSimulationFlowModel(ModelGeometry, BC, IC, FlowModel):
        def compute_residual_norm_old(
            self, residual: Optional[np.ndarray], reference_residual: np.ndarray
        ) -> float:
            if residual is None:
                return np.nan
            residual_norm = np.linalg.norm(residual)
            return float(residual_norm)
        
        def schur_complement_inverter(self) -> Callable[[sps.spmatrix], sps.spmatrix]:
            """Parallelized block diagonal inverter for local equilibrium equations,
            assuming they are defined on all subdomains in each cell."""

            # return super().schur_complement_inverter()
            if not hasattr(self, "_block_permutation"):
                # Local system size of p-h flash including saturations and phase mass
                # constraints.
                # ncomp = self.fluid.num_components
                # nphase = self.fluid.num_phases
                block_size: int = len(self.secondary_variables)
                # assert block_size == len(self.secondary_equations)
                assert block_size == len(self.secondary_variables)
                self._block_size = block_size

                # Permutation of DOFs in porepy, assuming the order is equ1, equ2,...
                # With an subdomain-minor order per variable.
                N = self.mdg.num_subdomain_cells()
                shift = np.kron(np.arange(block_size), np.ones(N))
                stride = np.arange(N) * block_size
                permutation = (np.kron(np.ones(block_size), stride) + shift).astype(
                    np.int32
                )
                identity = np.arange(N * block_size, dtype=np.int32)
                assert np.allclose(np.sort(permutation), identity, rtol=0.0, atol=1e-16)

                self._block_row_permutation = permutation

                # Above permutation can be used for both column and blocks, if there is
                # no point grid. If there is a point grid, above permutation assembles the
                # blocks belonging to the point grid cells already fully, due to how AD is
                # implemented in PorePy. We then need a special column permutation, which
                # permutes only on the those grids which are not point grids.
                # Note that the subdomains are sorted by default ascending w.r.t. their
                # dimension. Point grids come last
                subdomains = self.mdg.subdomains()
                if subdomains[-1].dim == 0:
                    non_point_grids = [g for g in subdomains if g.dim > 0]
                    sub_N = sum([g.num_cells for g in non_point_grids])
                    shift = np.kron(np.arange(block_size), np.ones(sub_N))
                    stride = np.arange(sub_N) * block_size
                    sub_permutation = (np.kron(np.ones(block_size), stride) + shift).astype(
                        np.int32
                    )
                    sub_identity = np.arange(sub_N * block_size, dtype=np.int32)
                    assert np.allclose(
                        np.sort(sub_permutation), sub_identity, rtol=0.0, atol=1e-16
                    )
                    permutation = np.arange(N * block_size, dtype=np.int32)
                    permutation[: sub_N * block_size] = sub_permutation
                    self._block_column_permutation = permutation

            def inverter(A: sps.csr_matrix) -> sps.csr_matrix:
                row_perm = pp.matrix_operations.ArraySlicer(
                    range_indices=self._block_row_permutation
                )
                inv_col_perm = pp.matrix_operations.ArraySlicer(
                    domain_indices=self._block_row_permutation
                )
                if hasattr(self, "_block_column_permutation"):
                    col_perm = pp.matrix_operations.ArraySlicer(
                        range_indices=self._block_column_permutation
                    )
                    inv_row_perm = pp.matrix_operations.ArraySlicer(
                        domain_indices=self._block_column_permutation
                    )
                else:
                    col_perm = row_perm
                    inv_row_perm = inv_col_perm
                
                A_block = (col_perm @ (row_perm @ A).transpose()).transpose()  # Veljko
                # A_block = (col_perm.transpose() @ (row_perm @ A).transpose()).transpose()

                # The local p-h flash system has a size of
                inv_A_block = pp.matrix_operations.invert_diagonal_blocks(
                    A_block,
                    (np.ones(self.mdg.num_subdomain_cells()) * self._block_size).astype(
                        np.int32
                    ),
                    method="numba",
                )
                # inv_A = (
                #     inv_col_perm @ (inv_row_perm @ inv_A_block).transpose()  # Veljko
                # ).transpose()

                inv_A = (
                    inv_row_perm @ (inv_col_perm @ inv_A_block).transpose()  # micheal
                ).transpose()
                # Because the matrix has a condition number of order 5 to 6, and we use
                # float64 (precision order -16), the last entries are useless, if not
                # problematic.
                treat_as_zero = np.abs(inv_A.data) < 1e-10
                inv_A.data[treat_as_zero] = 0.0
                inv_A.eliminate_zeros()
                return inv_A

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
            # TODO: The mismatch in the mapping of the equations to block indices 
            # is due to bug in the equation system. There is a quick fix to this
            # but I am lazy to that, hence the reason for the manual selection.
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
        primary_equations = model.get_primary_equations_cf()
        primary_variables = model.get_primary_variables_cf()

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
    pvd_file = "./visualization/data.pvd"
    mesh = data_util.get_last_mesh_from_pvd(pvd_file)

    # Load saved csmp++ temperature, pressure, and saturation data
    num_points = 200
    dx = 2.0 / (num_points - 1)  # cell size
    xc = np.arange(0.0, 2.0 + dx, dx)
    pressure_csmp = []
    temperature_csmp = []
    saturation_csmp = []

    OUTPUT_DIR = os.getcwd()
    if case_name == "single_phase_HP":
        save_path = os.path.join(OUTPUT_DIR, f"{case_name}.png")
        simulation_time = 250

        pressure_csmp, temperature_csmp = data_util.fig_4A_load_and_project_reference_data(xc)

        data_util.plot_temp_pressure_comparison(
            mesh,
            pressure_csmp,
            temperature_csmp,
            xc,
            [150, 350],
            np.linspace(150, 350, 5),
            [25, 50],
            np.linspace(25, 50, 6),
            simulation_time,
            save_path,
        )
    if case_name == "single_phase_MP":
        save_path = os.path.join(OUTPUT_DIR, f"{case_name}.png")
        simulation_time = 120
        pressure_csmp, temperature_csmp = data_util.fig_4C_load_and_project_reference_data(xc)

        data_util.plot_temp_pressure_comparison(
            mesh,
            pressure_csmp,
            temperature_csmp,
            xc,
            [295, 460],
            np.linspace(300, 450, 4),
            [20, 40],
            np.linspace(20, 40, 5),
            simulation_time,
            save_path,
        )
    if case_name == "single_phase_LP":
        save_path = os.path.join(OUTPUT_DIR, f"{case_name}.png")
        simulation_time = 1500
        pressure_csmp, temperature_csmp = data_util.fig_4E_load_and_project_reference_data(xc)

        data_util.plot_temp_pressure_comparison(
            mesh,
            pressure_csmp,
            temperature_csmp,
            xc,
            [280, 513],
            np.linspace(300, 500, 5),
            [0, 15],
            np.linspace(0.0, 15, 6),
            simulation_time,
            save_path,
        )
    
    if case_name == "two_phase_HP":
        simulation_time = 200
        # Extract the 'pressure' data (cell data)
        centroids = mesh.cell_centers().points
        x_coords = centroids[:, 0]*1e-3
        # Load saturation data
        s_gas = mesh.cell_data['s_gas']
        s_liq = 1 - s_gas
        mask = (s_liq >= 0.1) & (s_liq < 1.0)
        filtered_coords = centroids[mask][:, 0]*1e-3
        min_x = np.min(filtered_coords)
        max_x = np.max(filtered_coords)
        # Extract the 'pressure' and 'temperature' data (cell data)
        pressure = mesh.cell_data['pressure'] * 1e-6  # in MPa
        temperature = -273.15 + mesh.cell_data['temperature']  # in oC

        data_util.plot_temp_pressure_two_phase(
            x_coords,
            temperature,
            [145, 405],
            np.linspace(150, 400, 6),
            pressure,
            [0, 20],
            np.linspace(0, 20, 5),
            min_x,
            max_x,
            simulation_time,
            "porepy",
            os.path.join(OUTPUT_DIR, "two_phase_porepy_HP.png")
        )
        
        data_util.plot_liquid_saturation(
            x_coords,
            s_liq,
            min_x,
            max_x,
            os.path.join(OUTPUT_DIR, "two_phase_saturation_porepy_HP.png")
        )

        # Load CSMP++ results
        pressure_csmp, temperature_csmp, saturation_csmp = data_util.fig_5_load_and_project_reference_data(xc)
        # TWO-PHASE REGION
        mask = (saturation_csmp >= 0.23) & (saturation_csmp < 1.0)
        filtered_coords = centroids[mask][:, 0]*1e-3
        min_x_csmp = np.min(filtered_coords)
        max_x_csmp = np.max(filtered_coords)

        data_util.plot_temp_pressure_two_phase(
            x_coords,
            temperature_csmp,
            [145, 405],
            np.linspace(150, 400, 6),
            pressure_csmp,
            [0, 20],
            np.linspace(0, 20, 5),
            min_x_csmp,
            max_x_csmp,
            simulation_time,
            "csmp++",
            os.path.join(OUTPUT_DIR, "two_phase_csmp_HP.png")
        )

        data_util.plot_liquid_saturation(
            x_coords,
            saturation_csmp,
            min_x_csmp,
            max_x_csmp,
            os.path.join(OUTPUT_DIR, "two_phase_saturation_csmp_HP.png")
        )

    if case_name == "two_phase_LP":
        simulation_time = 200
        # Extract the 'pressure' data (cell data)
        centroids = mesh.cell_centers().points
        x_coords = centroids[:, 0]*1e-3
        # Load saturation data
        s_gas = mesh.cell_data['s_gas']
        s_liq = 1 - s_gas
        mask = (s_liq >= 0.1) & (s_liq < 0.7)
        filtered_coords = centroids[mask][:, 0]*1e-3
        min_x = np.min(filtered_coords)
        max_x = np.max(filtered_coords)
        # Extract the 'pressure' and 'temperature' data (cell data)
        pressure = mesh.cell_data['pressure'] * 1e-6  # in MPa
        temperature = -273.15 + mesh.cell_data['temperature']  # in oC

        data_util.plot_temp_pressure_two_phase(
            x_coords,
            temperature,
            [150, 300],
            np.linspace(150, 300, 4),
            pressure,
            [1, 4],
            np.linspace(1, 4, 4),
            min_x,
            max_x,
            simulation_time,
            "porepy",
            os.path.join(OUTPUT_DIR, "two_phase_porepy_LP.png")
        )

        data_util.plot_liquid_saturation(
            x_coords,
            s_liq,
            min_x,
            max_x,
            os.path.join(OUTPUT_DIR, "two_phase_saturation_porepy_LP.png")
        )

        # Load CSMP++ results
        pressure_csmp, temperature_csmp, saturation_csmp = data_util.fig_6_load_and_project_reference_data(xc)
        # TWO-PHASE REGION
        mask = (saturation_csmp >= 0.23) & (saturation_csmp < 1.0)
        filtered_coords = centroids[mask][:, 0]*1e-3
        min_x_csmp = np.min(filtered_coords)
        max_x_csmp = np.max(filtered_coords)

        data_util.plot_temp_pressure_two_phase(
            x_coords,
            temperature_csmp,
            [150, 300],
            np.linspace(150, 300, 4),
            pressure_csmp,
            [1, 4],
            np.linspace(1, 4, 4),
            min_x_csmp,
            max_x_csmp,
            simulation_time,
            "csmp++",
            os.path.join(OUTPUT_DIR, "two_phase_csmp_LP.png")
        )

        data_util.plot_liquid_saturation(
            x_coords,
            saturation_csmp,
            min_x_csmp,
            max_x_csmp,
            os.path.join(OUTPUT_DIR, "two_phase_saturation_csmp_LP.png")
        )


# ------------------------------------------------------
# Run Simulations for All Configured Cases
# ------------------------------------------------------

# Define file paths for VTK files used for thermodynamic property sampling
correl_vtk_phz_1 = VTK_DIR / "XHP_l2_original_sc.vtk"
correl_vtk_phz_2 = VTK_DIR / "XHP_l2_original_all.vtk"
correl_vtk_phz_3 = VTK_DIR / "XHP_l2_original.vtk"
correl_vtk_ptz_salt = VTK_DIR / "XTP_l2_original_salt.vtk"
correl_vtk_phz_salt = VTK_DIR / "XHP_l2_original_salt.vtk"

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
        run_simulation(
            case_name,
            config,

            correl_vtk_phz=correl_vtk_phz_2
        )