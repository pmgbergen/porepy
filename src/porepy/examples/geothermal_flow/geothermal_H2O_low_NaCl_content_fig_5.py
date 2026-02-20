"""
Geothermal flow simulation with H2O and low NaCl content (Figure 5).

This script includes a backtracking line search algorithm to improve Newton
convergence. The line search can be enabled/disabled via the 'use_line_search'
parameter in the params dictionary.

Line search parameters (in DriesnerModelConfiguration.backtracking_line_search):
- alpha_init: Initial step length (default: 1.0)
- rho: Step reduction factor (default: 0.5)
- c: Armijo parameter (default: 1e-4)
- max_iterations: Maximum backtracking steps (default: 10)
"""
from __future__ import annotations

import time
from typing import cast

import numpy as np

import porepy as pp

# geometry description horizontal case
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import (  # noqa: E501
    SimpleGeometryHorizontal as ModelGeometryH,
)
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import (  # noqa: E501
    SimpleGeometryVertical as ModelGeometryV,
)


# Figure 5 two with low pressure (lP) condition
# Horizontal without gravity
# Vertical with gravity

from porepy.examples.geothermal_flow.model_configuration.DriesnerModelConfiguration import (  # noqa: E501
    DriesnerBrineFlowModel as FlowModel,
)

from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (  # noqa: E501
    BC_two_phase_moderate_pressure as BC,
)

from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (  # noqa: E501
    IC_two_phase_moderate_pressure as IC,
)
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler

# Main directives
case_name = "case_lP"
geometry_case = "horizontal"

final_times = {
    "horizontal": [73000.0],  # final time [200 years]
    "vertical": [365000.0],  # final time [1000 years]
}

day_to_second = 86400
to_Mega = 1.0e-6
# Configuration dictionary mapping cases to their specific classes
simulation_cases = {
    "case_lP": {
        "tf": final_times[geometry_case][0] * day_to_second,  # final time [years]
        "dt": 1.0 *  365.0 * day_to_second,  # final time [1 years]
        "bc": BC,
        "ic": IC,
    }
}

geometry_cases = {
    "horizontal": ModelGeometryH,
    "vertical": ModelGeometryV,
}

tf = cast(float, simulation_cases[case_name]["tf"])
dt = cast(float, simulation_cases[case_name]["dt"])
BoundaryConditions: type = cast(type, simulation_cases[case_name]["bc"])
InitialConditions: type = cast(type, simulation_cases[case_name]["ic"])
ModelGeometry: type = cast(type, geometry_cases[geometry_case])

time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

# time_manager = pp.TimeManager(
#     schedule=[0.0, tf],
#     dt_init=dt,
#     constant_dt=False,
#     dt_min_max=(dt * 0.05, 1.0 * dt),
#     iter_relax_factors=(0.5, 1.5),
#     iter_optimal_range=(3, 8),
#     recomp_factor=0.3,
#     print_info=True,
# )

solid_constants = pp.SolidConstants(
    permeability=1e-15,
    porosity=0.1,
    thermal_conductivity=2.0 * to_Mega,
    density=2700.0,
    specific_heat_capacity=880.0 * to_Mega,
)
material_constants = {"solid": solid_constants}
params = {
    "fractional_flow": True,
    "enable_buoyancy_effects": True,
    "material_constants": material_constants,
    "time_manager": time_manager,
    "prepare_simulation": False,
    "apply_schur_complement_reduction": False,
    "nl_convergence_tol": np.inf,
    "nl_convergence_tol_res": 1.0e-5,
    "flag_failure_as_diverged": False,
    "max_iterations": 100,
    # "nonlinear_solver": line_search.ConstraintLineSearchNonlinearSolver,
    # "global_line_search": 1,
    "use_petsc": False,  # Set to True to use PETSc with MUMPS solver
    "petsc_preconditioner": "cpr",  # Options: 'bjacobi', 'asm', 'jacobi', 'lump_colsum', 'amg_hypre', 'ilu0', 'lu', 'cpr'

    # Step control method options:
    # - "LS": Line Search (backtracking with Armijo condition)
    # - "TR": Trust Region with CFL-aware dynamic radius adjustment
    # - "TR-LS": Trust Region + Line Search refinement
    # - "None": Plain Newton (no step control)
    "step_control_method": "TR",

    "step_control_alpha_min": 1.0e-5,  # Minimum acceptable step length
    "activate_step_control_after_iter": 1,  # Activate after this many iterations

    # Trust region specific parameters (only used for TR and TR-LS methods)
    "trust_region_min_radius": 0.5,          # Minimum trust region radius (prevents collapse)
    "trust_region_max_radius": 100.0,        # Maximum trust region radius (prevents unbounded growth)
    "trust_region_aggressive": True,         # For hyperbolic systems: accept any step that reduces residual
    "trust_region_block_structured": True,   # Leverage block structure: trust pressure (SPD), limit hyperbolic vars

    # CFL-based trust radius bounds (RECOMMENDED for hyperbolic stability)
    "trust_region_use_cfl_bounds": True,     # Use CFL to set physics-based bounds: min=1/CFL, max=CFL*10

    # CFL-aware dynamic radius adjustment (acts as dynamic CFL limiter)
    "trust_region_cfl_max_target": 10.0,              # Target CFL for expansion
}
# params = {
#     "material_constants": material_constants,
#     "fractional_flow": True,
#     "buoyancy_on": True,
#     "time_manager": time_manager,
#     "prepare_simulation": False,
#     "apply_schur_complement_reduction": False,
#     "nl_convergence_tol": np.inf,
#     "nl_convergence_tol_res": 1.0e-4,
#     "max_iterations": 500,
# }


class GeothermalWaterFlowModel(
    ModelGeometry, BoundaryConditions, InitialConditions, FlowModel
):
    def after_nonlinear_convergence(self) -> None:
        second_to_year = 1.0 / (365 * day_to_second)
        super().after_nonlinear_convergence()  # type:ignore[safe-super]
        print("Number of iterations: ", self.nonlinear_solver_statistics.num_iteration)
        print("Time value (year): ", self.time_manager.time * second_to_year)
        print("Time index: ", self.time_manager.time_index)
        print("")


# Instance of the computational model
model = GeothermalWaterFlowModel(params)

parametric_space_ref_level = 2
folder_prefix = "src/porepy/examples/geothermal_flow/"
file_name_prefix = (
    "model_configuration/constitutive_description/driesner_vtk_files/"
)
file_name_phz = (
    file_name_prefix
    + "XHP_l"
    + str(parametric_space_ref_level)
    + "_modified_low_salt_content.vtk"
)
file_name_ptz = (
    file_name_prefix
    + "XTP_l"
    + str(parametric_space_ref_level)
    + "_modified_low_salt_content.vtk"
)

brine_sampler_phz = VTKSampler(file_name_phz)
brine_sampler_phz.conversion_factors = (1.0, 1.0e3, 10.0)  # (z,h,p)
model.vtk_sampler = brine_sampler_phz

brine_sampler_ptz = VTKSampler(file_name_ptz)
brine_sampler_ptz.conversion_factors = (1.0, 1.0, 10.0)  # (z,t,p)
brine_sampler_ptz.translation_factors = (0.0, -273.15, 0.0)  # (z,t,p)
model.vtk_sampler_ptz = brine_sampler_ptz


tb = time.time()
model.prepare_simulation()
te = time.time()
print("Elapsed time prepare simulation: ", te - tb)
print("Simulation prepared for total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid employed: ", model.mdg)
model.schur_complement_primary_equations = (
    pp.compositional_flow.get_primary_equations_cf(model)
)
model.schur_complement_primary_variables = (
    pp.compositional_flow.get_primary_variables_cf(model)
)

# print geometry
model.exporter.write_vtu()
tb = time.time()
pp.run_time_dependent_model(model, params)
te = time.time()
print("Elapsed time run_time_dependent_model: ", te - tb)
print("Total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid information: ", model.mdg)

# Retrieve the grid and boundary information
grid = model.mdg.subdomains()[0]
bc_sides = model.domain_boundary_sides(grid)

# Integrated overall mass flux on all facets
mn = model.equation_system.evaluate(model.darcy_flux(model.mdg.subdomains()))
mn = cast(np.ndarray, mn)

inlet_idx, outlet_idx = model.get_inlet_outlet_sides(model.mdg.subdomains()[0])
print("Inflow values : ", mn[inlet_idx])
print("Outflow values : ", mn[outlet_idx])
