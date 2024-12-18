from __future__ import annotations

import time

import numpy as np

import porepy as pp

from porepy.examples.geothermal_flow.model_configuration.DriesnerModelConfiguration import (
    DriesnerBrineFlowModel as FlowModel,
)
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler

# geometry description horizontal case
from model_configuration.geometry_description.geometry_market import SimpleGeometryHorizontal as ModelGeometryH
from model_configuration.geometry_description.geometry_market import SimpleGeometryVertical as ModelGeometryV

# Figure 4 single with high pressure (hP) condition
from model_configuration.bc_description.bc_market import BC_single_phase_high_pressure as BC_hP
from model_configuration.ic_description.ic_market import IC_single_phase_high_pressure as IC_hP

# Figure 4 single with moderate pressure (mP) condition
from model_configuration.bc_description.bc_market import BC_single_phase_moderate_pressure as BC_mP
from model_configuration.ic_description.ic_market import IC_single_phase_moderate_pressure as IC_mP

# Figure 4 single with low pressure (lP) condition
from model_configuration.bc_description.bc_market import BC_single_phase_low_pressure as BC_lP
from model_configuration.ic_description.ic_market import IC_single_phase_low_pressure as IC_lP


# Main directives
case_name = "case_mP"
geometry_case = "vertical"

final_times = {
"horizontal" : [91250.0, 43800.0, 547500.0],
"vertical" : [273750.0, 127750.0, 547500.0]
}

day = 86400
# Configuration dictionary mapping cases to their specific classes
simulation_cases = {
    "case_hP": {
        "tf":  final_times[geometry_case][0] * day,  # final time [250 years]
        "dt":  365.0 * day,  # final time [1 years]
        "bc": BC_hP,
        "ic": IC_hP
    },
    "case_mP": {
        "tf":  final_times[geometry_case][1] * day,  # final time [120 years]
        "dt":  365.0 * day,  # final time [1 years]
        "bc": BC_mP,
        "ic": IC_mP
    },
    "case_lP": {
        "tf":  final_times[geometry_case][2] * day,  # final time [1500 years]
        "dt":  365.0 * day,  # final time [1 years]
        "bc": BC_lP,
        "ic": IC_lP
    },
}

geometry_cases = {
    "horizontal": ModelGeometryH,
    "vertical": ModelGeometryV,
}


tf = simulation_cases[case_name]["tf"]
dt = simulation_cases[case_name]["dt"]
BoundaryConditions = simulation_cases[case_name]["bc"]
InitialConditions = simulation_cases[case_name]["ic"]
ModelGeometry = geometry_cases[geometry_case]

time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    {
        "permeability": 1.0e-15,
        "porosity": 0.1,
        "thermal_conductivity": 2.0 * 1.0e-6,
        "density": 2700.0,
        "specific_heat_capacity": 880.0 * 1.0e-6,
    }
)
material_constants = {"solid": solid_constants}
params = {
    "material_constants": material_constants,
    "eliminate_reference_phase": True,  # s_liq eliminated, default is True
    "eliminate_reference_component": True,  # z_H2O eliminated, default is True
    "time_manager": time_manager,
    "prepare_simulation": False,
    "reduce_linear_system_q": False,
    "nl_convergence_tol": np.inf,
    "nl_convergence_tol_res": 1.0e-9,
    "max_iterations": 50,
}


class GeothermalWaterFlowModel(ModelGeometry, BoundaryConditions, InitialConditions, FlowModel):

    def after_nonlinear_convergence(self) -> None:
        day = 86400
        year = 365 * day
        super().after_nonlinear_convergence()
        print("Number of iterations: ", self.nonlinear_solver_statistics.num_iteration)
        print("Time value (year): ", self.time_manager.time / year)
        print("Time index: ", self.time_manager.time_index)
        print("")

    def after_simulation(self):
        self.exporter.write_pvd()


# Instance of the computational model
model = GeothermalWaterFlowModel(params)

parametric_space_ref_level = 2
file_name_prefix = (
    "model_configuration/constitutive_description/driesner_vtk_files/"
)
file_name_phz = (
    file_name_prefix + "XHP_l" + str(parametric_space_ref_level) + "_modified_low_salt_content.vtk"
)
file_name_ptz = (
    file_name_prefix + "XTP_l" + str(parametric_space_ref_level) + "_modified_low_salt_content.vtk"
)

brine_sampler_phz = VTKSampler(file_name_phz)
brine_sampler_phz.conversion_factors = (1.0, 1.0e+3, 10.0)  # (z,h,p)
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
mn = model.darcy_flux(model.mdg.subdomains()).value(model.equation_system)

inlet_idx, outlet_idx = model.get_inlet_outlet_sides(model.mdg.subdomains()[0])
print("Inflow values : ", mn[inlet_idx])
print("Outflow values : ", mn[outlet_idx])
