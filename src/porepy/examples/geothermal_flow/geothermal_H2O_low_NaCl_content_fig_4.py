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

# Figure 4 single with low pressure (lP) condition
# Figure 4 single with moderate pressure (mP) condition
# Figure 4 single with high pressure (hP) condition
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (  # noqa: E501
    BC_single_phase_high_pressure as BC_hP,
)
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (  # noqa: E501
    BC_single_phase_low_pressure as BC_lP,
)
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (  # noqa: E501
    BC_single_phase_moderate_pressure as BC_mP,
)
from porepy.examples.geothermal_flow.model_configuration.DriesnerModelConfiguration import (  # noqa: E501
    DriesnerBrineFlowModel as FlowModel,
)

from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (  # noqa: E501
    IC_single_phase_high_pressure as IC_hP,
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (  # noqa: E501
    IC_single_phase_low_pressure as IC_lP,
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (  # noqa: E501
    IC_single_phase_moderate_pressure as IC_mP,
)
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler

# Main directives
case_name = "case_mP"
geometry_case = "vertical"

final_times = {
    "horizontal": [91250.0, 43800.0, 547500.0],  # final time [250, 120, 1500 years]
    "vertical": [273750.0, 127750.0, 547500.0],  # final time [750, 350, 1500 years]
}

day_to_second = 86400
to_Mega = 1.0e-6
# Configuration dictionary mapping cases to their specific classes
simulation_cases = {
    "case_hP": {
        "tf": final_times[geometry_case][0] * day_to_second,  # final time [second]
        "dt": 365.0 * day_to_second,  # final time [second]
        "bc": BC_hP,
        "ic": IC_hP,
    },
    "case_mP": {
        "tf": final_times[geometry_case][1] * day_to_second,  # final time [second]
        "dt": 365.0 * day_to_second,  # final time [second]
        "bc": BC_mP,
        "ic": IC_mP,
    },
    "case_lP": {
        "tf": final_times[geometry_case][2] * day_to_second,  # final time [seconds]
        "dt": 365.0 * day_to_second,  # final time [1 years]
        "bc": BC_lP,
        "ic": IC_lP,
    },
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

solid_constants = pp.SolidConstants(
    permeability=1e-15,
    porosity=0.1,
    thermal_conductivity=2.0 * to_Mega,
    density=2700.0,
    specific_heat_capacity=880.0 * to_Mega,
)
material_constants = {"solid": solid_constants}
params = {
    "material_constants": material_constants,
    "fractional_flow": True,
    "time_manager": time_manager,
    "prepare_simulation": False,
    "apply_schur_complement_reduction": False,
    "nl_convergence_tol": np.inf,
    "nl_convergence_tol_res": 1.0e-4,
    "max_iterations": 100,
}


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

    def gravity_field(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        g_constant = pp.GRAVITY_ACCELERATION
        val = self.units.convert_units(g_constant, "m*s^-2") * to_Mega
        size = np.sum([g.num_cells for g in subdomains]).astype(int)
        gravity_field = pp.wrap_as_dense_ad_array(val, size=size)
        gravity_field.set_name("gravity_field")
        return gravity_field


# Instance of the computational model
model = GeothermalWaterFlowModel(params)

parametric_space_ref_level = 1
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
