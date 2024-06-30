"""Example implementing a multi-phase multi component flow of H2O-NaCl using Driesner
correlations.

The model relies on pressure (P), specific fluid mixture enthalpy (H), and NaCl overall
composition (z_NaCl) as primary variables.

Equilibrium calculations are included in the correlations. As a result, they contain
expressions for saturation, partial fractions, and temperature based on primary variables.

The correlations are interpolated with VTK using a standalone object (VTKSampler). This
object provides functions and their gradients in the product space (z_NaCl, H, P) in R^3.

"""

from __future__ import annotations
import os
import time
import numpy as np
import porepy as pp
from model_configuration.DriesnerModelConfiguration import (
    DriesnerBrineFlowModel as FlowModel,
)
from vtk_sampler import VTKSampler

os.environ["NUMBA_DISABLE_JIT"] = "1"

day = 86400
tf = 0.001 * day
dt = 0.0001 * day
dynamic_time_step_q = False

if dynamic_time_step_q:
    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=False,
        dt_min_max = (dt, 0.1 * day),
        iter_optimal_range = (5, 10),
        iter_relax_factors = (0.5,1.5),
        recomp_factor = 0.25,
        iter_max=50,
        print_info=True,
    )
else:
    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
        iter_max=50,
        print_info=True,
    )

solid_constants = pp.SolidConstants(
    {
        "permeability": 5.0e-14,
        "porosity": 0.1,
        "thermal_conductivity": 1.8,
        "density": 2650.0,
        "specific_heat_capacity": 1000.0,
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
    "nl_convergence_tol_res": 1.0e-4,
    "max_iterations": 50,
    "petsc_solver_q": True,
    "darcy_flux_discretization": 'mpfa',
    "fourier_flux_discretization": 'mpfa',
}


class GeothermalFlowModel(FlowModel):
    pass

# Instance of the computational model
model = GeothermalFlowModel(params)

parametric_space_ref_level = 2
file_name_prefix = "model_configuration/constitutive_description/driesner_vtk_files/"
file_name = (
    file_name_prefix + "XHP_l" + str(parametric_space_ref_level) + "_modified.vtk"
)
brine_sampler = VTKSampler(file_name)
brine_sampler.conversion_factors = (1.0, 1.0e3, 10.0)  # (z [-], h [kJ/kg], p [MPa])
model.vtk_sampler = brine_sampler


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