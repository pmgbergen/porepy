"""Example implementing a multi-phase multi component flow of H2O-NaCl using a passive

"""

from __future__ import annotations
import os
os.environ["NUMBA_DISABLE_JIT"] = str(0)

import time
from typing import cast

import numpy as np

import porepy as pp
from porepy.examples.geothermal_flow.model_configuration.TracerModelConfiguration import (
    TracerFlowModel as FlowModel,
)

day = 86400
t_scale = 1.0
tf = 2.5 * day
dt = 0.25 * day
time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    permeability=5.0e-14,
    porosity=0.1,
    thermal_conductivity=1.8,
    density=2650.0,
    specific_heat_capacity=1e3,
)
material_constants = {"solid": solid_constants}
params = {
    "rediscretize_darcy_flux" : True,
    "fractional_flow": True,
    "material_constants": material_constants,
    "eliminate_reference_phase": True,  # s_liq eliminated, default is True
    "eliminate_reference_component": True,  # z_H2O eliminated, default is True
    "time_manager": time_manager,
    "prepare_simulation": False,
    "reduce_linear_system": False,
    "nl_convergence_tol": np.inf,
    "nl_convergence_tol_res": 1.0e-5,
    "max_iterations": 25,
}


class TracerLikeFlowModel(FlowModel):
    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()  # type:ignore[safe-super]
        print("Number of iterations: ", self.nonlinear_solver_statistics.num_iteration)
        print("Time value: ", self.time_manager.time)
        print("Time index: ", self.time_manager.time_index)
        print("")

    def after_simulation(self):
        self.exporter.write_pvd()


model = TracerLikeFlowModel(params)

tb = time.time()
model.prepare_simulation()
te = time.time()
print("Elapsed time prepare simulation: ", te - tb)
print("Simulation prepared for total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid employed: ", model.mdg)
model.primary_equations = model.get_primary_equations_cf()
model.primary_variables = model.get_primary_variables_cf()

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
mn = cast(np.ndarray, mn)

inlet_idx, outlet_idx = model.get_inlet_outlet_sides(model.mdg.subdomains()[0])
print("Inflow values : ", mn[inlet_idx])
print("Outflow values : ", mn[outlet_idx])

# Check conservation of overall mass across boundaries
external_bc_idx = bc_sides.all_bf
assert np.isclose(np.sum(mn[external_bc_idx]), 0.0, atol=1.0e-10)
