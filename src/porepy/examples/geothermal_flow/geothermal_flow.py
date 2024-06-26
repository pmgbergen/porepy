"""Example implementing a multi-phase multi component flow of H2O-NaCl using Driesner
correlations and a tracer-like as constitutive descriptions.

This model uses pressure, specific fluid mixture enthalpy and NaCl overall fraction as
primary variables.

No equilibrium calculations included.

Ergo, the user must close the model to provide expressions for saturation, partial
fractions and temperature, depending on primary variables.

Note:
    With some additional work, it is straight forward to implement a model without
    h as the primary variable, but T.

    What needs to change is:

    1. Overwrite
       porepy.models.compositional_flow.VariablesCF
       mixin s.t. it does not create a h variable.
    2. Modify accumulation term in
       porepy.models.compositional_flow.TotalEnergyBalanceEquation_h
       to use T, not h.
    3. H20_NaCl_brine.dependencies_of_phase_properties: Use T instead of h.

"""

from __future__ import annotations
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
import time

import numpy as np
from model_configuration.DriesnerModelConfiguration import (
    DriesnerBrineFlowModel as FlowModel,
)
from vtk_sampler import VTKSampler

import porepy as pp

day = 86400
tf = 0.05 * day
dt = 0.00025 * day
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
    "nl_convergence_tol_res": 1.0e-3,
    "max_iterations": 50,
}


class GeothermalFlowModel(FlowModel):

    def after_nonlinear_convergence(self, iteration_counter) -> None:
        tb = time.time()
        _, residual = self.equation_system.assemble(evaluate_jacobian=True)
        res_norm = np.linalg.norm(residual)
        te = time.time()
        print("Elapsed time assemble: ", te - tb)
        print("Time step converged with residual norm: ", res_norm)
        print("Number of iterations: ", iteration_counter)
        print("Time value: ", self.time_manager.time)
        print("Time index: ", self.time_manager.time_index)
        print("")
        super().after_nonlinear_convergence(iteration_counter)

    def after_simulation(self):
        self.exporter.write_pvd()


# Instance of the computational model
model = GeothermalFlowModel(params)

parametric_space_ref_level = 2
file_name_prefix = "model_configuration/constitutive_description/driesner_vtk_files/"
file_name = (
    file_name_prefix + "XHP_l" + str(parametric_space_ref_level) + "_modified.vtk"
)
brine_sampler = VTKSampler(file_name)
brine_sampler.conversion_factors = (1.0, 1.0e-3, 1.0e-5)  # (z,h,p)
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

# Check conservation of overall mass across boundaries
# external_bc_idx = bc_sides.all_bf
# assert np.isclose(np.sum(mn[external_bc_idx]), 0.0, atol=1.0e-10)


