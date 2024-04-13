"""Example implementing a multi-phase multi component flow of H2O-NaCl using
the Driesner correlations as constitutive laws.

This model uses pressure, specific fluid mixture enthalpy and NaCl overall fraction as
primary variables.

No equilibrium calculations included.

Ergo, the user must close the model to provide expressions for saturations, partial
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

import time

import numpy as np

import porepy as pp

day = 86400
t_scale = 0.00001
time_manager = pp.TimeManager(
    schedule=[0.0, 100.0 * day * t_scale],
    dt_init=1.0 * day * t_scale,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

tracer_like_setting_q = True
if tracer_like_setting_q:
    from TracerModelConfiguration import TracerFlowModel as FlowModel
else:
    from DriesnerBrineOBL import DriesnerBrineOBL
    from DriesnerModelConfiguration import DriesnerBrineFlowModel as FlowModel

solid_constants = pp.SolidConstants(
    {"permeability": 9.869233e-14, "porosity": 0.2, "thermal_conductivity": 1.92}
)
material_constants = {"solid": solid_constants}
params = {
    "material_constants": material_constants,
    "eliminate_reference_phase": True,  # s_liq eliminated, default is True
    "eliminate_reference_component": True,  # z_H2O eliminated, default is True
    "time_manager": time_manager,
    "prepare_simulation": False,
    "reduce_linear_system_q": False,
    "nl_convergence_tol": 1.0e-3,
    "max_iterations": 25,
}


class GeothermalFlowModel(FlowModel):

    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        res_norm = np.linalg.norm(
            model.equation_system.assemble(evaluate_jacobian=False)
        )
        super().after_nonlinear_convergence(solution, errors, iteration_counter)
        print("Time step converged with residual norm: ", res_norm)
        print("Time value: ", self.time_manager.time)
        print("Time index: ", self.time_manager.time_index)
        print("")

    def _export(self):
        if hasattr(self, "exporter"):
            self.exporter.write_vtu(self.primary_variable_names(), time_dependent=True)

    def after_simulation(self):
        self.exporter.write_pvd()


if tracer_like_setting_q:
    model = GeothermalFlowModel(params)
else:
    model = GeothermalFlowModel(params)
    file_name = "binary_files/PHX_l0_with_gradients.vtk"
    brine_obl = DriesnerBrineOBL(file_name)
    brine_obl.conversion_factors = (1.0, 1.0e-3, 1.0e-5)  # (z,h,p)
    model.obl = brine_obl

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
