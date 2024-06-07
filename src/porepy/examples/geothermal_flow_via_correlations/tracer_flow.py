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

import time

import numpy as np
from LinearTracerModelConfiguration import LinearTracerFlowModel as FlowModel

import porepy as pp

day = 86400
t_scale = 0.00001
time_manager = pp.TimeManager(
    schedule=[0.0, 10.0 * day * t_scale],
    dt_init=1.0 * day * t_scale,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

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
    "petsc_solver_q": False,
    "nl_convergence_tol": 1.0e-8,
    "nl_convergence_tol_res": 1.0e-8,
    "max_iterations": 25,
}


class GeothermalFlowModel(FlowModel):

    def after_nonlinear_convergence(self) -> None:
        tb = time.time()
        res_norm = np.linalg.norm(
            model.equation_system.assemble(evaluate_jacobian=False)
        )
        te = time.time()
        print("Elapsed time residual assemble: ", te - tb)
        super().after_nonlinear_convergence()
        print("Time step converged with residual norm: ", res_norm)
        print("Time value: ", self.time_manager.time)
        print("Time index: ", self.time_manager.time_index)
        print("")

    def after_simulation(self):
        self.exporter.write_pvd()

    def solve_linear_system(self) -> np.ndarray:
        """After calling the parent method, the global solution is calculated by Schur
        expansion."""
        petsc_solver_q = self.params.get("petsc_solver_q", False)
        tb = time.time()
        if petsc_solver_q:
            from petsc4py import PETSc

            csr_mat, res_g = self.linear_system

            jac_g = PETSc.Mat().createAIJ(
                size=csr_mat.shape,
                csr=((csr_mat.indptr, csr_mat.indices, csr_mat.data)),
            )

            # solving ls
            st = time.time()
            ksp = PETSc.KSP().create()
            ksp.setOperators(jac_g)
            b = jac_g.createVecLeft()
            b.array[:] = res_g
            x = jac_g.createVecRight()

            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")

            ksp.setConvergenceHistory()
            ksp.solve(b, x)
            sol = x.array
        else:
            sol = super().solve_linear_system()

        reduce_linear_system_q = self.params.get("reduce_linear_system_q", False)
        if reduce_linear_system_q:
            raise ValueError("Case not implemented yet.")
        te = time.time()
        print("Elapsed time linear solve: ", te - tb)
        return sol


model = GeothermalFlowModel(params)

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
