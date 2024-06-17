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
import time

os.environ["NUMBA_DISABLE_JIT"] = str(0)

import matplotlib.pyplot as plt
import numpy as np

import porepy as pp

tracer_like_setting_q = True
if tracer_like_setting_q:
    from TracerModelConfiguration import TracerFlowModel as FlowModel
else:
    from DriesnerBrineOBL import DriesnerBrineOBL
    from DriesnerModelConfiguration import DriesnerBrineFlowModel as FlowModel

day = 86400
t_scale = 1.0
tf = 0.025 * day * t_scale
dt = 0.025 * day * t_scale
time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    {"permeability": 5.0e-14, "porosity": 0.1, "thermal_conductivity": 1.8, 'density': 2650.0, 'specific_heat_capacity': 1000.0}
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
    "nl_convergence_tol": np.inf,
    "nl_convergence_tol_res": 1.0e-5,
    "max_iterations": 25,
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


if tracer_like_setting_q:
    model = GeothermalFlowModel(params)
else:
    model = GeothermalFlowModel(params)
    file_name = "binary_files/PHX_l2_with_gradients.vtk"
    brine_obl = DriesnerBrineOBL(file_name)
    brine_obl.conversion_factors = (1.0, 1.0e-3, 1.0e-5)  # (z,h,p)
    model.obl = brine_obl

    if False:
        h = np.arange(1.5e6, 3.2e6, 0.025e6)
        p = 20.0e6 * np.ones_like(h)

        # p = np.arange(1.0e6, 20.0e6, 0.5e6)
        # h = 3.2e6 * np.ones_like(p)

        z_NaCl = (0.001 + 1.0e-5) * np.ones_like(h)
        par_points = np.array((z_NaCl, h, p)).T
        brine_obl.sample_at(par_points)

        T = brine_obl.sampled_could.point_data["Temperature"]
        plt.plot(
            h,
            T,
            label="T(H)",
            color="blue",
            linestyle="-",
            marker="o",
            markerfacecolor="blue",
            markersize=5,
        )

        s_l = brine_obl.sampled_could.point_data["S_l"]
        s_v = brine_obl.sampled_could.point_data["S_v"]
        # plt.plot(h, s_l, label='Liquid', color='blue', linestyle='-', marker='o',
        #          markerfacecolor='blue', markersize=5)
        # plt.plot(h, s_v, label='Vapor', color='red', linestyle='-', marker='o',
        #          markerfacecolor='red', markersize=5)

        h_l = brine_obl.sampled_could.point_data["H_l"]
        h_v = brine_obl.sampled_could.point_data["H_v"]
        # plt.plot(p, h_l, label='Liquid', color='blue', linestyle='-', marker='o',
        #          markerfacecolor='blue', markersize=5)
        # plt.plot(p, h_v, label='Vapor', color='red', linestyle='-', marker='o',
        #          markerfacecolor='red', markersize=5)

        # dTdh = 1/brine_obl.sampled_could.point_data['grad_Temperature'][:,1]
        # plt.plot(h, dTdh, label='dTdh(H)', color='red', linestyle='-', marker='o',
        #          markerfacecolor='red', markersize=5)
        plt.legend()
        plt.show()
        aka = 0


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

mn = model.darcy_flux(model.mdg.subdomains()).value(model.equation_system)[model.domain_boundary_sides(model.mdg.subdomains()[0]).north]
print("normal flux: ", mn)