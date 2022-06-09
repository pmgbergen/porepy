import sys
from datetime import datetime

sys.path.append("/mnt/c/Users/vl-work/Desktop/github/porepy/src")

import porepy as pp

timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
params = {
    "folder_name": "/mnt/c/Users/vl-work/Desktop/sim-results/",
    "file_name": "gt_vl_" + timestamp,
    "use_ad": True,
}

k_value = 0.5
monolithic = True
use_TRU = False
elimination = ("xi", "molar_phase_fraction_sum", "min")
elimination = None
max_iter_equilibrium = 200
tol_equilibrium = 1e-8

t = 0
T = 100
dt = 0.01
max_iter = 200
tol = 1e-7

model = pp.CompositionalFlowModel(
    params=params,
    k_value=k_value,
    monolithic_solver=monolithic,
    max_iter_flash=max_iter_equilibrium,
    tol_flash=tol_equilibrium,
)

model.prepare_simulation()
model.dt = t
equilibrated = model.solve_equilibrium(
    max_iter_equilibrium,
    tol_equilibrium,
    use_TRU=use_TRU,
    eliminate_unitary=elimination,
)
if not equilibrated:
    raise RuntimeError("Equilibrium calculations failed at time %s" % (str(t)))

while t < T:
    model.before_newton_loop()
    i_final = 0

    for i in range(max_iter):
        model.before_newton_iteration()
        dx = model.assemble_and_solve_linear_system(tol)
        if model.convergence_status:
            print("Newton converged: t=%f , iterations=%i" % (t, i))
            model._print("convergence SUCCESS")
            model.after_newton_convergence(dx, tol, i)
            i_final = i
            break
        model.after_newton_iteration(dx)

        if not monolithic:
            equilibrated = model.solve_equilibrium(
                max_iter_equilibrium,
                tol_equilibrium,
                use_TRU=use_TRU,
                eliminate_unitary=elimination,
            )
            if not equilibrated:
                raise RuntimeError(
                    "Equilibrium calculations failed at time %s" % (str(t))
                )

    if not model.convergence_status:
        model._print("convergence failure")
        model.after_newton_failure(dx, tol, i_final)
        model.dt = model.dt / 2
        print("Newton FAILED: t=%f , max iter=%i.\n Halving timestep to %f"
        % (t, max_iter, model.dt))
        if model.dt < 0.00001:
            raise RuntimeError("Time step halving due to convergence failure reached critical value.")
    else:
        t += model.dt
