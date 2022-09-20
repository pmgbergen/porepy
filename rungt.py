import sys
from datetime import datetime

sys.path.append("/mnt/c/Users/vl-work/Desktop/github/porepy/src")

import porepy as pp

timestamp = datetime.now().strftime("%Y_%m_%f__%H_%M_%S")
params = {
    "folder_name": "/mnt/c/Users/vl-work/Desktop/sim-results/",
    "file_name": "gt_vl_" + timestamp,
    "use_ad": True,
}

t = 0.
T = 100
dt = 0.01
max_iter = 200
tol = 1e-8

model = pp.CompositionalFlowModel(
    params=params,
    monolithic_solver=False,
)

model.set_composition()
model.prepare_simulation()
model.dt = dt

while t < T:
    print(".. Timestep t=%f , dt=%e" % (t, model.dt))
    model.before_newton_loop()
    i_final = 0

    for i in range(max_iter):
        model.before_newton_iteration()
        print(".. solving primary system at iteration %i" % (i))
        dx = model.assemble_and_solve_linear_system(tol)
        if model.convergence_status:
            print("Newton converged: t=%f , iterations=%i" % (t, i))
            model._print("convergence SUCCESS")
            model.after_newton_convergence(dx, tol, i)
            i_final = i
            break
        try:
            model.after_newton_iteration(dx, i)
        except RuntimeError as err:
            msg = str(err) + "\nSimulation time: t=%f" % (t)
            raise RuntimeError(msg)

    if not model.convergence_status:
        model._print("convergence failure")
        model.after_newton_failure(dx, tol, i_final)
        model.dt = model.dt / 2
        print("Newton FAILED: t=%f , max iter=%i.\n Halving timestep to %e"
        % (t, max_iter, model.dt))
        if model.dt < 0.00001:
            raise RuntimeError("Time step halving due to convergence failure reached critical value.")
    else:
        t += model.dt
