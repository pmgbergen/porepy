import sys
sys.path.append("/mnt/c/Users/vl-work/Desktop/github/porepy/src")

import porepy as pp

from datetime import datetime

timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
params = {
    "folder_name": "/mnt/c/Users/vl-work/Desktop/sim-results/",
    "file_name": "gt_vl_" + timestamp,
    "use_ad": True
}

model = pp.CompositionalFlowModel(params=params)
model.prepare_simulation()

t = 0
T = 100
max_iter = 50
tol = 1e-7

max_iter_equilibrium = 200
tol_equilibrium = 1e-8

while t < T:
    model.before_newton_loop()
    i_final = 0

    equilibrated = model.solve_equilibrium(max_iter_equilibrium, tol_equilibrium)

    if not equilibrated:
        # raise RuntimeError("Equilibrium calculations failed at time %s" % (str(t)))
        pass
    
    for i in range(max_iter):
        model.before_newton_iteration()
        dx = model.assemble_and_solve_linear_system(tol)
        if model.convergence_status:
            model.after_newton_convergence(dx, tol, i)
            i_final = i
            break
        model.after_newton_iteration(dx)

    if not model.convergence_status:
        model.after_newton_failure(dx, tol, i_final)
        model.dt = model.dt / 2
    else:
        t += model.dt
