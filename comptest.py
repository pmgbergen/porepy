import sys
from datetime import datetime

sys.path.append("/mnt/c/Users/vl-work/Desktop/github/porepy/src")

import porepy as pp

timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M")
file_name = "cf_test" # + timestamp
params = {
    "folder_name": "/mnt/c/Users/vl-work/Desktop/sim-results/" + file_name + "/",
    "file_name": file_name,
    "use_ad": False,
    "eliminate_ref_phase": True,
    "use_pressure_equation": True,
    "monolithic": True,
}

t = 0.
T = 10.
dt = T / 1e2
max_iter = 200
tol = 1e-5

model = pp.CompositionalFlowModel(params=params)

model.dt = dt
model.prepare_simulation()

while t < T:
    print(".. Timestep t=%f , dt=%e" % (t, model.dt))
    model.before_newton_loop()

    for i in range(1, max_iter + 1):
        model.before_newton_iteration()
        dx = model.assemble_and_solve_linear_system(tol)
        if model.converged:
            print(f"Success flow after iteration {i - 1}.")
            # model.print_x("convergence SUCCESS")
            model.after_newton_convergence(dx, tol, i - 1)
            break
        print(f".. .. flow iteration {i}.")
        model.after_newton_iteration(dx, i)

    if not model.converged:
        print(f"FAILURE: flow at time {t} after {max_iter} iterations.")
        # model.print_x("Flow convergence failure")
        model.after_newton_failure(dx, tol, max_iter)
        model.dt = model.dt / 2
        print(f"Halving timestep to {model.dt}")
        if model.dt < 0.001:
            model.after_simulation()
            raise RuntimeError("Time step halving due to convergence failure reached critical value.")
    else:
        t += model.dt
    
    if t >= T:
        print(f"Reached and of simulation: t={t}")

model.after_simulation()
