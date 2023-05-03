from datetime import datetime

import porepy as pp

timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M")
file_name = "cf_test"  # + timestamp
params = {
    "folder_name": "/mnt/c/Users/vl-work/Desktop/sim-results/" + file_name + "/",
    "file_name": file_name,
    "use_ad": True,
    "eliminate_ref_phase": True,
    "use_pressure_equation": True,
    "monolithic": False,
}

t = 0.0
T = 2.2
dt = 0.1  # T / 1e2
max_iter = 70
tol = 5e-5

model = pp.CompositionalFlowModel(params=params)

model.dt = dt
model.prepare_simulation()
# cond_start = list()
# cond_end = list()

while t < T:
    print(".. Timestep t=%f , dt=%e" % (t, model.dt))
    model.before_newton_loop()

    # A, b = model.ad_system.assemble_subsystem(
    #     equations=model._system_equations, variables=model._system_vars
    # )
    # cond_start.append(np.linalg.cond(A.todense()))

    for i in range(1, max_iter + 1):
        model.before_newton_iteration()
        DX = model.assemble_and_solve_linear_system(tol)
        if model.converged:
            print(f"Success flow after iteration {i - 1}.")
            model.after_newton_convergence(DX, tol, i - 1)
            break
        print(f".. .. flow iteration {i}.")
        model.after_newton_iteration(DX, i)

    if not model.converged:
        print(f"FAILURE: flow at time {t} after {max_iter} iterations.")
        # model.print_x("Flow convergence failure")
        model.after_newton_failure(DX, tol, max_iter)
        model.dt = model.dt / 2
        print(f"Halving timestep to {model.dt}")
        if model.dt < 0.001:
            model.after_simulation()
            raise RuntimeError(
                "Time step halving due to convergence failure reached critical value."
            )
    else:
        t += model.dt
        # A, b = model.ad_system.assemble_subsystem(
        #     equations=model._system_equations, variables=model._system_vars
        # )
        # cond_end.append(np.linalg.cond(A.todense()))

    if t >= T:
        print(f"Reached and of simulation: t={t}")
# cond_start = np.array(cond_start)
# cond_end = np.array(cond_end)
# print("CONDITION NUMBERS: ")
# print("\tAt beginning of iterations:")
# print(f"\tMin: {'{:.4e}'.format(np.min(cond_start))}\n\t"
# + f"Max: {'{:.4e}'.format(np.max(cond_start))}\n\t"
# + f"Mean: {'{:.4e}'.format(np.mean(cond_start))}")
# print("\tAt converged state:")
# print(f"\tMin: {'{:.4e}'.format(np.min(cond_end))}\n\t"
# + f"Max: {'{:.4e}'.format(np.max(cond_end))}\n\t"
# + f"Mean: {'{:.4e}'.format(np.mean(cond_end))}")
model.after_simulation()
