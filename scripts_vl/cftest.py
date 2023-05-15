from datetime import datetime

import porepy as pp

from porepy.models.compositional_flow_model import logger

timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M")
file_name = "cf_test"  # + timestamp
params = {
    "folder_name": "/mnt/c/Users/vl-work/Desktop/sim-results/" + file_name + "/",
    "file_name": file_name,
}

t = 0.0
T = 2.2
dt = 0.1  # T / 1e2
max_iter = 70
tol = 5e-7

model = pp.CompositionalFlowModel(params=params, verbosity=1)

model.dt.value = dt
model.prepare_simulation()
# cond_start = list()
# cond_end = list()

while t < T:
    logger.info(f"\n.. Timestep t={t} ; dt={dt}\n")
    model.before_newton_loop()

    # A, b = model.ad_system.assemble_subsystem(
    #     equations=model._system_equations, variables=model._system_vars
    # )
    # cond_start.append(np.linalg.cond(A.todense()))

    for _ in range(1, max_iter + 1):
        model.before_newton_iteration()
        DX = model.assemble_and_solve_linear_system(tol)
        if model.converged:
            model.after_newton_convergence(DX)
            break
        model.after_newton_iteration(DX)

    if not model.converged:
        model.after_newton_failure(DX)
        dt /= 2
        model.dt.value = dt
        logger.info(f"Halving timestep to {dt}")
        if dt < 0.001:
            model.after_simulation()
            raise RuntimeError(
                "Time step halving due to convergence failure reached critical value."
            )
    else:
        t += dt
        # A, b = model.ad_system.assemble_subsystem(
        #     equations=model._system_equations, variables=model._system_vars
        # )
        # cond_end.append(np.linalg.cond(A.todense()))

    if t >= T:
        logger.info(f"Reached simulation end time t={t}")
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
