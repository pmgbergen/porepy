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
T = 15.0
max_dt = 0.3
min_dt = 1e-4
dt = 0.05
simulation_success = False

max_iter = 20
flah_max_iter = 70
flash_armijo_iter = 100
faile_counter = 0
first_time_min_dt = True

tol = 5e-3
relaxed_tol = 1e-2
flash_tol = 1e-4
relaxed_flash_tol = 1e-3
use_iterate = (True,)

model = pp.CompositionalFlowModel(params=params, verbosity=1)

model.dt.value = dt
model.prepare_simulation()

while t < T:
    logger.info(f"\n.. Time t={t} , dt = {dt} , failures = {faile_counter}\n")
    model.before_newton_loop()

    for _ in range(1, max_iter + 1):
        if faile_counter >= 2:
            itol = relaxed_tol
            ftol = relaxed_flash_tol
        else:
            itol = tol
            ftol = flash_tol

        flash_success, DX = model.before_newton_iteration(
            flash_tol=ftol,
            flash_max_iter=flah_max_iter,
            flash_armijo_iter=flash_armijo_iter,
            use_iterate=use_iterate,
        )
        if not flash_success:
            break

        DX = model.assemble_and_solve_linear_system(itol)

        if model.converged:
            model.after_newton_convergence(DX)
            break
        model.after_newton_iteration(DX)

    if not model.converged:
        faile_counter += 1
        if faile_counter >= 2:
            use_iterate = False
        model.after_newton_failure(DX)
        dt *= 0.75
        model.dt.value = dt
        if dt < min_dt and not first_time_min_dt:
            logger.warning(
                f"\nDid not converge for min time step and increased tolerance. Aborting simulation.\n"
            )
            break
        elif dt < min_dt and first_time_min_dt:
            if faile_counter < 2:
                faile_counter = 2
            first_time_min_dt = False
            dt = min_dt
            model.dt.value = dt
            logger.warning(f"\nReached minimal admissible time step size {min_dt}.")
        else:
            logger.info(f"\nReducing timestep to {dt}")
    else:
        faile_counter = 0
        use_iterate = True
        t += dt
        # gradually increase step size if it converges and if it was decrease previously.
        if dt < max_dt:
            dt *= 1.5
            if dt > max_dt:
                dt = max_dt
            logger.info(f"\nRelaxing timestep to {dt}")
            model.dt.value = dt
        # resetting flag to allow the procedure to do it again with min_dt if necessary
        first_time_min_dt = True

    if t >= T:
        simulation_success = True
        logger.info(f"\nReached simulation end time t={t}\n")

logger.info(f"\nSimulation success: {simulation_success}\n")
model.after_simulation()
