import matplotlib.pyplot as plt
import numpy as np
from time import time

import porepy as pp
from porepy.models.verification_setups.manu_poromech_unfrac import ManuPoromechanics2d

# from porepy.models.verification_setups.manu_poromech_unfrac_old import
# ManufacturedBiot

# Convergence analysis paramaters
mesh_sizes = [0.20, 0.10, 0.05, 0.025, 0.0125, 0.00625]
time_steps = [1, 1, 1, 1, 1, 1]

# New models
error_pressure = []
error_displacement = []
error_flux = []
error_force = []

# Old models
# error_pressure_old = []
# error_displacement_old = []
# error_flux_old = []
# error_force_old = []

for mesh_size, time_step in zip(mesh_sizes, time_steps):

    # ----> New models
    print(f"Solving for mesh size {mesh_size} [m].")
    tic = time()
    mesh_arguments = {"mesh_size_frac": mesh_size, "mesh_size_bound": mesh_size}
    fluid = pp.FluidConstants({"compressibility": 0.2})
    solid = pp.SolidConstants({"biot_coefficient": 1.0})
    material_constants = {"fluid": fluid, "solid": solid}
    params = {
        "plot_results": False,
        "mesh_arguments": mesh_arguments,
        "material_constants": material_constants,
        "time_manager": pp.TimeManager([0, 1], time_step, True),
    }
    setup = ManuPoromechanics2d(params)
    pp.run_time_dependent_model(setup, params)
    error_pressure.append(setup.results[-1].error_pressure)
    error_displacement.append(setup.results[-1].error_displacement)
    error_flux.append(setup.results[-1].error_flux)
    error_force.append(setup.results[-1].error_force)
    toc = time()
    print(f"Simulation finished in {round(toc-tic)} [s].")

    # ----> Old models
    # print(f"Solving for mesh size {mesh_size} [m].")
    # tic = time()
    # mesh_arguments = {"mesh_size_frac": mesh_size, "mesh_size_bound": mesh_size}
    # params = {
    #     "plot_results": False,
    #     "use_ad": True,
    #     "stored_times": [1],
    #     "mesh_arguments": mesh_arguments,
    #     "biot_coefficient": 1.0
    # }
    # setup = ManufacturedBiot(params)
    # pp.run_time_dependent_model(setup, params)
    # error_pressure_old.append(setup.results[-1].error_pressure)
    # error_displacement_old.append(setup.results[-1].error_displacement)
    # error_flux_old.append(setup.results[-1].error_flux)
    # error_force_old.append(setup.results[-1].error_force)
    # toc = time()
    # print(f"Simulation finished in {round(toc - tic)} [s].")


#%% Plot
plt.subplot(121)

# ----> Convergence rate lines
# First order
rate = 1
x1 = np.log2(1 / mesh_sizes[0])
x2 = np.log2(1 / mesh_sizes[-1])
y1 = -3.5
y2 = y1 - rate * (x2 - x1)
plt.plot([x1, x2], [y1, y2], "k-", linewidth=4, label="First order")

# Second order
rate = 2
x1 = np.log2(1 / mesh_sizes[0])
x2 = np.log2(1 / mesh_sizes[-1])
y1 = -6.5
y2 = y1 - rate * (x2 - x1)
plt.plot([x1, x2], [y1, y2], "k--", linewidth=4, label="Second order")

# ----> New models
plt.plot(
    np.log2(1 / np.array(mesh_sizes)),
    np.log2(np.array(error_pressure)),
    "ro-",
    label="Pressure",
)
plt.plot(
    np.log2(1 / np.array(mesh_sizes)),
    np.log2(np.array(error_flux)),
    "bo-",
    label="Darcy flux",
)
plt.plot(
    np.log2(1 / np.array(mesh_sizes)),
    np.log2(np.array(error_displacement)),
    "go-",
    label="Displacement",
)
plt.plot(
    np.log2(1 / np.array(mesh_sizes)),
    np.log2(np.array(error_force)),
    "m-o",
    label="Poroelastic force",
)

# # ----> Old models
# plt.plot(
#     np.log2(1 / np.array(mesh_sizes)),
#     np.log2(np.array(error_pressure_old)),
#     "r-",
#     label="Pressure (Old)",
# )
# plt.plot(
#     np.log2(1 / np.array(mesh_sizes)),
#     np.log2(np.array(error_flux)),
#     "b-",
#     label="Darcy flux (Old)",
# )
# plt.plot(
#     np.log2(1 / np.array(mesh_sizes)),
#     np.log2(np.array(error_displacement)),
#     "g-",
#     label="Displacement (Old)",
# )
# plt.plot(
#     np.log2(1 / np.array(mesh_sizes)),
#     np.log2(np.array(error_force)),
#     "m-",
#     label="Poroelastic force (Old)",
# )
# plt.text(9.2, -4, r"$p = t x (1-x) y (1-y)$", fontsize=13)
# plt.text(9.2, -5.5, r"$u_x = t x (1-x) y (1-y)$", fontsize=13)
# plt.text(9.2, -7, r"$u_y = t x (1-x) y (1-y)$", fontsize=13)


plt.xlabel("log2(1/h)")
plt.ylabel("log2(L2-discrete relative error)")
plt.legend(bbox_to_anchor=(2.25, 0.5), loc="right", ncol=1, fontsize=11)
# plt.title(r"$c_f = 0~[Pa^{-1}] \quad \tau = 1~[s] \quad \alpha = 1.0~[-]$")
plt.show()
