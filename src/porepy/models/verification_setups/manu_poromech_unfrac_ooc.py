import matplotlib.pyplot as plt
import numpy as np

import porepy as pp
from porepy.models.verification_setups.manu_poromech_unfrac import ManuPoromechanics2d

mesh_sizes = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]
error_pressure = []
error_displacement = []
error_flux = []
error_force = []

for mesh_size in mesh_sizes:
    mesh_arguments = {"mesh_size_frac": mesh_size, "mesh_size_bound": mesh_size}
    fluid = pp.FluidConstants({"compressibility": 0.0})
    solid = pp.SolidConstants(
        {"porosity": 0.1, "biot_coefficient": 1.0, "permeability": 1000}
    )
    material_constants = {"fluid": fluid, "solid": solid}
    params = {
        "plot_results": False,
        "stored_times": [1.0],
        "mesh_arguments": mesh_arguments,
        "time_manager": pp.TimeManager([0, 1], 1, True),
        "material_constants": material_constants,
    }

    setup = ManuPoromechanics2d(params)
    pp.run_time_dependent_model(setup, params)
    error_pressure.append(setup.results[-1].error_pressure)
    error_displacement.append(setup.results[-1].error_displacement)
    error_flux.append(setup.results[-1].error_flux)
    error_force.append(setup.results[-1].error_force)


#%% Plot
plt.subplot(121)
# -----> New models

# rate = 1
# x1 = np.log2(1/0.1)
# x2 = np.log2(1/0.0125)
# y1 = -4
# y2 = y1 - rate * (x2 - x1)
# plt.plot([x1, x2], [y1, y2], "k-", linewidth=4, label="First order")

rate = 2
x1 = np.log2(1 / 0.2)
x2 = np.log2(1 / 0.00625)
y1 = -6.5
y2 = y1 - rate * (x2 - x1)
plt.plot([x1, x2], [y1, y2], "k--", linewidth=4, label="Second order")

plt.plot(
    np.log2(1 / np.array(mesh_sizes)),
    np.log2(np.array(error_pressure)),
    "r-o",
    label="Pressure",
)
plt.plot(
    np.log2(1 / np.array(mesh_sizes)),
    np.log2(np.array(error_flux)),
    "b-o",
    label="Darcy flux",
)
plt.plot(
    np.log2(1 / np.array(mesh_sizes)),
    np.log2(np.array(error_displacement)),
    "g-o",
    label="Displacement",
)
plt.plot(
    np.log2(1 / np.array(mesh_sizes)),
    np.log2(np.array(error_force)),
    "m-o",
    label="Poroelastic force",
)

plt.xlabel("log2(1/h)")
plt.ylabel("log2(L2-discrete relative error)")
plt.legend(bbox_to_anchor=(2.2, 0.5), loc="right", ncol=1)
plt.show()
