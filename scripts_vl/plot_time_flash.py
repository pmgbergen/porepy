"""
Important:
    This script assumes local modifications to the Flash class, where respective times
    are stored.

"""
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import porepy as pp

average_iter_time = list()
average_assembly_time = list()
average_solver_time = list()
total_flash_time = list()

cells = [1, 10, 100, 500, 1000]

temperature = 300
pressure = 0.1
co2_fraction = 0.01
h2o_fraction = 0.99

for nc in cells:
    print(f"Calculating for nc={nc}", flush=True)
    M = pp.composite.PengRobinsonMixture(nc=nc)
    ads = M.AD.system
    vec = np.ones(nc)

    h2o = pp.composite.H2O(ads)
    co2 = pp.composite.CO2(ads)
    LIQ = pp.composite.PR_Phase(ads, False, name="L")
    GAS = pp.composite.PR_Phase(ads, True, name="G")

    M.add([h2o, co2], [LIQ, GAS])

    ads.set_variable_values(
        h2o_fraction * vec,
        variables=[h2o.fraction.name],
        to_iterate=True,
        to_state=True,
    )
    ads.set_variable_values(
        co2_fraction * vec,
        variables=[co2.fraction.name],
        to_iterate=True,
        to_state=True,
    )

    M.AD.set_up()

    ads.set_variable_values(
        temperature * vec, variables=[M.AD.T.name], to_iterate=True, to_state=True
    )
    ads.set_variable_values(
        pressure * vec, variables=[M.AD.p.name], to_iterate=True, to_state=True
    )
    ads.set_variable_values(
        0 * vec, variables=[M.AD.h.name], to_iterate=True, to_state=True
    )

    FLASH = pp.composite.Flash(M, auxiliary_npipm=False)
    FLASH.use_armijo = True
    FLASH.armijo_parameters["rho"] = 0.99
    FLASH.armijo_parameters["j_max"] = 50
    FLASH.armijo_parameters["return_max"] = True
    FLASH.newton_update_chop = 1.0
    FLASH.flash_tolerance = 1e-7
    FLASH.max_iter_flash = 50

    start = time.time()
    FLASH.flash("pT", "npipm", "rachford_rice", True, False)
    stop = time.time()

    total_flash_time.append(stop - start)
    average_iter_time.append(sum(FLASH.iter_times) / len(FLASH.iter_times))
    average_assembly_time.append(sum(FLASH.assembly_times) / len(FLASH.assembly_times))
    average_solver_time.append(sum(FLASH.solver_times) / len(FLASH.solver_times))

gs = gridspec.GridSpec(2, 2)

figwidth = 15
fig = plt.figure(figsize=(figwidth, 1080 / 1920 * figwidth))

ax_tot = plt.subplot(gs[0, 0])
img_tot = ax_tot.plot(cells, total_flash_time, color="black", marker="o", linestyle="-")
ax_tot.set_title("Total execution time of flash")
ax_tot.set_xlabel("Number of cells")
ax_tot.set_ylabel("Time [s]")
ax_tot.set_xscale("log")
ax_tot.set_yscale("log")
ax_tot.grid(True, "both", "y")
ax_tot.grid(True, "major", "x")

ax_avg = plt.subplot(gs[0, 1])
img_avg = ax_avg.plot(
    cells, average_iter_time, color="black", marker="o", linestyle="-"
)
ax_avg.set_title("Average time per iteration.")
ax_avg.set_xlabel("Number of cells")
ax_avg.set_ylabel("Time [s]")
ax_avg.set_xscale("log")
ax_avg.set_yscale("log")
ax_avg.grid(True, "both", "y")
ax_avg.grid(True, "major", "x")

ax_avg_ass = plt.subplot(gs[1, 0])
img_avg = ax_avg_ass.plot(
    cells, average_assembly_time, color="black", marker="o", linestyle="-"
)
ax_avg_ass.set_title("Average assembly time.")
ax_avg_ass.set_xlabel("Number of cells")
ax_avg_ass.set_ylabel("Time [s]")
ax_avg_ass.set_xscale("log")
ax_avg_ass.set_yscale("log")
ax_avg_ass.grid(True, "both", "y")
ax_avg_ass.grid(True, "major", "x")

ax_avg_solv = plt.subplot(gs[1, 1])
img_avg = ax_avg_solv.plot(
    cells, average_solver_time, color="black", marker="o", linestyle="-"
)
ax_avg_solv.set_title("Average (direct) solver time.")
ax_avg_solv.set_xlabel("Number of cells")
ax_avg_solv.set_ylabel("Time [s]")
ax_avg_solv.set_xscale("log")
ax_avg_solv.set_yscale("log")
ax_avg_solv.grid(True, "both", "y")
ax_avg_solv.grid(True, "major", "x")

fig.tight_layout()
fig.savefig("/mnt/c/Users/vl-work/Desktop/flash_time.png", format="png", dpi=500)
fig.show()


print("Done.")
