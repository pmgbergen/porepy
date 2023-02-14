import numpy as np
import porepy as pp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import time

average_iter_time = list()
total_flash_time = list()
iterations = list()

cells = [1, 10, 100]

temperature = 300
pressure = 0.1
co2_fraction = 0.01
h2o_fraction = 0.99

for nc in cells:

    M = pp.composite.PR_Composition(nc=nc)
    adsys = M.ad_system
    vec = np.ones(nc)

    h2o = pp.composite.H2O(adsys)
    co2 = pp.composite.CO2(adsys)
    LIQ = pp.composite.PR_Phase(adsys, False, name="L")
    GAS = pp.composite.PR_Phase(adsys, True, name="G")

    M.add_components([h2o, co2])
    M.add_phases([LIQ, GAS])

    adsys.set_variable_values(
        h2o_fraction * vec, variables=[h2o.fraction_name], to_iterate=True, to_state=True
    )
    adsys.set_variable_values(
        co2_fraction * vec, variables=[co2.fraction_name], to_iterate=True, to_state=True
    )

    adsys.set_variable_values(
        temperature * vec, variables=[M.T_name], to_iterate=True, to_state=True
    )
    adsys.set_variable_values(
        pressure * vec, variables=[M.p_name], to_iterate=True, to_state=True
    )
    adsys.set_variable_values(0 * vec, variables=[M.h_name], to_iterate=True, to_state=True)

    M.initialize()

    FLASH = pp.composite.Flash(M, auxiliary_npipm=False)
    FLASH.use_armijo = True
    FLASH.armijo_parameters["rho"] = 0.99
    FLASH.armijo_parameters["j_max"] = 50
    FLASH.armijo_parameters["return_max"] = True
    FLASH.newton_update_chop = 1.0
    FLASH.flash_tolerance = 1e-7
    FLASH.max_iter_flash = 50

    start = time.time()
    FLASH.flash("isothermal", "npipm", "rachford_rice", True, False)
    stop = time.time()

    total_flash_time.append(stop - start)
    average_iter_time.append(sum(FLASH.iter_times) / len(FLASH.iter_times))
    iterations.append(FLASH.flash_history[-1]['iterations'])

gs = gridspec.GridSpec(1,3)

figwidth = 15
fig = plt.figure(figsize=(figwidth, 1080/1920 * figwidth))

ax_iter = plt.subplot(gs[0, 0])
img_iter = ax_iter.plot(cells, iterations, color='black', marker='o', linestyle='-')
ax_iter.set_title('Number of iterations')
ax_iter.set_xlabel('Number of cells')
ax_iter.set_ylabel('Iterations')
ax_iter.set_xticklabels([f"1e{np.log(nc)}" for nc in cells])

ax_tot = plt.subplot(gs[0, 1])
img_tot = ax_tot.plot(cells, total_flash_time, color='black', marker='o', linestyle='-')
ax_tot.set_title('Total execution time of flash')
ax_tot.set_xlabel('Number of cells')
ax_tot.set_ylabel('Time [s]')
ax_tot.set_xticklabels([f"1e{np.log(nc)}" for nc in cells])

ax_avg = plt.subplot(gs[0, 2])
img_avg = ax_avg.plot(cells, average_iter_time, color='black', marker='o', linestyle='-')
ax_avg.set_title('Average time per iteration.')
ax_avg.set_xlabel('Number of cells')
ax_avg.set_ylabel('Time [s]')
ax_avg.set_xticklabels([f"1e{np.log(nc)}" for nc in cells])

fig.tight_layout()
fig.savefig(
    '/mnt/c/Users/vl-work/Desktop/flash_time.png',
    format='png',
    dpi=500
)
fig.show()


print("Done.")