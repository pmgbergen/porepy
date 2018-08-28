"""
Error evaluation and visualization for Case 2.
"""
import numpy as np
import csv
import scipy.sparse as sps
from cycler import cycler

import matplotlib
import matplotlib.pyplot as plt


def read_csv(fn, l):
    f = open(fn, "r")
    reader = csv.reader(f)
    v = np.zeros(l)

    for i, r in enumerate(reader):
        # print(r[0])
        v[i] = float(r[0])

    return v


def L2(ref, other, cell_volumes, value_range=None, mapping=None):
    """
    Value_range: 1 absolute error, None: relative error, other: e.g. global
    range
    """
    if mapping is None:
        l2 = np.multiply(np.power(ref - other, 2), cell_volumes)
    else:
        l2 = np.multiply(np.power(ref - mapping * other, 2), cell_volumes)
    l2 = np.power(np.sum(l2), .5)

    if value_range is None:

        value_range = np.amax(ref) - np.amin(ref)

    value_range *= np.power(np.sum(cell_volumes), 1 / 2)

    return l2 / value_range


def inft(ref, other):
    return np.amax(np.absolute(ref - other))


def breakthrough(tvec, threshold):
    log = tvec < threshold
    i = np.where(log)[0][0]
    return i / 100


def small_in_big(c, ny, nsmall):
    i = c // ny
    j = c % ny
    k = int(nsmall / ny)

    cells = (
        i * k * nsmall
        + np.tile(np.arange(j * k, (j + 1) * k), (k, 1))
        + np.outer(np.arange(k), np.ones(k)) * nsmall
    )
    return cells


ny = [4, 8, 16, 32]

yfactor = [1, 2, 4, 6]
deg = [10, 30, 60]
deg = [30]
bt_eq = np.empty((len(ny), len(yfactor), len(deg)))
bt_eq_coarse = np.empty((len(ny), len(yfactor), len(deg)))
bt_red = np.empty((len(ny), len(yfactor), len(deg)))

L2_pressures = np.empty((len(ny), len(yfactor), len(deg)))
L2_temperatures = np.empty((len(ny), len(yfactor), len(deg)))
L2_temperatures_equi = np.empty((len(ny), len(yfactor), len(deg)))
L2_pressures_equi = np.empty((len(ny), len(yfactor), len(deg)))
ny_ref = 256
nc_ref = ny_ref * (ny_ref + 1)
nc_m_ref = ny_ref ** 2
main_folder = "results_equi"
cell_volumes_ref = read_csv(
    main_folder + "/cell_volumes_nx_{}.csv".format(str(ny_ref)), nc_ref
)


frac_cells = np.zeros(nc_ref, dtype=bool)
fracs = np.arange(int(ny_ref / 2 * ny_ref), int((ny_ref / 2 + 1) * ny_ref))
frac_cells[fracs] = True
matrixc_ref = np.logical_not(frac_cells)
cv_ref_matrix = cell_volumes_ref[matrixc_ref]

# Get data and compute errors
for i, n in enumerate(ny):
    n_layer = n
    nc = n_layer * (n + 1)
    nc_m = n ** 2
    matrix_red = np.zeros(nc, dtype=bool)

    matrixc = np.arange(int(n ** 2))
    matrix_red[matrixc] = True
    small_from_big = sps.lil_matrix((nc_m_ref, nc_m), dtype=bool)
    for c in np.arange(nc_m):
        small = small_in_big(c, n, ny_ref)
        small_from_big[small, c] = True

    small_from_big_frac = sps.lil_matrix((ny_ref, n), dtype=bool)
    for c in np.arange(n):
        small = np.int_(c * ny_ref / n + np.arange(ny_ref / n))
        small_from_big_frac[small, c] = True

    reordering = np.concatenate(
        (
            np.arange(0, int(n / 2 * n_layer)),
            np.arange(n * n_layer, (n + 1) * n_layer),
            np.arange(n / 2 * n_layer, n * n_layer),
        )
    )
    reordering = np.int_(reordering)

    frac_cells_coarse = np.zeros(nc, dtype=bool)
    fracs_coarse = np.arange(int(n / 2 * n), int((n / 2 + 1) * n))
    frac_cells_coarse[fracs_coarse] = True

    matrix_eq_coarse = np.logical_not(frac_cells_coarse)
    for j, y in enumerate(yfactor):
        for k, d in enumerate(deg):

            appendix = "{}cells_{}degrees_{}factor".format(ny_ref, d, y)
            appendix_coarse = "{}cells_{}degrees_{}factor".format(n, d, y)
            reduced_folder = "results"
            equi_folder = main_folder
            equi_folder_coarse = main_folder
            p_eq = read_csv(equi_folder + "/pressures" + appendix + ".csv", nc_ref)
            t_eq = read_csv(equi_folder + "/tracer" + appendix + ".csv", nc_ref)

            p_eq_coarse = read_csv(
                equi_folder_coarse + "/pressures" + appendix_coarse + ".csv", nc
            )
            t_eq_coarse = read_csv(
                equi_folder_coarse + "/tracer" + appendix_coarse + ".csv", nc
            )

            p_red = read_csv(
                reduced_folder + "/pressures" + appendix_coarse + ".csv", nc
            )[reordering]
            t_red = read_csv(reduced_folder + "/tracer" + appendix_coarse + ".csv", nc)[
                reordering
            ]

            p_eq_matrix = p_eq[matrixc_ref]
            p_red_matrix = p_red[matrix_eq_coarse]
            t_eq_matrix = t_eq[matrixc_ref]
            t_red_matrix = t_red[matrix_eq_coarse]
            p_eq_matrix_coarse = p_eq_coarse[matrix_eq_coarse]
            t_eq_matrix_coarse = t_eq_coarse[matrix_eq_coarse]

            l2_pressure_map = L2(
                p_eq_matrix,
                p_red_matrix,
                cv_ref_matrix,
                value_range=1,
                mapping=small_from_big,
            )
            l2_temp_map = L2(
                t_eq_matrix,
                t_red_matrix,
                cv_ref_matrix,
                value_range=1,
                mapping=small_from_big,
            )
            l2_pressure_equi = L2(
                p_eq_matrix,
                p_eq_matrix_coarse,
                cv_ref_matrix,
                value_range=1,
                mapping=small_from_big,
            )
            l2_temp_equi = L2(
                t_eq_matrix,
                t_eq_matrix_coarse,
                cv_ref_matrix,
                value_range=1,
                mapping=small_from_big,
            )
            L2_pressures[i, j, k] = l2_pressure_map
            L2_temperatures[i, j, k] = l2_temp_map

            L2_pressures_equi[i, j, k] = l2_pressure_equi
            L2_temperatures_equi[i, j, k] = l2_temp_equi

# Visualize
ls = ["-", "--", ":"]
cl = ["b", "g", "r", "c", "m", "y"]
cl = cl[0 : len(yfactor)]
i = 0
matplotlib.rc("font", **{"size": 14})
matplotlib.rc("lines", linewidth=3)

fsz = (172 / 25.4, 129 / 25.4)
plt.figure(figsize=fsz)
plt.gca().set_prop_cycle(cycler("color", cl))
plt.semilogy(L2_pressures[:, :, i], ls=ls[i])
plt.xticks(np.arange(len(ny)), ny)
plt.yticks([1e-2, 2e-2], ["1e-2", "2e-2"])

plt.legend(yfactor)
plt.xlabel(r"1/h")
plt.ylabel(r"error")
plt.savefig("figures/pressure_logarithmic_{}_deg.png".format(deg[i]))
plt.gca().set_prop_cycle(cycler("color", cl))
plt.semilogy(L2_pressures_equi[:, :, i], ls=ls[i + 1])

plt.legend(yfactor, title=r"$K_{max}/K_{min}$")
plt.xlabel(r"$\frac{1}{h}$")
plt.ylabel(r"$E$", rotation=0)

plt.semilogy([1, 2], [np.power(1 / 2, i) * 3.5e-2 for i in [1, 2]], c="k", ls=ls[-1])
plt.xticks(np.arange(len(ny)), ny)
plt.yticks([1e-2, 5e-2], ["$10^{-2}$", "$2x10^{-2}$"])
plt.subplots_adjust(left=0.15)
plt.subplots_adjust(bottom=0.15)

plt.savefig("figures/pressure_logarithmic_equi_{}_deg.png".format(deg[i]))

plt.figure(figsize=fsz)
plt.gca().set_prop_cycle(cycler("color", cl))
red_tplot = plt.semilogy(L2_temperatures[:, :, i], ls=ls[i])

plt.xlabel(r"1/h")
plt.ylabel(r"$E$", rotation=0)

plt.gca().set_prop_cycle(cycler("color", cl))
equi_tplots = plt.semilogy(L2_temperatures_equi[:, :, i], ls=ls[i + 1])
plt.xticks(np.arange(len(ny)), ny)
plt.legend(yfactor, title=r"$K_{max}/K_{min}$")
plt.xlabel(r"$\frac{1}{h}$")
plt.ylabel(r"$E$", rotation=0)
plt.semilogy([1, 2], [np.power(1 / 2, i) * 1.2e-1 for i in [1, 2]], c="k", ls=ls[-1])
plt.subplots_adjust(left=0.18)
plt.subplots_adjust(bottom=0.15)
plt.savefig("figures/temperature_logarithmic_equi_{}_deg.png".format(deg[i]))

plt.show()
