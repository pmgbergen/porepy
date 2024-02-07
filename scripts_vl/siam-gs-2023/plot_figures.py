"""Script for creating all figures.

This scripts assumes that data has been generated with a previous call to
``calculate_data.ph``.

Figures are stored in ``figs/``, numbered as in the publication.

"""
from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import rcParams

# from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import porepy as pp

sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

from _config import (
    DATA_PATH,
    DPI,
    FIG_PATH,
    FIGURE_WIDTH,
    PHASES,
    PRESSURE_SCALE,
    PRESSURE_SCALE_NAME,
    SALINITIES,
    SPECIES,
    T_HEADER,
    composition_HEADER,
    create_index_map,
    del_log,
    gas_frac_HEADER,
    logger,
    p_HEADER,
    path,
    phases_HEADER,
    plot_crit_point_pT,
    plot_phase_split_pT,
    read_data_column,
    read_results,
    sal_path,
    success_HEADER,
)

# some additional plots for debugging
DEBUG: bool = True

# bounding errors from below for plotting purpose
ERROR_CAP = 1e-10

PLOT_ROOTS: bool = False

FIG_SIZE = (FIGURE_WIDTH, 0.33 * FIGURE_WIDTH)  # 1080 / 1920

font_size = 20
plt.rc("font", size=font_size)  # controls default text size
plt.rc("axes", titlesize=font_size)  # fontsize of the title
plt.rc("axes", labelsize=font_size)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=font_size)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=font_size)  # fontsize of the y tick labels
plt.rc("legend", fontsize=13)  # fontsize of the legend


def _fmt(x, pos):
    a, b = "{:.1e}".format(x).split("e")
    b = int(b)
    return r"${}e{{{}}}$".format(a, b)


def _add_colorbar(axis_, img_, fig_, vals_, for_errors=True):
    divider = make_axes_locatable(axis_)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig_.colorbar(
        img_,
        cax=cax,
        orientation="vertical",
        format=ticker.FuncFormatter(_fmt),
        # format=ticker.LogFormatterMathtext(),
    )
    if for_errors:
        cb.set_label(
            "Max. abs. error: "
            + "{:.0e}".format(float(vals_.max()))
            + "\nL2-error: "
            + "{:.0e}".format(float(np.sqrt(np.sum(np.square(vals_)))))
        )
    return cb


def _error_norm(err_):
    # return mpl.colors.LogNorm(vmin=err_.min(), vmax=err_.max())
    return mpl.colors.SymLogNorm(
        linthresh=1e-3, linscale=0.5, vmin=err_.min(), vmax=err_.max()
    )


if __name__ == "__main__":

    fig_num: int = 1  # for numbering the figures
    fig_path: str = f"{path()}/{FIG_PATH}"

    # read p-T data
    logger.info("Reading p-T ..")
    paths = [sal_path(DATA_PATH, s) for s in SALINITIES]
    p_points = read_data_column(paths[0], p_HEADER)
    T_points = read_data_column(paths[1], T_HEADER)
    idx_map = create_index_map(p_points, T_points)

    results = [read_results(p) for p in paths]

    # create p-T mesh
    p_vec = np.unique(np.sort(np.array(p_points)))
    T_vec = np.unique(np.sort(np.array(T_points)))
    T, p = np.meshgrid(T_vec, p_vec)
    num_p, num_T = p.shape

    split_pp = [np.zeros(p.shape) for _ in SALINITIES]
    gas_frac = [np.zeros(p.shape) for _ in SALINITIES]
    co2_frac = [np.zeros(p.shape) for _ in SALINITIES]

    logger.info("Calculating plot data ..\n")
    for i in range(num_p):
        for j in range(num_T):
            p_ = p[i, j]
            T_ = T[i, j]
            idx = idx_map[(p_, T_)]

            for sidx, s in enumerate(SALINITIES):

                # check for failure and skip if detected for both
                success = int(results[sidx][success_HEADER][idx])
                # if success in [0, 1, 3]:
                # porepy split from initial guess
                if phases_HEADER in results[sidx]:
                    split = results[sidx][phases_HEADER][idx]
                    if split == "L":
                        split_pp[sidx][i, j] = 1
                    elif split == "GL":
                        split_pp[sidx][i, j] = 2
                    elif split == "G":
                        split_pp[sidx][i, j] = 3
                else:
                    y_pp = float(results[sidx][gas_frac_HEADER][idx])
                    if y_pp <= 0.0:
                        split_pp[sidx][i, j] = 1
                    elif 0 < y_pp < 1.0:
                        split_pp[sidx][i, j] = 2
                    elif y_pp >= 1.0:
                        split_pp[sidx][i, j] = 3

                gas_frac[sidx][i, j] = float(results[sidx][gas_frac_HEADER][idx])
                if gas_frac[sidx][i, j] > 0:
                    x = float(
                        results[sidx][composition_HEADER[SPECIES[1]][PHASES[0]]][idx]
                    )
                    co2_frac[sidx][i, j] = x

    for sidx, s in enumerate(SALINITIES):
        gas_frac[sidx][np.isnan(gas_frac[sidx])] = 0.0
        gas_frac[sidx][gas_frac[sidx] < 0.0] = 0.0
        co2_frac[sidx][np.isnan(co2_frac[sidx])] = 0.0

    # region Plotting phase splits
    logger.info(f"{del_log}Plotting phase split regions ..")
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(1, len(SALINITIES))

    for sidx, s in enumerate(SALINITIES):
        axis = fig.add_subplot(gs[0, sidx])
        sal = str(s)
        sal = sal[sal.rfind(".") + 1 :]
        axis.set_title(f"salinity: {s}")
        axis.set_xlabel("T [K]")
        if sidx == 0:
            axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
        else:
            axis.set(yticklabels=[])
            axis.set(ylabel=None)
            axis.tick_params(left=False)
        img = plot_phase_split_pT(axis, p, T, split_pp[sidx])
        # plot_widom_line(axis)
        crit = plot_crit_point_pT(axis)

    # last plot contains legend
    axis.legend(crit[0], crit[1], loc="upper right", markerscale=2)

    fig.subplots_adjust(right=0.8)
    divider = make_axes_locatable(axis)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.3)
    cb = fig.colorbar(img, cax=cbar_ax, orientation="vertical")
    cb.set_ticks([3 / 4 * k - 3 / 8 for k in range(1, 5)])
    cb.set_ticklabels(["N/A", "L", "GL", "G"])

    fig.tight_layout()
    fig.savefig(
        f"{fig_path}salinity_comparison.png",
        format="png",
        dpi=DPI,
    )
    fig_num += 1
    # endregion

    # region Plotting fractions
    logger.info(f"{del_log}Plotting phase split regions ..")
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(1, 2)

    axis = fig.add_subplot(gs[0, 0])
    axis.set_title("Gas fraction")
    axis.set_xlabel("T [K]")
    axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
    norm = _error_norm(gas_frac[-1])
    img = axis.pcolormesh(
        T,
        p * PRESSURE_SCALE,
        gas_frac[-1],
        cmap="Greys",
        shading="nearest",
        norm=norm,
    )

    fig.subplots_adjust(right=0.8)
    divider = make_axes_locatable(axis)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.3)
    cb = fig.colorbar(
        img,
        cax=cbar_ax,
        orientation="vertical",
        format=ticker.FuncFormatter(_fmt),
    )

    axis = fig.add_subplot(gs[0, 1])
    axis.set_title("CO2 fraction in gas")
    axis.set_xlabel("T [K]")
    axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
    norm = _error_norm(co2_frac[-1])
    img = axis.pcolormesh(
        T, p * PRESSURE_SCALE, co2_frac[-1], cmap="Greys", shading="nearest", norm=norm
    )

    fig.subplots_adjust(right=0.8)
    divider = make_axes_locatable(axis)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.3)
    cb = fig.colorbar(
        img,
        cax=cbar_ax,
        orientation="vertical",
        format=ticker.FuncFormatter(_fmt),
    )

    fig.tight_layout()
    fig.savefig(
        f"{fig_path}salinity_fractions.png",
        format="png",
        dpi=DPI,
    )
    fig_num += 1
    # endregion
