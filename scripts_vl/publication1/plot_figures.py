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
    DPI,
    FIG_PATH,
    FIGURE_WIDTH,
    NAN_ENTRY,
    NUM_COMP,
    PHASES,
    PT_FLASH_DATA_PATH,
    SPECIES,
    THERMO_DATA_PATH,
    composition_HEADER,
    create_index_map,
    gas_frac_HEADER,
    liq_frac_HEADER,
    logger,
    path,
    phases_HEADER,
    plot_abs_error_pT,
    plot_crit_point_pT,
    plot_phase_split_pT,
    read_px_data,
    read_results,
    success_HEADER,
)

if __name__ == "__main__":

    nan = str(NAN_ENTRY)
    fig_num: int = 1  # for numbering the figures
    fig_path: str = f"{path()}/{FIG_PATH}"

    # read p-T data
    logger.info("Reading p-T data for thermo comparison ..\n")
    p_points, T_points = read_px_data(THERMO_DATA_PATH, "T")
    idx_map = create_index_map(p_points, T_points)
    res_thermo = read_results(THERMO_DATA_PATH)
    res_pp = read_results(PT_FLASH_DATA_PATH)
    # create p-T mesh
    p_vec = np.unique(np.sort(np.array(p_points)))
    T_vec = np.unique(np.sort(np.array(T_points)))
    T, p = np.meshgrid(T_vec, p_vec)

    # calculating values to be plotted
    logger.info("Calculating plot data ..\n")
    num_p, num_T = p.shape
    split_thermo = np.zeros(p.shape)
    ll_split = np.zeros(p.shape)
    split_pp = np.zeros(p.shape)
    err_gas_frac = np.zeros(p.shape)
    err_gas_comp = np.zeros((NUM_COMP, num_p, num_T))
    err_liq_comp = np.zeros((NUM_COMP, num_p, num_T))

    for i in range(num_p):
        for j in range(num_T):
            p_ = p[i, j]
            T_ = T[i, j]
            idx = idx_map[(p_, T_)]
            # check for failure and skip if detected for both
            success_pp = int(res_pp[success_HEADER][idx])
            success_thermo = int(res_thermo[success_HEADER][idx])

            # thermo split
            if success_thermo:
                split_t = res_thermo[phases_HEADER][idx]
                if "L" in split_t and "G" not in split_t:
                    split_thermo[i, j] = 1
                elif "L" in split_t and "G" in split_t:
                    split_thermo[i, j] = 2
                elif "L" not in split_t and "G" in split_t:
                    split_thermo[i, j] = 3
                if "LL" in split_t:
                    ll_split[i, j] = 1

            # porepy split
            if success_pp:
                split = res_pp[phases_HEADER][idx]
                if split == "L":
                    split_pp[i, j] = 1
                elif split == "GL":
                    split_pp[i, j] = 2
                elif split == "G":
                    split_pp[i, j] = 3

            # skip remainder if both failed
            if not (success_pp and success_thermo):
                continue

            # absolute error in gas fraction
            y_pp = float(res_pp[gas_frac_HEADER][idx])
            y_thermo = float(res_thermo[gas_frac_HEADER][idx])
            err_gas_frac[i, j] = np.abs(y_pp - y_thermo)

            # absolute error in phase compositions, where there is a 2-phase regime
            if 0.0 < y_pp < 1 and 0.0 < y_thermo < 1.0:
                for k, s in enumerate(SPECIES):
                    # gas phase error
                    x_Gk_pp = float(res_pp[composition_HEADER[s][PHASES[0]]][idx])
                    x_Gk_thermo = float(
                        res_thermo[composition_HEADER[s][PHASES[0]]][idx]
                    )
                    err_gas_comp[k, i, j] = np.abs(x_Gk_pp - x_Gk_thermo)

                    # liquid phase error
                    x_Lk_pp = float(res_pp[composition_HEADER[s][PHASES[1]]][idx])
                    # if thermo indicates 2 liquid phases, cluster the fractions
                    if ll_split[i, j]:
                        yL1 = float(res_thermo[liq_frac_HEADER[0]][idx])
                        yL2 = float(res_thermo[liq_frac_HEADER[1]][idx])
                        x_L1k_thermo = float(
                            res_thermo[composition_HEADER[s][PHASES[1]]][idx]
                        )
                        x_L2k_thermo = float(
                            res_thermo[composition_HEADER[s][PHASES[2]]][idx]
                        )
                        x_Lk_thermo = yL1 * x_L1k_thermo + yL2 * x_L2k_thermo
                    else:
                        x_Lk_thermo = float(
                            res_thermo[composition_HEADER[s][PHASES[1]]][idx]
                        )

                    err_liq_comp[k, i, j] = np.abs(x_Lk_pp - x_Lk_thermo)

    # region plotting first figure: phase splits
    logger.info("Plotting phase split regions ..\n")
    fig = plt.figure(figsize=(FIGURE_WIDTH, 1080 / 1920 * FIGURE_WIDTH))
    gs = fig.add_gridspec(1, 2)
    fig.suptitle(f"Phase split")
    axis = fig.add_subplot(gs[0, 0])
    axis.set_title("Unified flash")
    axis.set_xlabel("T [K]")
    axis.set_ylabel("p [MPa]")
    img = plot_phase_split_pT(axis, p, T, split_pp)
    plot_crit_point_pT(axis)

    axis = fig.add_subplot(gs[0, 1])
    axis.set_title("thermo")
    axis.set_xlabel("T [K]")
    img = plot_phase_split_pT(axis, p, T, split_thermo)
    img_, leg_ = plot_crit_point_pT(axis)
    idx = ll_split == 1  # plotting LL split in thermo plot
    if np.any(idx):
        img_ += [
            axis.plot(
                (T[idx]).flat,
                (p[idx] / pp.composite.PRESSURE_SCALE).flat,
                "+",
                markersize=5,
                color="black",
            )[0]
        ]
        leg_ += ["2 liquids"]
    axis.legend(img_, leg_, loc="upper right")
    # remove p ticks from second plot
    axis.set(yticklabels=[])
    axis.set(ylabel=None)
    axis.tick_params(left=False)

    fig.subplots_adjust(right=0.8)
    divider = make_axes_locatable(axis)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.3)
    cb = fig.colorbar(img, cax=cbar_ax, orientation="vertical")
    cb.set_ticks([3 / 4 * k - 3 / 8 for k in range(1, 5)])
    cb.set_ticklabels(["N/A", "L", "GL", "G"])

    fig.tight_layout()
    fig.savefig(
        f"{fig_path}figure_{fig_num}.png",
        format="png",
        dpi=DPI,
    )
    fig_num += 1
    # endregion

    # region plotting absolute errors for gas fraction

    def _fmt(x, pos):
        a, b = "{:.1e}".format(x).split("e")
        b = int(b)
        return r"${}e{{{}}}$".format(a, b)

    def _error_cb(axis_, img_, fig_, err_):
        divider = make_axes_locatable(axis_)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig_.colorbar(
            img_,
            cax=cax,
            orientation="vertical",
            format=ticker.FuncFormatter(_fmt),
            # format=ticker.LogFormatterMathtext(),
        )
        cb.set_label(
            "Max. abs. error: "
            + "{:.0e}".format(float(err_.max()))
            + "\nL2-error: "
            + "{:.0e}".format(float(np.sqrt(np.sum(np.square(err_)))))
        )
        return cb

    def _error_norm(err_):
        # return mpl.colors.LogNorm(vmin=err_.min(), vmax=err_.max())
        return mpl.colors.SymLogNorm(
            linthresh=1e-3, linscale=0.5, vmin=err_.min(), vmax=err_.max()
        )

    logger.info("Plotting absolute errors in gas fractions ..\n")
    fig = plt.figure(figsize=(FIGURE_WIDTH, 1080 / 1920 * FIGURE_WIDTH))
    gs = fig.add_gridspec(1, 1)
    fig.suptitle(f"Abs. error in gas fraction")
    axis = fig.add_subplot(gs[0, 0])
    axis.set_xlabel("T [K]")
    axis.set_ylabel("p [MPa]")
    norm = _error_norm(err_gas_frac)
    img = plot_abs_error_pT(axis, p, T, err_gas_frac, norm=None)
    cb = _error_cb(axis, img, fig, err_gas_frac)

    fig.tight_layout()
    fig.savefig(
        f"{fig_path}figure_{fig_num}.png",
        format="png",
        dpi=DPI,
    )
    fig_num += 1
    # endregion

    # region plotting phase composition errors

    logger.info("Plotting absolute errors in phase compositions ..\n")
    fig = plt.figure(figsize=(FIGURE_WIDTH, 1080 / 1920 * FIGURE_WIDTH))
    gs = fig.add_gridspec(2, 2)
    fig.suptitle(f"Absolute errors in phase compositions")
    axis = fig.add_subplot(gs[0, 0])
    axis.set_title(f"{SPECIES[0]}")
    axis.set_ylabel("p [MPa]")
    axis.set(xticklabels=[])
    axis.set(xlabel=None)
    axis.tick_params(bottom=False)
    errors = err_liq_comp[0, :, :]
    norm = _error_norm(errors)
    img = plot_abs_error_pT(axis, p, T, errors, norm=None)
    cb = _error_cb(axis, img, fig, errors)

    axis = fig.add_subplot(gs[0, 1])
    axis.set_title(f"{SPECIES[1]}")
    axis.set(xticklabels=[])
    axis.set(xlabel=None)
    axis.tick_params(bottom=False)
    axis.set(yticklabels=[])
    axis.set(ylabel=None)
    axis.tick_params(left=False)
    errors = err_liq_comp[1, :, :]
    norm = _error_norm(errors)
    img = plot_abs_error_pT(axis, p, T, errors, norm=None)
    cb = _error_cb(axis, img, fig, errors)

    axis = fig.add_subplot(gs[1, 0])
    axis.set_ylabel("p [MPa]")
    axis.set_xlabel("T [K]")
    errors = err_gas_comp[0, :, :]
    norm = _error_norm(errors)
    img = plot_abs_error_pT(axis, p, T, errors, norm=None)
    cb = _error_cb(axis, img, fig, errors)

    axis = fig.add_subplot(gs[1, 1])
    axis.set_xlabel("T [K]")
    axis.set(yticklabels=[])
    axis.set(ylabel=None)
    axis.tick_params(left=False)
    errors = err_gas_comp[1, :, :]
    norm = _error_norm(errors)
    img = plot_abs_error_pT(axis, p, T, errors, norm=None)
    cb = _error_cb(axis, img, fig, errors)
    fig.subplots_adjust(left=0.1)
    fig.tight_layout()

    fig.text(0.02, 0.7, "Liquid\nphase", fontsize=rcParams["axes.titlesize"])
    fig.text(0.02, 0.26, "Gas\nphase", fontsize=rcParams["axes.titlesize"])
    fig.subplots_adjust(left=0.1)
    fig.savefig(
        f"{fig_path}figure_{fig_num}.png",
        format="png",
        dpi=DPI,
    )
    fig.show()
    fig_num += 1
    # endregion
