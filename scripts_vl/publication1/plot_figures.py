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
    FEED,
    FIG_PATH,
    FIGURE_WIDTH,
    ISOTHERM_DATA_PATH,
    ISOTHERMS,
    NAN_ENTRY,
    NUM_COMP,
    PH_FLASH_DATA_PATH,
    PHASES,
    PT_FLASH_DATA_PATH,
    SPECIES,
    T_HEADER,
    THERMO_DATA_PATH,
    RESOLUTION_pT,
    composition_HEADER,
    create_index_map,
    del_log,
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

# some additional plots for debugging
DEBUG: bool = False


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


if __name__ == "__main__":

    nan = str(NAN_ENTRY)
    fig_num: int = 1  # for numbering the figures
    fig_path: str = f"{path()}/{FIG_PATH}"

    # read p-T data
    logger.info("Reading p-T data for thermo comparison ..")
    p_points, T_points = read_px_data(THERMO_DATA_PATH, "T")
    idx_map = create_index_map(p_points, T_points)
    res_thermo = read_results(THERMO_DATA_PATH)
    res_pp = read_results(PT_FLASH_DATA_PATH)
    # create p-T mesh
    p_vec = np.unique(np.sort(np.array(p_points)))
    T_vec = np.unique(np.sort(np.array(T_points)))
    T, p = np.meshgrid(T_vec, p_vec)
    x_vec = np.linspace(0 + 1e-5, 1 - 1e-5, RESOLUTION_pT, dtype=float)
    _, X = np.meshgrid(T_vec, x_vec)

    eos_l = pp.composite.peng_robinson.PengRobinsonEoS(False)
    eos_g = pp.composite.peng_robinson.PengRobinsonEoS(True)
    species = pp.composite.load_fluid_species(SPECIES)
    comps = [
        pp.composite.peng_robinson.H2O.from_species(species[0]),
        pp.composite.peng_robinson.CO2.from_species(species[1]),
    ]
    eos_l.components = comps
    eos_g.components = comps
    feed = [np.ones(1) * z for z in FEED]

    # calculating values to be plotted
    logger.info("Calculating plot data ..\n")
    num_p, num_T = p.shape
    Gibbs_energy_l = np.zeros(p.shape)
    Gibbs_energy_g = np.zeros(p.shape)
    split_thermo = np.zeros(p.shape)
    ll_split = np.zeros(p.shape)
    split_pp = np.zeros(p.shape)
    err_gas_frac = np.zeros(p.shape)
    gas_frac = np.zeros(p.shape)
    err_gas_comp = np.zeros((NUM_COMP, num_p, num_T))
    err_liq_comp = np.zeros((NUM_COMP, num_p, num_T))

    for i in range(num_p):
        for j in range(num_T):
            x_ = X[i, j]
            T_ = T[i, j]
            feed = [np.ones(1) * x_, np.ones(1) * (1 - x_)]
            pv = np.ones(1) * 15
            Tv = np.ones(1) * T_
            prop_l = eos_l.compute(pv, Tv, feed)
            prop_g = eos_g.compute(pv, Tv, feed)
            G_l = eos_l._g_ideal(Tv, feed) + eos_l._g_dep(prop_l.A, prop_l.B, prop_l.Z)
            G_g = eos_g._g_ideal(Tv, feed) + eos_g._g_dep(prop_g.A, prop_g.B, prop_g.Z)

            Gibbs_energy_l[i, j] = G_l[0]
            Gibbs_energy_g[i, j] = G_g[0]

    for i in range(num_p):
        for j in range(num_T):
            p_ = p[i, j]
            T_ = T[i, j]
            idx = idx_map[(p_, T_)]

            # Gibbs energy
            # pv = np.ones(1) * p_ / pp.composite.PRESSURE_SCALE
            # Tv = np.ones(1) * T_
            # prop_l = eos_l.compute(pv, Tv, feed)
            # prop_g = eos_g.compute(pv, Tv, feed)
            # G_l = eos_l._g_ideal(Tv, feed) + eos_l._g_dep(prop_l.A, prop_l.B, prop_l.Z)
            # G_g = eos_g._g_ideal(Tv, feed) + eos_g._g_dep(prop_g.A, prop_g.B, prop_g.Z)

            # Gibbs_energy_l[i, j] = G_l[0]
            # Gibbs_energy_g[i, j] = G_g[0]

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
                # if phase split is not available, use gas fraction
                if phases_HEADER in res_pp:
                    split = res_pp[phases_HEADER][idx]
                    if split == "L":
                        split_pp[i, j] = 1
                    elif split == "GL":
                        split_pp[i, j] = 2
                    elif split == "G":
                        split_pp[i, j] = 3
                else:
                    y_pp = float(res_pp[gas_frac_HEADER][idx])
                    if y_pp <= 0.0:
                        split_pp[i, j] = 1
                    elif 0 < y_pp < 1.0:
                        split_pp[i, j] = 2
                    elif y_pp >= 1.0:
                        split_pp[i, j] = 3

            # skip remainder if both failed
            if not (success_pp and success_thermo):
                continue

            # absolute error in gas fraction
            y_pp = float(res_pp[gas_frac_HEADER][idx])
            if p_ > 20e6 and (450 < T_ < 650) and y_pp > 0:
                print("investigate", p_, T_)
            gas_frac[i, j] = y_pp
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

    logger.info("Reading data for comparison along isotherms ..")
    p_points, T_points = read_px_data(ISOTHERM_DATA_PATH, "T")
    _, h_points = read_px_data(PH_FLASH_DATA_PATH, "h")
    res_pp_ph = read_results(PH_FLASH_DATA_PATH)
    idx_map_ph = create_index_map(p_points, h_points)

    T_vec_isotherms = np.array(ISOTHERMS)
    err_T_isotherms: list[list[float]] = [[] for _ in ISOTHERMS]

    logger.info("Calculating plot data ..\n")
    for T_, h_ in zip(T_points, h_points):
        T_idx = ISOTHERMS.index(T_)
        T_target = T_vec_isotherms[T_idx]
        for p_ in np.unique(np.sort(np.array(p_points))):

            if (p_, h_) not in idx_map_ph:
                continue

            idx = idx_map_ph[(p_, h_)]
            success_ph = int(res_pp_ph[success_HEADER][idx])

            if success_ph:
                T_res = float(res_pp_ph[T_HEADER][idx])
                err = np.abs(T_res - T_target)
                if err > 1:
                    print("investigate: phT", p_, h_, T_)
            else:
                err = 0.0  # np.nan

            err_T_isotherms[T_idx].append(err)

    # region Gibbs Energy plot
    logger.info(f"{del_log}Plotting Gibbs Energy plot ..")
    fig, axis = plt.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d"})
    fig.set_size_inches(FIGURE_WIDTH, 1080 / 1920 * FIGURE_WIDTH)
    fig.suptitle(f"Gibbs energy")
    axis[0].set_title("Liquid phase")
    axis[0].set_xlabel("T [K]")
    axis[0].set_ylabel("z_1 [-]")
    axis[0].set_zlabel("g [kJ]")
    img = axis[0].plot_surface(
        T,
        X,
        Gibbs_energy_l,
        # T, p / pp.composite.PRESSURE_SCALE, Gibbs_energy_l,
        cmap="coolwarm",
        linewidth=0,
        antialiased=False,
    )

    axis[1].set_title("Gas phase")
    axis[1].set_xlabel("T [K]")
    axis[1].set_ylabel("z_1 [-]")
    axis[1].set_zlabel("g [kJ]")
    img = axis[1].plot_surface(
        T,
        X,
        Gibbs_energy_g,
        # T, p / pp.composite.PRESSURE_SCALE, Gibbs_energy_g,
        cmap="coolwarm",
        linewidth=0,
        antialiased=False,
    )

    fig.tight_layout()
    fig.savefig(
        f"{fig_path}figure_{fig_num}.png",
        format="png",
        dpi=DPI,
    )
    fig_num += 1
    # endregion

    # region plotting first figure: phase splits
    logger.info(f"{del_log}Plotting phase split regions ..")
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
    axis.set(yticklabels=[])
    axis.set(ylabel=None)
    axis.tick_params(left=False)
    img = plot_phase_split_pT(axis, p, T, split_thermo)
    img_, leg_ = ([], [])
    idx = ll_split == 1  # plotting LL split in thermo plot
    if np.any(idx):
        img_ += [
            axis.plot(
                (T[idx]).flat,
                (p[idx] / pp.composite.PRESSURE_SCALE).flat,
                "+",
                markersize=3,
                color="black",
            )[0]
        ]
        leg_ += ["2 liquids"]
    crit = plot_crit_point_pT(axis)
    img_ += crit[0]
    leg_ += crit[1]
    axis.legend(img_, leg_, loc="upper right")

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
    logger.info(f"{del_log}Plotting absolute errors in gas fractions ..")
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

    # region Gas fraction values
    if DEBUG:
        logger.info(f"{del_log}Plotting gas fraction values ..")
        fig = plt.figure(figsize=(FIGURE_WIDTH, 1080 / 1920 * FIGURE_WIDTH))
        gs = fig.add_gridspec(1, 1)
        fig.suptitle(f"Gas fraction values with over- and undershot values")
        axis = fig.add_subplot(gs[0, 0])
        axis.set_xlabel("T [K]")
        axis.set_ylabel("p [MPa]")
        norm = _error_norm(err_gas_frac)
        img = axis.pcolormesh(
            T, p, gas_frac, cmap="Greys", shading="nearest", norm=_error_norm(gas_frac)
        )
        overshoot = gas_frac > 1.0
        if np.any(overshoot):
            axis.plot(
                (T[overshoot]).flat,
                (p[overshoot] / pp.composite.PRESSURE_SCALE).flat,
                "^",
                markersize=8,
                color="red",
            )
        undershoot = gas_frac < 0.0
        if np.any(undershoot):
            axis.plot(
                (T[undershoot]).flat,
                (p[undershoot] / pp.composite.PRESSURE_SCALE).flat,
                "v",
                markersize=6,
                color="orange",
            )
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig.colorbar(
            img,
            cax=cax,
            orientation="vertical",
            format=ticker.FuncFormatter(_fmt),
            # format=ticker.LogFormatterMathtext(),
        )
        cb.set_label(f"Max.: {gas_frac.max()}\nMin.: {gas_frac.min()}")

        fig.tight_layout()
        fig.savefig(
            f"{fig_path}figure_{fig_num - 1}_1.png",
            format="png",
            dpi=DPI,
        )
    # endregion

    # region plotting phase composition errors

    logger.info(f"{del_log}Plotting absolute errors in phase compositions ..")
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
    axis.set_ylim(0, 25.0)
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
    axis.set_ylim(0, 25.0)
    cb = _error_cb(axis, img, fig, errors)

    axis = fig.add_subplot(gs[1, 0])
    axis.set_ylabel("p [MPa]")
    axis.set_xlabel("T [K]")
    errors = err_gas_comp[0, :, :]
    norm = _error_norm(errors)
    img = plot_abs_error_pT(axis, p, T, errors, norm=None)
    axis.set_ylim(0, 25.0)
    cb = _error_cb(axis, img, fig, errors)

    axis = fig.add_subplot(gs[1, 1])
    axis.set_xlabel("T [K]")
    axis.set(yticklabels=[])
    axis.set(ylabel=None)
    axis.tick_params(left=False)
    errors = err_gas_comp[1, :, :]
    norm = _error_norm(errors)
    img = plot_abs_error_pT(axis, p, T, errors, norm=None)
    axis.set_ylim(0, 25.0)
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
    fig_num += 1
    # endregion

    # region Plotting L2 error for isotherms
    err_T_l2 = [np.sqrt(np.sum(np.array(vec) ** 2)) for vec in err_T_isotherms]
    err_T_l2 = np.array(err_T_l2)

    logger.info(f"{del_log}Plotting L2 error for temperature ..")
    fig = plt.figure(figsize=(FIGURE_WIDTH, 1080 / 1920 * FIGURE_WIDTH))
    gs = fig.add_gridspec(1, 1)
    fig.suptitle(f"L2-error along isotherms in temperature from p-h flash")
    axis = fig.add_subplot(gs[0, 0])
    axis.set_xlabel("T [K]")
    # axis.set_ylabel("L-2 error")
    axis.set_yscale("log")

    axis.plot(T_vec_isotherms, err_T_l2, "-*", color="black")

    fig.tight_layout()
    fig.savefig(
        f"{fig_path}figure_{fig_num}.png",
        format="png",
        dpi=DPI,
    )
    fig_num += 1
    # endregion
