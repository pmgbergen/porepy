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
    A_LIMITS,
    B_LIMITS,
    DPI,
    FEED,
    FIG_PATH,
    FIGURE_WIDTH,
    ISOTHERM_DATA_PATH,
    ISOTHERMS,
    NAN_ENTRY,
    NUM_COMP,
    P_LIMITS_ISOTHERMS,
    PH_FLASH_DATA_PATH,
    PHASES,
    PRESSURE_SCALE,
    PRESSURE_SCALE_NAME,
    PT_FLASH_DATA_PATH,
    PT_QUICKSHOT_DATA_PATH,
    RESOLUTION_AB,
    SPECIES,
    T_HEADER,
    THERMO_DATA_PATH,
    RESOLUTION_isotherms,
    RESOLUTION_pT,
    composition_HEADER,
    conditioning_HEADER,
    create_index_map,
    del_log,
    gas_frac_HEADER,
    h_HEADER,
    liq_frac_HEADER,
    logger,
    path,
    phases_HEADER,
    plot_abs_error_pT,
    plot_crit_point_pT,
    plot_extension_markers,
    plot_max_iter_reached,
    plot_phase_split_pT,
    plot_root_regions,
    read_px_data,
    read_results,
    success_HEADER,
    plot_widom_line,
)

# some additional plots for debugging
DEBUG: bool = True

FIG_SIZE = (FIGURE_WIDTH, 0.33 * FIGURE_WIDTH)  # 1080 / 1920


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

    nan = str(NAN_ENTRY)
    fig_num: int = 1  # for numbering the figures
    fig_path: str = f"{path()}/{FIG_PATH}"

    # read p-T data
    logger.info("Reading p-T data for thermo comparison ..")
    p_points, T_points = read_px_data(THERMO_DATA_PATH, "T")
    idx_map = create_index_map(p_points, T_points)
    res_thermo = read_results(THERMO_DATA_PATH)
    res_pp = read_results(PT_FLASH_DATA_PATH)
    res_pp_initial = read_results(PT_QUICKSHOT_DATA_PATH)
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

    num_p, num_T = p.shape
    Gibbs_energy_l = np.zeros(p.shape)
    Gibbs_energy_g = np.zeros(p.shape)
    logger.info("Calculating Gibbs energy plot data ..\n")
    for i in range(num_p):
        for j in range(num_T):
            x_ = X[i, j]
            T_ = T[i, j]
            feed = [np.ones(1) * x_, np.ones(1) * (1 - x_)]
            pv = np.ones(1) * 15e6
            Tv = np.ones(1) * T_
            prop_l = eos_l.compute(pv, Tv, feed)
            prop_g = eos_g.compute(pv, Tv, feed)
            G_l = eos_l._g_ideal(Tv, feed) + eos_l._g_dep(prop_l.A, prop_l.B, prop_l.Z)
            G_g = eos_g._g_ideal(Tv, feed) + eos_g._g_dep(prop_g.A, prop_g.B, prop_g.Z)

            Gibbs_energy_l[i, j] = G_l[0]
            Gibbs_energy_g[i, j] = G_g[0]

    split_thermo = np.zeros(p.shape)
    max_iter_reached = np.zeros(p.shape, dtype=bool)
    ll_split = np.zeros(p.shape)
    split_pp = np.zeros(p.shape)
    split_pp_initial = np.zeros(p.shape)
    cond_initial = np.zeros(p.shape)
    cond_end = np.zeros(p.shape)
    err_gas_frac = np.zeros(p.shape)
    err_gas_frac_initial = np.zeros(p.shape)
    err_enthalpy = np.zeros(p.shape)
    sc_mismatch_initial_p = np.zeros(p.shape)
    sc_mismatch_initial_T = np.zeros(p.shape)
    sc_mismatch_p = np.zeros(p.shape)
    sc_mismatch_T = np.zeros(p.shape)
    gas_frac = np.zeros(p.shape)
    err_gas_comp = np.zeros((NUM_COMP, num_p, num_T))
    err_liq_comp = np.zeros((NUM_COMP, num_p, num_T))
    unity_gap = np.zeros(p.shape)

    p_crit_water = species[0].p_crit
    T_crit_water = species[0].T_crit

    logger.info("Calculating isothermal plot data ..\n")
    for i in range(num_p):
        for j in range(num_T):
            p_ = p[i, j]
            T_ = T[i, j]
            idx = idx_map[(p_, T_)]

            # check for failure and skip if detected for both
            success_pp = int(res_pp[success_HEADER][idx])
            success_thermo = int(res_thermo[success_HEADER][idx])

            # thermo split
            if success_thermo == 0:
                split_t = res_thermo[phases_HEADER][idx]
                if "L" in split_t and "G" not in split_t:
                    split_thermo[i, j] = 1
                elif "L" in split_t and "G" in split_t:
                    split_thermo[i, j] = 2
                elif "L" not in split_t and "G" in split_t:
                    split_thermo[i, j] = 3
                if "LL" in split_t:
                    ll_split[i, j] = 1

            # porepy split from initial guess
            if phases_HEADER in res_pp_initial:
                split = res_pp_initial[phases_HEADER][idx]
                if split == "L":
                    split_pp_initial[i, j] = 1
                elif split == "GL":
                    split_pp_initial[i, j] = 2
                elif split == "G":
                    split_pp_initial[i, j] = 3
            else:
                y_pp = float(res_pp_initial[gas_frac_HEADER][idx])
                if y_pp <= 0.0:
                    split_pp_initial[i, j] = 1
                elif 0 < y_pp < 1.0:
                    split_pp_initial[i, j] = 2
                elif y_pp >= 1.0:
                    split_pp_initial[i, j] = 3

            # porepy split
            if success_pp in [0, 1, 3]:
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

                if success_pp == 1:
                    max_iter_reached[i, j] = True

            # exclude mismatching roots in supercritical region
            if p_ >= p_crit_water and T_ >= T_crit_water:
                if split_pp_initial[i, j] != split_thermo[i, j]:
                    skip_y_error_init = True
                    sc_mismatch_initial_p[i, j] = p_
                    sc_mismatch_initial_T[i, j] = T_
                if split_pp[i, j] != split_thermo[i, j]:
                    skip_y_error = True
                    sc_mismatch_p[i, j] = p_
                    sc_mismatch_T[i, j] = T_
            else:
                skip_y_error_init = False
                skip_y_error = False

            # condition number after initial guess
            cond_initial[i, j] = float(res_pp_initial[conditioning_HEADER][idx])

            # absolute error in gas fraction after initial guess
            y_thermo = float(res_thermo[gas_frac_HEADER][idx])
            y_pp_initial = float(res_pp_initial[gas_frac_HEADER][idx])
            if not skip_y_error_init:
                err_gas_frac_initial[i, j] = np.abs(y_pp_initial - y_thermo)

            # skip remainder if both failed
            if success_pp == 2 and success_thermo == 2:
                continue

            # final condition number
            cond_end[i, j] = float(res_pp[conditioning_HEADER][idx])

            # absolute error in enthalpy
            h_t = float(res_thermo[h_HEADER][idx])
            h_pp = float(res_pp[h_HEADER][idx])

            err_enthalpy[i, j] = np.abs(h_t - h_pp)

            # absolute error in gas fraction
            y_pp = float(res_pp[gas_frac_HEADER][idx])
            gas_frac[i, j] = y_pp
            if not skip_y_error:
                err_gas_frac[i, j] = np.abs(y_pp - y_thermo)
            if success_pp == 2:
                err_gas_frac[i, j] = -1

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
            else:  # unity gap
                if y_pp <= 0.:
                    phase_idx = PHASES[0]
                elif y_pp >= 1.:
                    phase_idx = PHASES[1]
                sum_ = []
                for k, s in enumerate(SPECIES):
                    sum_.append(float(res_pp[composition_HEADER[s][phase_idx]][idx]))

                unity_gap[i, j] = 1 - sum(sum_)

    sc_mismatch_initial_p = sc_mismatch_initial_p[sc_mismatch_initial_p != 0.0]
    sc_mismatch_initial_T = sc_mismatch_initial_T[sc_mismatch_initial_T != 0.0]
    sc_mismatch_p = sc_mismatch_p[sc_mismatch_p != 0.0]
    sc_mismatch_T = sc_mismatch_T[sc_mismatch_T != 0.0]

    cond_end[np.isnan(cond_end)] = 0
    cond_end[cond_end == 0] = cond_end.max()
    cond_initial[np.isnan(cond_initial)] = 0
    cond_initial[cond_initial == 0] = cond_initial.max()

    err_gas_frac[err_gas_frac == -1] = err_gas_frac.max()

    logger.info("Reading data for comparison along isotherms ..\n")
    p_points, T_points = read_px_data(ISOTHERM_DATA_PATH, "T")
    _, h_points = read_px_data(PH_FLASH_DATA_PATH, "h")
    res_pp_ph = read_results(PH_FLASH_DATA_PATH)
    res_pp_isotherms = read_results(ISOTHERM_DATA_PATH)
    idx_map_ph = create_index_map(p_points, h_points)
    idx_map_isotherms = create_index_map(p_points, T_points)

    T_vec_isotherms = np.array(ISOTHERMS)
    err_T_isotherms: list[list[float]] = [[] for _ in ISOTHERMS]
    err_y_isotherms: list[list[float]] = [[] for _ in ISOTHERMS]

    logger.info("Calculating isenthalpic plot data ..\n")
    for T_, h_ in zip(T_points, h_points):
        T_idx = ISOTHERMS.index(T_)
        T_target = T_vec_isotherms[T_idx]
        for p_ in np.unique(np.sort(np.array(p_points))):

            if (p_, h_) not in idx_map_ph:
                continue

            idx = idx_map_ph[(p_, h_)]
            success_ph = int(res_pp_ph[success_HEADER][idx])

            T_res = res_pp_ph[T_HEADER][idx]

            if T_res not in [NAN_ENTRY, str(NAN_ENTRY)]:
                err = np.abs(float(T_res) - T_target)
                if err > 3.0 and DEBUG:
                    print("investigate: phT", p_, h_, T_, f"\terr: {err}")
            else:
                err = 0.0  # np.nan

            err_T_isotherms[T_idx].append(err)

            idx_ = idx_map_isotherms[(p_, T_target)]
            y_pT = float(res_pp_isotherms[gas_frac_HEADER][idx_])
            y_ph = float(res_pp_ph[gas_frac_HEADER][idx])
            err_y_isotherms[T_idx].append(np.abs(y_pT - y_ph))

    logger.info("Calculating root data ..")
    A = np.linspace(A_LIMITS[0], A_LIMITS[1], RESOLUTION_AB)
    B = np.linspace(B_LIMITS[0], B_LIMITS[1], RESOLUTION_AB)
    A = np.hstack([A, np.array([0.0])])
    A = np.sort(np.unique(A))
    B = np.hstack([B, np.array([0.0])])
    B = np.sort(np.unique(B))

    A_mesh, B_mesh = np.meshgrid(A, B)
    n, m = A_mesh.shape
    regions = np.zeros(A_mesh.shape)
    liq_root = np.zeros(A_mesh.shape)
    gas_root = np.zeros(A_mesh.shape)
    gas_extended_widom = np.zeros(A_mesh.shape, dtype=bool)
    gas_extended = np.zeros(A_mesh.shape, dtype=bool)
    liq_extended = np.zeros(A_mesh.shape, dtype=bool)
    counter: int = 1
    nm = n * m
    eos = pp.composite.peng_robinson.PengRobinsonEoS(False)
    for i in range(n):
        for j in range(m):
            A_ = A_mesh[i, j]
            B_ = B_mesh[i, j]

            A_ij = np.array([A_])
            B_ij = np.array([B_])

            Z_L, Z_G = eos._Z(A_ij, B_ij, ensure_B_bound=False, use_widom_line=True)
            is_extended_w = int(eos.is_extended[0])

            _, _ = eos._Z(A_ij, B_ij, ensure_B_bound=False, use_widom_line=False)
            is_extended = int(eos.is_extended[0])
            Z_L = float(Z_L[0])
            Z_G = float(Z_G[0])
            for r_ in range(4):
                if eos.regions[r_][0]:
                    regions[i, j] = r_

            liq_root[i, j] = Z_L
            gas_root[i, j] = Z_G

            if regions[i, j] == 1:
                if is_extended_w == 0:
                    gas_extended_widom[i, j] = True
                elif is_extended_w == 1:
                    liq_extended[i, j] = True

                if is_extended == 0:
                    gas_extended[i, j] = True

            logger.info(f"{del_log}Calculating root data: {counter}/{nm}")
            counter += 1

    # region Gibbs Energy plot
    logger.info(f"{del_log}Plotting Gibbs Energy plot ..")
    fig, axis = plt.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d"})
    fig.set_size_inches(FIG_SIZE)
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

    # region Root plot
    logger.info(f"{del_log}Plotting roots ..")
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(1, 2)
    # fig.suptitle(f"Root cases for the Peng-Robinson EoS")
    axis = fig.add_subplot(gs[0, 0])
    axis.set_title("Number of distinct real roots")
    axis.set_xlabel("A")
    axis.set_ylabel("B")
    img = plot_root_regions(axis, A_mesh, B_mesh, regions, liq_root, gas_root)

    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb_rr = fig.colorbar(img, cax=cax, orientation="vertical")
    cb_rr.set_ticks([3 / 4 * k - 3 / 8 for k in range(1, 5)])
    cb_rr.set_ticklabels(["triple", "1 root", "2 roots", "3 roots"])

    axis = fig.add_subplot(gs[0, 1])
    axis.set_xlabel("A")
    axis.set_title("Usage of extended roots")
    axis.set(yticklabels=[])
    axis.set(ylabel=None)
    axis.tick_params(left=False)
    img = plot_extension_markers(
        axis, A_mesh, B_mesh, liq_extended, gas_extended_widom, gas_extended, liq_root, gas_root
    )

    fig.tight_layout()
    fig.savefig(
        f"{fig_path}figure_{fig_num}.png",
        format="png",
        dpi=DPI,
    )
    fig_num += 1
    # endregion

    # region Plotting phase splits
    logger.info(f"{del_log}Plotting phase split regions ..")
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(1, 3)
    # fig.suptitle(f"Phase split")

    axis = fig.add_subplot(gs[0, 0])
    axis.set_title("Unified flash: after initial guess")
    axis.set_xlabel("T [K]")
    axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
    img = plot_phase_split_pT(axis, p, T, split_pp_initial)
    plot_widom_line(axis)
    plot_crit_point_pT(axis)

    axis = fig.add_subplot(gs[0, 1])
    axis.set_title("Unified flash: after iterations")
    axis.set_xlabel("T [K]")
    axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
    img = plot_phase_split_pT(axis, p, T, split_pp)
    wid = plot_widom_line(axis)
    crit = plot_crit_point_pT(axis)
    img_ = crit[0] + wid[0]
    leg_ = crit[1] + wid[1]
    img_m, leg_m = plot_max_iter_reached(axis, p, T, max_iter_reached)
    if img_m:
        img_ += img_m
        leg_ += leg_m
    axis.legend(img_, leg_, loc="upper center")

    axis = fig.add_subplot(gs[0, 2])
    axis.set_title("thermo")
    axis.set_xlabel("T [K]")
    axis.set(yticklabels=[])
    axis.set(ylabel=None)
    axis.tick_params(left=False)
    img = plot_phase_split_pT(axis, p, T, split_thermo)
    plot_widom_line(axis)
    plot_crit_point_pT(axis)
    img_, leg_ = ([], [])
    idx = ll_split == 1  # plotting LL split in thermo plot
    if np.any(idx):
        img_ += [
            axis.plot(
                (T[idx]).flat,
                (p[idx] * PRESSURE_SCALE).flat,
                "+",
                markersize=3,
                color="black",
            )[0]
        ]
        leg_ += ["2 liquids"]
        axis.legend(img_, leg_, loc="upper center")

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

    # region Plotting condition numbers
    logger.info(f"{del_log}Plotting condition numbers ..")
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(1, 2)
    # fig.suptitle(f"Condition number of the unified p-T-flash")
    axis = fig.add_subplot(gs[0, 0])
    axis.set_title("After initial guess")
    axis.set_xlabel("T [K]")
    axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
    norm = _error_norm(cond_initial)
    img = plot_abs_error_pT(axis, p, T, cond_initial, norm=norm)
    cb = _add_colorbar(axis, img, fig, cond_initial, for_errors=False)

    axis = fig.add_subplot(gs[0, 1])
    axis.set_title("After iterations")
    axis.set_xlabel("T [K]")
    axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
    norm = _error_norm(cond_end)
    img = plot_abs_error_pT(axis, p, T, cond_end, norm=norm)
    cb = _add_colorbar(axis, img, fig, cond_end, for_errors=False)
    crit = plot_crit_point_pT(axis)
    img_, leg_ = plot_max_iter_reached(axis, p, T, max_iter_reached)
    axis.legend(crit[0] + img_, crit[1] + leg_, loc="lower right")

    fig.tight_layout()
    fig.savefig(
        f"{fig_path}figure_{fig_num}.png",
        format="png",
        dpi=DPI,
    )
    fig_num += 1
    # endregion

    # region Plotting error in enthalpy values
    if DEBUG:
        logger.info(f"{del_log}Plotting absolute errors in enthalpy ..")
        fig = plt.figure(figsize=FIG_SIZE)
        gs = fig.add_gridspec(1, 1)
        fig.suptitle(f"Abs. error in spec. enthalpy of the mixture")
        axis = fig.add_subplot(gs[0, 0])
        axis.set_xlabel("T [K]")
        axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
        norm = _error_norm(err_enthalpy)
        img = plot_abs_error_pT(axis, p, T, err_enthalpy, norm=None)
        cb = _add_colorbar(axis, img, fig, err_enthalpy)

        img_ = []
        leg_ = []
        crit = plot_crit_point_pT(axis)
        img_ += crit[0]
        leg_ += crit[1]
        axis.legend(img_, leg_, loc="upper right")

        fig.tight_layout()
        fig.savefig(
            f"{fig_path}figure_{fig_num - 1}_1.png",
            format="png",
            dpi=DPI,
        )
    # endregion

    # region Plotting absolute errors for gas fraction
    logger.info(f"{del_log}Plotting absolute errors in gas fractions ..")
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(1, 2)
    # fig.suptitle(f"Abs. error in gas fraction")
    axis = fig.add_subplot(gs[0, 0])
    axis.set_title("After initial guess")
    axis.set_xlabel("T [K]")
    axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
    norm = _error_norm(err_gas_frac_initial)
    img = plot_abs_error_pT(axis, p, T, err_gas_frac_initial, norm=None)
    cb = _add_colorbar(axis, img, fig, err_gas_frac_initial)

    img_ = []
    leg_ = []
    crit = plot_crit_point_pT(axis)
    img_ += crit[0]
    leg_ += crit[1]
    if np.any(sc_mismatch_initial_p) or np.any(sc_mismatch_initial_T):
        img_ += [
            axis.plot(
                sc_mismatch_initial_T.flat,
                (sc_mismatch_initial_p * PRESSURE_SCALE).flat,
                "x",
                color="red",
                markersize=3,
            )[0]
        ]
        leg_ += ["supercrit. root mismatch"]
    axis.legend(img_, leg_, loc="upper right")

    axis = fig.add_subplot(gs[0, 1])
    axis.set_title("After convergence")
    axis.set_xlabel("T [K]")
    axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
    norm = _error_norm(err_gas_frac)
    img = plot_abs_error_pT(axis, p, T, err_gas_frac, norm=None)
    cb = _add_colorbar(axis, img, fig, err_gas_frac)

    img_ = []
    leg_ = []
    crit = plot_crit_point_pT(axis)
    img_ += crit[0]
    leg_ += crit[1]
    if np.any(sc_mismatch_p) or np.any(sc_mismatch_T):
        img_ += [
            axis.plot(
                sc_mismatch_T.flat,
                (sc_mismatch_p * PRESSURE_SCALE).flat,
                "x",
                color="red",
                markersize=3,
            )[0]
        ]
        leg_ += ["supercrit. root mismatch"]
    axis.legend(img_, leg_, loc="upper right")

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
        fig = plt.figure(figsize=FIG_SIZE)
        gs = fig.add_gridspec(1, 1)
        fig.suptitle(f"Gas fraction values with over- and undershot values")
        axis = fig.add_subplot(gs[0, 0])
        axis.set_xlabel("T [K]")
        axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
        norm = _error_norm(err_gas_frac)
        img = axis.pcolormesh(
            T, p, gas_frac, cmap="Greys", shading="nearest", norm=_error_norm(gas_frac)
        )
        overshoot = gas_frac > 1.0
        if np.any(overshoot):
            axis.plot(
                (T[overshoot]).flat,
                (p[overshoot] * PRESSURE_SCALE).flat,
                "^",
                markersize=8,
                color="red",
            )
        undershoot = gas_frac < 0.0
        if np.any(undershoot):
            axis.plot(
                (T[undershoot]).flat,
                (p[undershoot] * PRESSURE_SCALE).flat,
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

    # region Plotting phase composition errors

    logger.info(f"{del_log}Plotting absolute errors in phase compositions ..")
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(2, 2)
    # fig.suptitle(f"Absolute errors in phase compositions")
    axis = fig.add_subplot(gs[0, 0])
    axis.set_title(f"{SPECIES[0]}")
    axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
    axis.set(xticklabels=[])
    axis.set(xlabel=None)
    axis.tick_params(bottom=False)
    errors = err_liq_comp[0, :, :]
    norm = _error_norm(errors)
    img = plot_abs_error_pT(axis, p, T, errors, norm=None)
    axis.set_ylim(0, 25.0)
    cb = _add_colorbar(axis, img, fig, errors)

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
    cb = _add_colorbar(axis, img, fig, errors)

    axis = fig.add_subplot(gs[1, 0])
    axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
    axis.set_xlabel("T [K]")
    errors = err_gas_comp[0, :, :]
    norm = _error_norm(errors)
    img = plot_abs_error_pT(axis, p, T, errors, norm=None)
    axis.set_ylim(0, 25.0)
    cb = _add_colorbar(axis, img, fig, errors)

    axis = fig.add_subplot(gs[1, 1])
    axis.set_xlabel("T [K]")
    axis.set(yticklabels=[])
    axis.set(ylabel=None)
    axis.tick_params(left=False)
    errors = err_gas_comp[1, :, :]
    norm = _error_norm(errors)
    img = plot_abs_error_pT(axis, p, T, errors, norm=None)
    axis.set_ylim(0, 25.0)
    cb = _add_colorbar(axis, img, fig, errors)
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

    # region Plotting unity gap
    logger.info(f"{del_log}Plotting unity gap ..")
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(1, 1)
    # fig.suptitle(f"Unity gaps in single-phase regions")
    axis = fig.add_subplot(gs[0, 0])
    axis.set_xlabel("T [K]")
    axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
    norm = _error_norm(unity_gap)
    img = plot_abs_error_pT(axis, p, T, unity_gap, norm=False)
    crit = plot_crit_point_pT(axis)
    axis.legend(crit[0], crit[1], loc="upper right")
    cb = _add_colorbar(axis, img, fig, unity_gap, for_errors=False)

    fig.tight_layout()
    fig.savefig(
        f"{fig_path}figure_{fig_num}.png",
        format="png",
        dpi=DPI,
    )
    fig_num += 1
    # endregion

    # region Plotting L2 error across isotherms
    err_T_l2 = [np.sqrt(np.sum(np.array(vec) ** 2)) for vec in err_T_isotherms]
    err_T_l2 = np.array(err_T_l2)
    err_y_l2 = [np.sqrt(np.sum(np.array(vec) ** 2)) for vec in err_y_isotherms]
    err_y_l2 = np.array(err_y_l2)

    # bound errors from below for plot
    cap = 1e-10
    err_T_l2[err_T_l2 <cap] = cap
    err_y_l2[err_y_l2 <cap] = cap

    logger.info(f"{del_log}Plotting L2 errors for isenthalpic flash ..")
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(1, 1)
    # fig.suptitle(f"L2-error along isotherms in temperature from p-h flash")
    axis = fig.add_subplot(gs[0, 0])
    axis.set_xlabel("T [K]")
    axis.set_yscale("log")

    img_T = axis.plot(T_vec_isotherms, err_T_l2, "-*", color="red")[0]
    img_y = axis.plot(T_vec_isotherms, err_y_l2, "-*", color="black")[0]

    axis.legend([img_T, img_y], ["L2-err in T", "L2-err in y"], loc="upper left")

    fig.tight_layout()
    fig.savefig(
        f"{fig_path}figure_{fig_num}.png",
        format="png",
        dpi=DPI,
    )
    fig_num += 1
    # endregion

    # region Plotting absolute error per isotherm
    logger.info(f"{del_log}Plotting abs error per isotherm ..")
    p_ = (
        np.linspace(
            P_LIMITS_ISOTHERMS[0],
            P_LIMITS_ISOTHERMS[1],
            RESOLUTION_isotherms,
            endpoint=True,
            dtype=float,
        )
        * PRESSURE_SCALE
    )

    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(2, 3)
    # fig.suptitle(f"Abs. error per isotherm in temperature resulting from p-h-flash")

    nT = len(ISOTHERMS)
    nrow = 2
    if nT % 2 == 0:
        ncol = [int(nT / 2)] * 2
    else:
        ncol = [int(np.ceil(nT / 2)), int(np.floor(nT / 2))]

    n = 0
    err_cap = 1e-10
    for r in range(nrow):
        for c in range(ncol[r]):

            axis = fig.add_subplot(gs[r, c])
            axis.set_xlabel(f"p [{PRESSURE_SCALE_NAME}]")
            axis.set_title(f"T = {T_vec_isotherms[n]}")
            axis.set_yscale("log")
            # caping errors from below for plot
            err_T_abs = np.array(err_T_isotherms[n])
            err_y_abs = np.array(err_y_isotherms[n])
            err_T_abs[err_T_abs < cap] = cap
            err_y_abs[err_y_abs < cap] = cap
            img_T = axis.plot(p_, err_T_abs, "-*", color="red")[0]
            img_y = axis.plot(p_, err_y_abs, "-*", color="black")[0]
            n += 1

            # if r == 0 and c == (ncol[0] - 1):
            #     axis.legend([img_T, img_y], ["Abs. err. in T", "Abs. err. in y"], loc="upper right")

    fig.tight_layout()
    fig.savefig(
        f"{fig_path}figure_{fig_num}.png",
        format="png",
        dpi=DPI,
    )
    # endregion
