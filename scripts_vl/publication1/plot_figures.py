"""Script for creating all figures.

This scripts assumes that data has been generated with a previous call to
``calculate_data.ph``.

Figures are stored in ``figs/``, numbered as in the publication.

"""
from __future__ import annotations

import os
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import rcParams

import porepy as pp

sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

from _config import (
    A_LIMITS,
    ASPECT_RATIO,
    B_LIMITS,
    DPI,
    FIG_PATH,
    FIGURE_FORMAT,
    FIGURE_WIDTH,
    GEO_DATA_PATH,
    GEO_THERMO_DATA_PATH,
    HV_FLASH_DATA_PATH,
    HV_ISOBAR,
    HV_ISOBAR_DATA_PATH,
    HV_ISOTHERM,
    HV_ISOTHERM_DATA_PATH,
    ISOTHERM_DATA_PATH,
    ISOTHERMS,
    MARKER_SCALE,
    MARKER_SIZE,
    NAN_ENTRY,
    NUM_COMP,
    P_LIMITS_ISOTHERMS,
    PH_FLASH_DATA_PATH,
    PHASES,
    PRESSURE_SCALE,
    PRESSURE_SCALE_NAME,
    PT_FLASH_DATA_PATH,
    RESOLUTION_AB,
    SPECIES,
    T_HEADER,
    THERMO_DATA_PATH,
    X_SCALE,
    X_SCALE_NAME,
    EXAMPLE_2_flash_type,
    RESOLUTION_ph,
    composition_HEADER,
    conditioning_HEADER,
    create_index_map,
    del_log,
    gas_frac_HEADER,
    gas_satur_HEADER,
    h_HEADER,
    liq_frac_HEADER,
    logger,
    num_iter_HEADER,
    p_HEADER,
    path,
    phases_HEADER,
    plot_abs_error_pT,
    plot_conjugate_x_for_px_flash,
    plot_crit_point_H2O,
    plot_hv_iso,
    plot_max_iter_reached,
    plot_phase_split_GL,
    plot_root_extensions,
    plot_root_regions,
    plot_Widom_points_experimental,
    read_data_column,
    read_results,
    success_HEADER,
    v_HEADER,
)

# Max iter number, for visualization of respective plot
MAX_ITER: int = 150

# bounding errors from below for plotting purpose
ERROR_CAP = 1e-10

# Skip calculation of root data for A-B plot for performance
PLOT_ROOTS: bool = False
# Plots for water-CO2 mixture
PLOT_FIRST_EXAMPLE: bool = True
# plots for multicomponent mixture
PLOT_SECOND_EXAMPLE: bool = False

# Padding from figure borders
FIG_PAD: float = 0.05

# prints some additional information for debugging
DEBUG: bool = True

font_size = 20
plt.rc("font", size=font_size)  # controls default text size
plt.rc("axes", titlesize=font_size)  # fontsize of the title
plt.rc("axes", labelsize=font_size)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=font_size)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=font_size)  # fontsize of the y tick labels
plt.rc("legend", fontsize=17)  # fontsize of the legend


def _fmt(x, pos):
    """Colorbar ticks formatter"""
    a, b = "{:.1e}".format(x).split("e")
    b = int(b)
    return r"${}e{{{}}}$".format(a, b)


if __name__ == "__main__":

    nan = str(NAN_ENTRY)
    fig_num: int = 1  # for numbering the figures
    fig_path: str = f"{path()}/{FIG_PATH}"

    if not os.path.isdir(fig_path):
        logger.info("Creating figure directory ..\n")
        os.mkdir(fig_path)

    # read p-T data
    logger.info("Reading p-T data for thermo comparison ..")
    p_points = read_data_column(THERMO_DATA_PATH, p_HEADER)
    T_points = read_data_column(THERMO_DATA_PATH, T_HEADER)
    idx_map = create_index_map(p_points, T_points)
    res_thermo = read_results(THERMO_DATA_PATH)
    res_pp = read_results(PT_FLASH_DATA_PATH)
    # create p-T mesh
    p_vec = np.unique(np.sort(np.array(p_points)))
    T_vec = np.unique(np.sort(np.array(T_points)))
    T, p = np.meshgrid(T_vec, p_vec)
    num_p, num_T = p.shape

    species = pp.composite.load_species(["ethane", "heptane"])

    # region Calculating values to be plotted
    split_thermo = np.zeros(p.shape)
    max_iter_reached = np.zeros(p.shape, dtype=bool)
    num_iter = np.zeros(p.shape)
    ll_split = np.zeros(p.shape)
    split_pp = np.zeros(p.shape)
    cond_end = np.zeros(p.shape)
    err_gas_frac = np.zeros(p.shape)
    err_enthalpy = np.zeros(p.shape)
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

                num_iter[i, j] = int(res_pp[num_iter_HEADER][idx])

            # exclude mismatching roots in supercritical region
            skip_y_error = False
            if p_ >= p_crit_water and T_ >= T_crit_water:
                if split_pp[i, j] != split_thermo[i, j]:
                    skip_y_error = True
                    sc_mismatch_p[i, j] = p_
                    sc_mismatch_T[i, j] = T_

            # skip remainder if both failed
            if success_pp == 2 and success_thermo == 2:
                continue

            # final condition number
            # cond_end[i, j] = float(res_pp[conditioning_HEADER][idx])
            cond_end[i, j] = 1.

            # absolute discrepancy in enthalpy,
            # this is significant due to different models
            h_t = float(res_thermo[h_HEADER][idx])
            h_pp = float(res_pp[h_HEADER][idx])

            err_enthalpy[i, j] = np.abs(h_t - h_pp)

            # absolute error in gas fraction
            y_thermo = float(res_thermo[gas_frac_HEADER][idx])
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
                if y_pp <= 0.0:
                    phase_idx = PHASES[0]
                elif y_pp >= 1.0:
                    phase_idx = PHASES[1]
                sum_ = []
                for k, s in enumerate(SPECIES):
                    sum_.append(float(res_pp[composition_HEADER[s][phase_idx]][idx]))

                unity_gap[i, j] = 1 - sum(sum_)

    # get only points where there is a mismatch
    sc_mismatch_p = sc_mismatch_p[sc_mismatch_p != 0.0]
    sc_mismatch_T = sc_mismatch_T[sc_mismatch_T != 0.0]

    # removing the max iter for plotting purpose.
    # points where max iter reached are plotted with markers, not coloring.
    num_iter[num_iter >= MAX_ITER] = 0

    # This happens, or is severely ill-conditioned. Set to max for plotting purpose.
    cond_end[np.isnan(cond_end)] = 0
    cond_end[cond_end == 0] = cond_end.max()

    # if a flash failed (-1), set the error there to max error, for plotting purpose
    # failures are plotted by coloring in a separate plot
    err_gas_frac[err_gas_frac == -1] = err_gas_frac.max()

    logger.info("Reading data for comparison along isotherms ..\n")
    p_points = read_data_column(ISOTHERM_DATA_PATH, p_HEADER)
    T_points = read_data_column(ISOTHERM_DATA_PATH, T_HEADER)
    h_points = read_data_column(ISOTHERM_DATA_PATH, h_HEADER)
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
                if err > 1.0 and DEBUG:
                    print("investigate: phT", p_, h_, T_, f"\terr: {err}")
            else:
                err = np.nan  # np.nan

            err_T_isotherms[T_idx].append(err)

            idx_ = idx_map_isotherms[(p_, T_target)]
            y_pT = float(res_pp_isotherms[gas_frac_HEADER][idx_])
            y_ph = float(res_pp_ph[gas_frac_HEADER][idx])
            err_y_isotherms[T_idx].append(np.abs(y_pT - y_ph))

    err_T_isotherms = [np.array(_) for _ in err_T_isotherms]
    err_y_isotherms = [np.array(_) for _ in err_y_isotherms]

    logger.info("Calculating h-v flash plot data ..\n")
    results_hv = read_results(HV_FLASH_DATA_PATH)
    results_hv_ip = read_results(HV_ISOBAR_DATA_PATH)
    results_hv_iT = read_results(HV_ISOTHERM_DATA_PATH)

    idx_map_hv = create_index_map(
        np.array([float(_) for _ in results_hv[h_HEADER]]),
        np.array([float(_) for _ in results_hv[v_HEADER]]),
    )
    idx_map_hv_ip = create_index_map(
        np.array([float(_) for _ in results_hv_ip[p_HEADER]]),
        np.array([float(_) for _ in results_hv_ip[T_HEADER]]),
    )
    idx_map_hv_iT = create_index_map(
        np.array([float(_) for _ in results_hv_iT[p_HEADER]]),
        np.array([float(_) for _ in results_hv_iT[T_HEADER]]),
    )

    err_hv_T_ip = []
    err_hv_p_ip = []
    err_hv_s_ip = []
    err_hv_y_ip = []

    vec_ = np.array([float(_) for _ in results_hv_ip[p_HEADER]])
    p_ip = np.sort(np.unique(vec_))
    assert len(p_ip) == 1 and p_ip[0] == HV_ISOBAR, "Unknown isobar for HV"
    vec_ = np.array([float(_) for _ in results_hv_ip[T_HEADER]])
    T_ip = np.sort(np.unique(vec_))
    for T_ in T_ip:
        # get results from corresponding p-T flash
        idx = idx_map_hv_ip[(HV_ISOBAR, T_)]
        y_pT = float(results_hv_ip[gas_frac_HEADER][idx])
        s_pT = float(results_hv_ip[gas_satur_HEADER][idx])

        # get results from h-v flash
        h_pT = float(results_hv_ip[h_HEADER][idx])
        v_pT = float(results_hv_ip[v_HEADER][idx])

        idx = idx_map_hv[(h_pT, v_pT)]
        success_ = int(results_hv[success_HEADER][idx])

        if success_ != 2:
            p_hv = float(results_hv[p_HEADER][idx])
            T_hv = float(results_hv[T_HEADER][idx])
            y_hv = float(results_hv[gas_frac_HEADER][idx])
            s_hv = float(results_hv[gas_satur_HEADER][idx])

            err_hv_p_ip.append(np.abs(HV_ISOBAR - p_hv) * PRESSURE_SCALE)
            err_hv_T_ip.append(np.abs(T_ - T_hv))
            err_hv_s_ip.append(np.abs(s_pT - s_hv))
            err_hv_y_ip.append(np.abs(y_pT - y_hv))

            if DEBUG and (err_hv_p_ip[-1] > 1 or err_hv_T_ip[-1] > 1):
                print(
                    "investigate: pThv",
                    HV_ISOBAR,
                    T_,
                    h_pT,
                    v_pT,
                    f"\terr p: {err_hv_p_ip[-1]} \t err T: {err_hv_T_ip[-1]}",
                )
        else:
            err_hv_p_ip.append(0.0)
            err_hv_T_ip.append(0.0)
            err_hv_s_ip.append(0.0)
            err_hv_y_ip.append(0.0)

    err_hv_p_ip = np.array(err_hv_p_ip)
    err_hv_T_ip = np.array(err_hv_T_ip)
    err_hv_s_ip = np.array(err_hv_s_ip)
    err_hv_y_ip = np.array(err_hv_y_ip)
    # Capping numerical zero errors from below for plotting purpose.
    err_hv_s_ip[err_hv_s_ip < ERROR_CAP] = ERROR_CAP
    err_hv_y_ip[err_hv_y_ip < ERROR_CAP] = ERROR_CAP

    err_hv_T_iT = []
    err_hv_p_iT = []
    err_hv_s_iT = []
    err_hv_y_iT = []

    vec_ = np.array([float(_) for _ in results_hv_iT[T_HEADER]])
    T_iT = np.sort(np.unique(vec_))
    assert len(T_iT) == 1 and T_iT[0] == HV_ISOTHERM, "Unknown isotherm for HV"
    vec_ = np.array([float(_) for _ in results_hv_iT[p_HEADER]])
    p_iT = np.sort(np.unique(vec_))
    for p_ in p_iT:
        # get results from corresponding p-T flash
        idx = idx_map_hv_iT[(p_, HV_ISOTHERM)]
        y_pT = float(results_hv_iT[gas_frac_HEADER][idx])
        s_pT = float(results_hv_iT[gas_satur_HEADER][idx])

        # get results from h-v flash
        h_pT = float(results_hv_iT[h_HEADER][idx])
        v_pT = float(results_hv_iT[v_HEADER][idx])

        idx = idx_map_hv[(h_pT, v_pT)]
        success_ = int(results_hv[success_HEADER][idx])

        if success_ != 2:
            p_hv = float(results_hv[p_HEADER][idx])
            T_hv = float(results_hv[T_HEADER][idx])
            y_hv = float(results_hv[gas_frac_HEADER][idx])
            s_hv = float(results_hv[gas_satur_HEADER][idx])

            err_hv_p_iT.append(np.abs(p_ - p_hv) * PRESSURE_SCALE)
            err_hv_T_iT.append(np.abs(HV_ISOTHERM - T_hv))
            err_hv_s_iT.append(np.abs(s_pT - s_hv))
            err_hv_y_iT.append(np.abs(y_pT - y_hv))

            if DEBUG and (err_hv_p_iT[-1] > 1 or err_hv_T_iT[-1] > 1):
                print(
                    "investigate: pThv",
                    HV_ISOBAR,
                    T_,
                    h_pT,
                    v_pT,
                    f"\terr p: {err_hv_p_iT[-1]} \t err T: {err_hv_T_iT[-1]}",
                )
        else:
            err_hv_p_iT.append(0.0)
            err_hv_T_iT.append(0.0)
            err_hv_s_iT.append(0.0)
            err_hv_y_iT.append(0.0)

    err_hv_p_iT = np.array(err_hv_p_iT)
    err_hv_T_iT = np.array(err_hv_T_iT)
    err_hv_s_iT = np.array(err_hv_s_iT)
    err_hv_y_iT = np.array(err_hv_y_iT)
    # Capping numerical zero errors from below for plotting purpose.
    err_hv_y_iT[err_hv_y_iT < ERROR_CAP] = ERROR_CAP
    err_hv_s_iT[err_hv_s_iT < ERROR_CAP] = ERROR_CAP

    logger.info("Calculating geothermal example plot data ..\n")
    results_geo = read_results(GEO_DATA_PATH)
    results_geo_thermo = read_results(GEO_THERMO_DATA_PATH)

    if EXAMPLE_2_flash_type == "p-h":
        x_header = h_HEADER
        not_x_header = T_HEADER
    elif EXAMPLE_2_flash_type == "p-T":
        x_header = T_HEADER
        not_x_header = h_HEADER

    p_points_geo = np.array([float(_) for _ in results_geo[p_HEADER]])
    x_points_geo = np.array([float(_) for _ in results_geo[x_header]])
    idx_map_geo = create_index_map(
        p_points_geo,
        x_points_geo,
    )
    idx_map_geo_thermo = create_index_map(
        np.array([float(_) for _ in results_geo_thermo[p_HEADER]]),
        np.array([float(_) for _ in results_geo_thermo[x_header]]),
    )

    p_vec_geo = np.unique(np.sort(np.array(p_points_geo)))
    x_vec_geo = np.unique(np.sort(np.array(x_points_geo)))

    x_geo, p_geo = np.meshgrid(x_vec_geo, p_vec_geo)
    num_p_geo, num_x_geo = p_geo.shape

    split_geo = np.zeros(p_geo.shape)
    cx_result_geo = np.zeros(p_geo.shape)
    cx_error_geo = np.zeros(p_geo.shape)
    max_iter_reached_geo = np.zeros(p_geo.shape, dtype=bool)
    num_iter_geo = np.zeros(p_geo.shape)
    doubt_geo = np.zeros(p_geo.shape, dtype=bool)
    y_error_geo = np.zeros(p_geo.shape)

    for i in range(num_p_geo):
        for j in range(num_x_geo):
            p_ = p_geo[i, j]
            h_ = x_geo[i, j]
            idx = idx_map_geo[(p_, h_)]
            idx_thermo = idx_map_geo_thermo[(p_, h_)]

            # check for failure and skip if detected for both
            success_geo = int(results_geo[success_HEADER][idx])

            # porepy split
            if success_geo in [0, 1, 3]:
                y_pp = float(results_geo[gas_frac_HEADER][idx])
                y_th = float(results_geo_thermo[gas_frac_HEADER][idx_thermo])
                T_th = float(results_geo_thermo[T_HEADER][idx_thermo])
                # if phase split is not available, use gas fraction

                # if y_th <= 0.0:
                #     split_geo[i, j] = 1
                # elif 0 < y_th < 1.0:
                #     split_geo[i, j] = 2
                #     # print(f"investigate: p = {p_} h = {h_}")
                # elif y_th >= 1.0:
                #     split_geo[i, j] = 3

                if phases_HEADER in results_geo:
                    split = results_geo[phases_HEADER][idx]
                    if split == "L":
                        split_geo[i, j] = 1
                    elif split == "GL":
                        split_geo[i, j] = 2
                    elif split == "G":
                        split_geo[i, j] = 3
                    elif "LL" in split and "G" not in split:
                        split_geo[i, j] = 4
                    elif "GLL" in split:
                        split_geo[i, j] = 5
                else:
                    if y_pp <= 0.0:
                        split_geo[i, j] = 1
                    elif 0 < y_pp < 1.0:
                        split_geo[i, j] = 2
                        # print(f"investigate: p = {p_} h = {h_}")
                    elif y_pp >= 1.0:
                        split_geo[i, j] = 3

                if success_geo == 1:
                    max_iter_reached_geo[i, j] = True
                    # doubt_geo[i, j] = True

                y_error_geo[i, j] = np.abs(y_pp - y_th)

                cx_result_geo[i, j] = float(results_geo[not_x_header][idx])
                cx_error_geo[i, j] = np.abs(cx_result_geo[i, j] - T_th)

                num_iter_geo[i, j] = int(results_geo[num_iter_HEADER][idx])
            else:
                doubt_geo[i, j] = True

    # removing the max iter for plotting purpose.
    # points where max iter reached are plotted with markers, not coloring.
    num_iter_geo[num_iter_geo >= MAX_ITER] = 0

    # endregion

    if PLOT_ROOTS:
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
        # indicater which root is extended
        # 1 - liquid extended, 2 - gas extended with Widom, 3 - gas extended with Gharbia
        root_extensions = np.zeros(A_mesh.shape)
        counter: int = 1
        nm = n * m
        eos = pp.composite.peng_robinson.PengRobinsonEoS(False)
        for i in range(n):
            for j in range(m):
                A_ = A_mesh[i, j]
                B_ = B_mesh[i, j]

                A_ij = np.array([A_])
                B_ij = np.array([B_])

                Z_L, Z_G = eos._Z(
                    A_ij, B_ij, asymmetric_extension=False, use_widom_line=True
                )
                is_extended_w = int(eos.is_extended[0])

                _, _ = eos._Z(
                    A_ij, B_ij, asymmetric_extension=False, use_widom_line=False
                )
                is_extended = int(eos.is_extended[0])
                Z_L = float(Z_L[0])
                Z_G = float(Z_G[0])
                for r_ in range(4):
                    if eos.regions[r_][0]:
                        regions[i, j] = r_
                    # filtering out triple and double, since they are not visible due to
                    # refinement
                if eos.regions[0][0] or eos.regions[2][0]:
                    regions[i, j] = 3

                liq_root[i, j] = Z_L
                gas_root[i, j] = Z_G

                # liquid extension in super-crit area
                if regions[i, j] == 3:
                    b_c = eos.critical_line(A_)
                    if B_ >= b_c:
                        root_extensions[i, j] = 1
                # plotting extension in one-root region
                if regions[i, j] == 1:
                    # liquid extended by default
                    root_extensions[i, j] = 1
                    # if Gas extended using Widom line
                    if is_extended_w == 0:
                        root_extensions[i, j] = 2
                    # if Gas extended with Gharbia extension
                    if is_extended == 0:
                        root_extensions[i, j] = 3

                logger.info(f"{del_log}Calculating root data: {counter}/{nm}")
                counter += 1

        a_ticks = np.around(np.linspace(A_mesh.min(), A_mesh.max(), 6), decimals=1)
        b_ticks = np.around(np.linspace(B_mesh.min(), B_mesh.max(), 7)[1:], decimals=2)
        b_ticks = np.hstack([b_ticks, np.array([0.0])])
        logger.info(f"{del_log}Plotting roots ..")
        fig = plt.figure(figsize=(2 * FIGURE_WIDTH, ASPECT_RATIO * FIGURE_WIDTH))
        axis = fig.add_subplot(1, 2, 1)
        axis.set_box_aspect(1)
        axis.set_xlabel("A")
        axis.set_ylabel("B")
        axis.set_xticks(a_ticks)
        axis.set_yticks(b_ticks)
        img = plot_root_regions(axis, A_mesh, B_mesh, regions, liq_root)

        cax = axis.inset_axes([1.02, 0.4, 0.05, 0.2])
        cb_rr = fig.colorbar(img, ax=axis, cax=cax, orientation="vertical")
        cb_rr.set_ticks([3 / 4, 3 / 2 + 3 / 4])
        cb_rr.set_ticklabels(["1\nroot", "3\nroots"])

        axis = fig.add_subplot(1, 2, 2)
        axis.set_box_aspect(1)
        axis.set_xlabel("A")
        axis.set_ylabel("B")
        # axis.set(yticklabels=[])
        # axis.set(ylabel=None)
        # axis.tick_params(left=False)
        axis.set_xticks(a_ticks)
        img = plot_root_extensions(
            axis,
            A_mesh,
            B_mesh,
            root_extensions,
        )

        cax = axis.inset_axes([1.02, 0.2, 0.05, 0.6])
        cb_rr = fig.colorbar(img, ax=axis, cax=cax, orientation="vertical")
        cb_rr.set_ticks([3 / 4 * k - 3 / 8 for k in range(1, 5)])
        cb_rr.set_ticklabels(
            [
                "not\nextended",
                "liquid\nextended",
                "gas\nextended\n(Widom)",
                "gas\nextended",
            ]
        )
        axis.text(
            0.1,
            0.08,
            "supercritical\nliquid extension\nEquation (4.9)",
            fontsize=rcParams["axes.titlesize"],
        )
        axis.text(
            0.64,
            0.105,
            "supercritical\ngas extension\nEquation (4.8)",
            fontsize=rcParams["axes.titlesize"],
        )
        axis.text(
            0.45,
            0.03,
            "subcritical\nextensions\nEquation (4.5)",
            fontsize=rcParams["axes.titlesize"],
        )
        axis.arrow(
            0.42, 0.04, -0.07, 0.015, linewidth=0.5, head_width=0.005, color="black"
        )
        axis.arrow(
            0.42, 0.04, -0.05, -0.02, linewidth=0.5, head_width=0.005, color="black"
        )

        fig.tight_layout(pad=FIG_PAD)
        fig.savefig(
            f"{fig_path}figure_{fig_num}.{FIGURE_FORMAT}",
            format=FIGURE_FORMAT,
            dpi=DPI,
        )

    fig_num += 1  # to preserve figure numbers even if not plotted

    if PLOT_FIRST_EXAMPLE:

        # region Plotting phase splits
        logger.info(f"{del_log}Plotting phase split regions ..")
        fig = plt.figure(figsize=(FIGURE_WIDTH, ASPECT_RATIO * 2 * FIGURE_WIDTH))
        axis = fig.add_subplot(2, 1, 1)
        axis.set_box_aspect(1)
        axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
        # axis.set_xlabel("T [K]")
        axis.set(xticklabels=[])
        axis.set(xlabel=None)
        axis.tick_params(bottom=False)
        img = plot_phase_split_GL(axis, p, T, split_pp)
        wid = plot_Widom_points_experimental(axis)
        crit = plot_crit_point_H2O(axis)
        img_ = crit[0] + wid[0]
        leg_ = crit[1] + wid[1]
        axis.legend(img_, leg_, loc="upper left", markerscale=MARKER_SCALE)

        axis = fig.add_subplot(2, 1, 2)
        axis.set_box_aspect(1)
        axis.set_xlabel("T [K]")
        axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
        img = plot_phase_split_GL(axis, p, T, split_thermo)
        plot_Widom_points_experimental(axis)
        plot_crit_point_H2O(axis)
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
            axis.legend(img_, leg_, loc="upper left", markerscale=MARKER_SCALE)

        fig.tight_layout(pad=FIG_PAD, w_pad=0.1)
        fig.subplots_adjust(right=0.95)
        cax = fig.add_axes([0.87, 0.33, 0.05, 0.33])
        cb = fig.colorbar(img, cax=cax, orientation="vertical")
        cb.set_ticks([3 / 4 * k - 3 / 8 for k in range(1, 5)])
        cb.set_ticklabels(["N/A", "L", "GL", "G"])

        fig.savefig(
            f"{fig_path}figure_{fig_num}.{FIGURE_FORMAT}",
            format=FIGURE_FORMAT,
            dpi=DPI,
        )
        fig_num += 1
        # endregion

        # region Plotting iteration numbers
        logger.info(f"{del_log}Plotting iteration numbers ..")
        fig = plt.figure(figsize=(FIGURE_WIDTH, ASPECT_RATIO * FIGURE_WIDTH))
        axis = fig.add_subplot(1, 1, 1)
        axis.set_box_aspect(1)
        axis.set_xlabel("T [K]")
        axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")

        img = plot_abs_error_pT(axis, p, T, num_iter, norm=None)
        crit = plot_crit_point_H2O(axis)
        img_, leg_ = plot_max_iter_reached(axis, p, T, max_iter_reached)
        axis.legend(
            crit[0] + img_, crit[1] + leg_, loc="upper left", markerscale=MARKER_SCALE
        )

        cax = axis.inset_axes([1.04, 0.2, 0.05, 0.6])
        cb = fig.colorbar(
            img,
            ax=axis,
            cax=cax,
            orientation="vertical",  # format=ticker.FuncFormatter(_fmt),
        )
        cbt = cb.get_ticks()
        cbt = np.sort(np.hstack([cbt, np.array([num_iter.max()])]))
        cbt = cbt[cbt <= num_iter.max()]
        cb.set_ticks(cbt.astype(int))

        fig.tight_layout(pad=FIG_PAD)
        fig.savefig(
            f"{fig_path}figure_{fig_num}.{FIGURE_FORMAT}",
            format=FIGURE_FORMAT,
            dpi=DPI,
        )
        fig_num += 1
        # endregion

        print(f"\nExample 1: averate num iter {np.mean(num_iter)}")
        print(
            f"Example 1: num of max iter reached: {max_iter_reached.sum()} / {num_p * num_T}"
        )

        # region Plotting condition numbers
        logger.info(f"{del_log}Plotting condition numbers ..")
        fig = plt.figure(figsize=(FIGURE_WIDTH, ASPECT_RATIO * FIGURE_WIDTH))
        axis = fig.add_subplot(1, 1, 1)
        axis.set_box_aspect(1)
        axis.set_xlabel("T [K]")
        axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
        norm = mpl.colors.LogNorm(vmin=cond_end.min(), vmax=cond_end.max(), clip=False)
        img = plot_abs_error_pT(axis, p, T, cond_end, norm=norm)
        crit = plot_crit_point_H2O(axis)
        axis.legend(crit[0], crit[1], loc="upper left", markerscale=MARKER_SCALE)

        cax = axis.inset_axes([1.04, 0.2, 0.05, 0.6])
        cb = fig.colorbar(
            img,
            ax=axis,
            cax=cax,
            orientation="vertical",
            format=ticker.FuncFormatter(_fmt),
        )
        cbt = cb.get_ticks()
        cbt = cbt[(cbt <= cond_end.max()) & (cbt >= cond_end.min())]
        cbt = np.sort(cbt)[1:]
        cbt = np.sort(np.hstack([cbt, np.array([cond_end.min(), cond_end.max()])]))
        cb.set_ticks(cbt)

        fig.tight_layout(pad=FIG_PAD)
        fig.savefig(
            f"{fig_path}figure_{fig_num}.{FIGURE_FORMAT}",
            format=FIGURE_FORMAT,
            dpi=DPI,
        )
        fig_num += 1
        # endregion

        # region Plotting absolute errors for gas fraction
        logger.info(f"{del_log}Plotting absolute errors in gas fractions ..")
        fig = plt.figure(figsize=(FIGURE_WIDTH, ASPECT_RATIO * FIGURE_WIDTH))
        axis = fig.add_subplot(1, 1, 1)
        axis.set_box_aspect(1)
        axis.set_xlabel("T [K]")
        axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
        vmax = err_gas_frac.max()
        norm = mpl.colors.LogNorm(vmin=ERROR_CAP, vmax=vmax, clip=True)
        img = plot_abs_error_pT(axis, p, T, err_gas_frac, norm=norm)

        img_ = []
        leg_ = []
        crit = plot_crit_point_H2O(axis)
        img_ += crit[0]
        leg_ += crit[1]
        if np.any(sc_mismatch_p) or np.any(sc_mismatch_T):
            img_ += [
                axis.plot(
                    sc_mismatch_T.flat,
                    (sc_mismatch_p * PRESSURE_SCALE).flat,
                    "2",
                    color="red",
                    markersize=MARKER_SIZE,
                )[0]
            ]
            leg_ += ["supercrit.\nroot mismatch"]
        axis.legend(img_, leg_, loc="upper left", markerscale=MARKER_SCALE)

        cax = axis.inset_axes([1.04, 0.2, 0.05, 0.6])
        cb = fig.colorbar(
            img,
            ax=axis,
            cax=cax,
            orientation="vertical",
            format=ticker.FuncFormatter(_fmt),
        )
        cbt = cb.get_ticks()
        cbt = cbt[cbt < vmax]
        cbt = np.sort(np.hstack([cbt, np.array([vmax])]))
        cb.set_ticks(cbt)
        # cbt = cb.get_ticks()
        # cbt = cbt[cbt < err_gas_frac.max()]
        # cbt = np.sort(np.hstack([cbt, np.array([err_gas_frac.max()])]))
        # cb.set_ticks(cbt)

        fig.tight_layout(pad=FIG_PAD)
        fig.savefig(
            f"{fig_path}figure_{fig_num}.{FIGURE_FORMAT}",
            format=FIGURE_FORMAT,
            dpi=DPI,
        )
        fig_num += 1
        # endregion

        # region Plotting phase composition errors

        logger.info(f"{del_log}Plotting absolute errors in phase compositions ..")
        fig = plt.figure(figsize=(FIGURE_WIDTH, 0.5 * FIGURE_WIDTH))

        nf = 1
        vmax = [e.max() for e in err_liq_comp] + [e.max() for e in err_gas_comp]
        vmax = np.max(vmax)
        norm = mpl.colors.LogNorm(vmin=ERROR_CAP, vmax=vmax, clip=True)
        p_min = p.min() * PRESSURE_SCALE
        for r in [0, 1]:
            for c in [0, 1]:
                axis = fig.add_subplot(2, 2, nf)
                nf += 1
                axis.set_box_aspect(0.5)

                if r == 0:
                    err_c = err_liq_comp[c, :, :]
                    axis.set_title(f"{SPECIES[c]}")
                    if c == 0:
                        axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
                        axis.set(xticklabels=[])
                        axis.set(xlabel=None)
                        axis.tick_params(bottom=False)
                    elif c == 1:
                        axis.set(xticklabels=[])
                        axis.set(xlabel=None)
                        axis.tick_params(bottom=False)
                        axis.set(yticklabels=[])
                        axis.set(ylabel=None)
                        axis.tick_params(left=False)
                elif r == 1:
                    err_c = err_gas_comp[c, :, :]
                    if c == 0:
                        axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
                        axis.set_xlabel("T [K]")
                    elif c == 1:
                        axis.set_xlabel("T [K]")
                        axis.set(yticklabels=[])
                        axis.set(ylabel=None)
                        axis.tick_params(left=False)

                # capping errors
                err_c[err_c < ERROR_CAP] = ERROR_CAP

                img = axis.pcolormesh(
                    T,
                    p * PRESSURE_SCALE,
                    err_c,
                    cmap="Greys",
                    shading="nearest",
                    # vmin=0., vmax=vmax,
                    norm=norm,
                )
                axis.set_ylim(p_min, 25.0)

        fig.tight_layout(pad=FIG_PAD)
        fig.subplots_adjust(right=0.75)
        cax = fig.add_axes([0.8, 0.15, 0.05, 0.7])
        cb = fig.colorbar(
            img,
            cax=cax,
            orientation="vertical",
            format=ticker.FuncFormatter(_fmt),
        )
        cbt = cb.get_ticks()
        cbt = cbt[cbt < vmax]
        cbt = np.sort(np.hstack([cbt, np.array([vmax])]))
        cb.set_ticks(cbt)
        fig.subplots_adjust(left=0.2)
        fig.text(0.0, 0.66, "Liquid\nphase", fontsize=rcParams["axes.titlesize"])
        fig.text(0.0, 0.33, "Gas\nphase", fontsize=rcParams["axes.titlesize"])

        fig.savefig(
            f"{fig_path}figure_{fig_num}.{FIGURE_FORMAT}",
            format=FIGURE_FORMAT,
            dpi=DPI,
        )
        fig_num += 1
        # endregion

        # region Plotting unity gap
        logger.info(f"{del_log}Plotting unity gap ..")
        fig = plt.figure(figsize=(FIGURE_WIDTH, ASPECT_RATIO * FIGURE_WIDTH))
        axis = fig.add_subplot(1, 1, 1)
        axis.set_box_aspect(1)
        axis.set_xlabel("T [K]")
        axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
        img = plot_abs_error_pT(axis, p, T, unity_gap)
        crit = plot_crit_point_H2O(axis)
        axis.legend(crit[0], crit[1], loc="upper left", markerscale=MARKER_SCALE)

        cax = axis.inset_axes([1.04, 0.2, 0.05, 0.6])
        cb = fig.colorbar(
            img,
            ax=axis,
            cax=cax,
            orientation="vertical",
            format=ticker.FuncFormatter(_fmt),
        )
        cbt = cb.get_ticks()
        cbt = cbt[cbt < unity_gap.max()]
        cbt = np.sort(np.hstack([cbt, np.array([unity_gap.max()])]))
        cb.set_ticks(cbt)

        fig.tight_layout(pad=FIG_PAD)
        fig.savefig(
            f"{fig_path}figure_{fig_num}.{FIGURE_FORMAT}",
            format=FIGURE_FORMAT,
            dpi=DPI,
        )
        fig_num += 1
        # endregion

        # region Plotting L2 error across isotherms
        err_T_l2 = [
            np.sqrt(np.sum(vec[np.logical_not(np.isnan(vec))] ** 2)) / len(vec)
            for vec in err_T_isotherms
        ]
        err_T_l2 = np.array(err_T_l2)
        err_y_l2 = [
            np.sqrt(np.sum(vec[np.logical_not(np.isnan(vec))] ** 2)) / len(vec)
            for vec in err_y_isotherms
        ]
        err_y_l2 = np.array(err_y_l2)

        # bound errors from below for plot
        err_T_l2[err_T_l2 < ERROR_CAP] = ERROR_CAP
        err_y_l2[err_y_l2 < ERROR_CAP] = ERROR_CAP

        logger.info(f"{del_log}Plotting L2 errors for isenthalpic flash ..")
        fig = plt.figure(figsize=(FIGURE_WIDTH, ASPECT_RATIO * 0.5 * FIGURE_WIDTH))
        axis = fig.add_subplot(1, 1, 1)
        axis.set_box_aspect(0.5)
        axis.set_xlabel("T [K]")
        axis.set_yscale("log")

        img_T = axis.plot(
            T_vec_isotherms, err_T_l2, "-s", color="red", markersize=MARKER_SIZE
        )[0]
        img_y = axis.plot(
            T_vec_isotherms, err_y_l2, "-D", color="black", markersize=MARKER_SIZE
        )[0]

        axis.legend(
            [img_T, img_y],
            ["L2-err in T", "L2-err in y"],
            loc="upper left",
            markerscale=MARKER_SCALE,
        )

        fig.tight_layout(pad=FIG_PAD)
        fig.savefig(
            f"{fig_path}figure_{fig_num}.{FIGURE_FORMAT}",
            format=FIGURE_FORMAT,
            dpi=DPI,
        )
        fig_num += 1
        # endregion

        err_ = np.array(err_T_isotherms)
        print(
            f"\nExample 1: p-h flash max error in T: {(err_[np.logical_not(np.isnan(err_))]).max()}"
        )
        err_ = np.array(err_y_isotherms)
        print(
            f"Example 1: p-h flash max error in y: {(err_[np.logical_not(np.isnan(err_))]).max()}"
        )

        # region Plotting absolute error per isotherm
        logger.info(f"{del_log}Plotting abs error per isotherm ..")
        p_ = (
            np.linspace(
                P_LIMITS_ISOTHERMS[0],
                P_LIMITS_ISOTHERMS[1],
                RESOLUTION_ph,
                endpoint=True,
                dtype=float,
            )
            * PRESSURE_SCALE
        )

        nT = len(ISOTHERMS)
        nrow = 0
        ncol = [0, 0]
        if nT % 2 == 0:
            nrow = int(nT / 2)
            ncol = [2] * nrow
        else:
            nrow = int(np.ceil(nT / 2))
            ncol = [2] * (nrow - 1) + [1]

        fig = plt.figure(
            figsize=(FIGURE_WIDTH, (0.33 * nrow) * ASPECT_RATIO * FIGURE_WIDTH)
        )
        gs = fig.add_gridspec(nrow, 2)

        n = 0
        marker_size = int(np.floor(MARKER_SIZE / 2))
        for r in range(nrow):
            for c in range(ncol[r]):

                axis = fig.add_subplot(gs[r, c])
                axis.set_box_aspect(0.5)
                if r == nrow - 1:
                    axis.set_xlabel(f"p [{PRESSURE_SCALE_NAME}]")
                else:
                    axis.set(xticklabels=[])
                    axis.set(xlabel=None)
                    axis.tick_params(bottom=False)

                axis.set_title(f"T = {T_vec_isotherms[n]} [K]")

                # caping errors from below for plot
                err_T_abs = err_T_isotherms[n]
                err_y_abs = err_y_isotherms[n]
                err_T_abs[err_T_abs < ERROR_CAP] = ERROR_CAP
                err_y_abs[err_y_abs < ERROR_CAP] = ERROR_CAP
                img_T = axis.plot(
                    p_, err_T_abs, "-s", color="red", markersize=marker_size
                )[0]
                img_y = axis.plot(
                    p_, err_y_abs, "-D", color="black", markersize=marker_size
                )[0]
                axis.set_yscale("log")

                if c == 0:
                    yticks = [1e1, 1e-1, 1e-3, 1e-6, ERROR_CAP]
                    axis.set_yticks(yticks)
                else:
                    axis.set(yticklabels=[])
                    axis.set(ylabel=None)
                    axis.tick_params(left=False)

                n += 1

        fig.tight_layout(pad=FIG_PAD, h_pad=0.5, w_pad=0.5)
        fig.savefig(
            f"{fig_path}figure_{fig_num}.{FIGURE_FORMAT}",
            format=FIGURE_FORMAT,
            dpi=DPI,
        )
        fig_num += 1
        # endregion

        # region Plotting errors for h-v flash
        logger.info(f"{del_log}Plotting errors for h-v flash ..")
        fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH))
        axis = fig.add_subplot(2, 1, 1)
        axis.set_box_aspect(0.5)
        axis.set_title(f"p = {HV_ISOBAR * PRESSURE_SCALE} [{PRESSURE_SCALE_NAME}]")
        axis.set_xlabel(f"T [K]")
        axis.set_yscale("log")
        img_ip, leg_ip = plot_hv_iso(
            axis, T_ip, err_hv_p_ip, err_hv_T_ip, err_hv_s_ip, err_hv_y_ip
        )
        axis.legend(img_ip, leg_ip, loc="upper left", markerscale=MARKER_SCALE)

        axis = fig.add_subplot(2, 1, 2)
        axis.set_box_aspect(0.5)
        axis.set_title(f"T = {HV_ISOTHERM} [K]")
        axis.set_xlabel(f"p [{PRESSURE_SCALE_NAME}]")
        axis.set_yscale("log")
        img_ip, leg_ip = plot_hv_iso(
            axis,
            p_iT * PRESSURE_SCALE,
            err_hv_p_iT,
            err_hv_T_iT,
            err_hv_s_iT,
            err_hv_y_iT,
        )

        fig.tight_layout(pad=FIG_PAD, h_pad=0.5)
        fig.savefig(
            f"{fig_path}figure_{fig_num}.{FIGURE_FORMAT}",
            format=FIGURE_FORMAT,
            dpi=DPI,
        )
        fig_num += 1
        # endregion

        # region Plotting h-v isolines
        logger.info(f"{del_log}Plotting h-v flash isolines ..")
        fig = plt.figure(figsize=(FIGURE_WIDTH, 0.6 * ASPECT_RATIO * FIGURE_WIDTH))
        axis = fig.add_subplot(1, 1, 1)
        axis.set_box_aspect(0.5)
        axis.set_xlabel("T [K]")
        axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")

        img = plot_phase_split_GL(axis, p, T, split_pp)

        isobar_T = T_ip
        isobar_p = np.ones(len(T_ip)) * HV_ISOBAR * PRESSURE_SCALE

        isothermo_T = np.ones(len(p_iT)) * HV_ISOTHERM
        isotherm_p = p_iT * PRESSURE_SCALE

        marker_size = int(np.floor(MARKER_SIZE * 2 / 3))
        img_ip = axis.plot(
            isobar_T,
            isobar_p,
            "-o",
            fillstyle="none",
            color="black",
            markersize=marker_size,
            linewidth=1,
        )[0]
        img_iT = axis.plot(
            isothermo_T,
            isotherm_p,
            "-s",
            fillstyle="none",
            color="black",
            markersize=marker_size,
            linewidth=1,
        )[0]

        axis.set_ylim(p_min, 25.0)

        img_ = []
        leg_ = []
        crit = plot_crit_point_H2O(axis)
        img_ += crit[0]
        leg_ += crit[1]

        img_ += [img_ip, img_iT]
        leg_ += [
            f"isobar p = {HV_ISOBAR * PRESSURE_SCALE} [{PRESSURE_SCALE_NAME}]",
            f"isotherm T = {HV_ISOTHERM} [K] ",
        ]
        axis.legend(img_, leg_, loc="upper left", markerscale=MARKER_SCALE)

        cax = axis.inset_axes([1.04, 0.2, 0.05, 0.6])
        cb = fig.colorbar(img, cax=cax, orientation="vertical")
        cb.set_ticks([3 / 4 * k - 3 / 8 for k in range(1, 5)])
        cb.set_ticklabels(["N/A", "L", "GL", "G"])

        fig.tight_layout(pad=FIG_PAD)
        fig.savefig(
            f"{fig_path}figure_{fig_num}.{FIGURE_FORMAT}",
            format=FIGURE_FORMAT,
            dpi=DPI,
        )
        fig_num += 1
        # endregion
    else:
        fig_num += 10

    if PLOT_SECOND_EXAMPLE:

        # printing average number of iterations and failures
        print(f"\nExample 2: average num iter: {np.mean(num_iter_geo)}")
        print(
            f"Example 2: num of max iter reached: {max_iter_reached_geo.sum()} / {num_p_geo * num_x_geo}"
        )
        print(
            f"Example 2: num of failures: {doubt_geo.sum()} / {num_p_geo * num_x_geo}"
        )
        doubt_geo = np.logical_or(doubt_geo, max_iter_reached_geo)

        # region errors in y and T
        fig = plt.figure(figsize=(FIGURE_WIDTH, 2 * ASPECT_RATIO * FIGURE_WIDTH))
        axis = fig.add_subplot(2, 1, 1)
        axis.set_box_aspect(1)
        axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
        axis.set(xticklabels=[])
        axis.set(xlabel=None)
        axis.tick_params(bottom=False)

        y_error_geo[doubt_geo] = 0.0
        img = plot_abs_error_pT(axis, p_geo, x_geo * X_SCALE, y_error_geo, norm=None)
        cax = axis.inset_axes([1.04, 0.2, 0.05, 0.6])
        cb = fig.colorbar(
            img,
            ax=axis,
            cax=cax,
            orientation="vertical",  # format=ticker.FuncFormatter(_fmt),
        )

        mr = np.ma.array(split_geo, mask=np.logical_not(split_geo == 3))
        hatch = axis.pcolor(
            x_geo * X_SCALE,
            p_geo * PRESSURE_SCALE,
            mr,
            hatch="//",
            edgecolor="black",
            cmap=mpl.colors.ListedColormap(["none"]),
            facecolor="none",
            vmin=0,
            vmax=3,
            shading="nearest",
            lw=0,
            zorder=2,
        )
        img_v = [hatch]
        leg_v = [f"gas phase"]

        if np.any(doubt_geo):
            img_d = axis.plot(
                x_geo[doubt_geo] * X_SCALE,
                p_geo[doubt_geo] * PRESSURE_SCALE,
                "X",
                markersize=MARKER_SIZE,
                color="red",
            )
            leg_d = ["failure"]
        else:
            img_d = []
            leg_d = []

        axis.legend(
            img_v + img_d, leg_v + leg_d, loc="upper left", markerscale=MARKER_SCALE
        )

        # remove faulty entries to not mess with the scaling of the plot
        # mark faulty entries with separate markers
        t_mean = np.mean(cx_result_geo[np.logical_not(doubt_geo)])
        cx_result_geo[doubt_geo] = t_mean
        cx_error_geo[doubt_geo] = 0.0

        axis = fig.add_subplot(2, 1, 2)
        axis.set_box_aspect(1)
        axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
        if EXAMPLE_2_flash_type == "p-h":
            axis.set_xlabel(f"h [{X_SCALE_NAME}]")
        else:
            axis.set_xlabel("T [K]")
        img = plot_conjugate_x_for_px_flash(axis, p_geo, x_geo, cx_result_geo)
        cax = axis.inset_axes([1.04, 0.2, 0.05, 0.6])
        cb = fig.colorbar(
            img,
            ax=axis,
            cax=cax,
            orientation="vertical",
        )
        cx_max = cx_result_geo.max()
        cx_min = cx_result_geo.min()
        smallest, nextsmallest, *_ = np.partition(
            cx_result_geo[cx_result_geo >= 0].flatten(), 1
        )
        cbt = np.linspace(cx_min, cx_max, 5, endpoint=True)
        cb.set_ticks(cbt.astype(int))

        if np.any(doubt_geo):
            img_d = axis.plot(
                x_geo[doubt_geo] * X_SCALE,
                p_geo[doubt_geo] * PRESSURE_SCALE,
                "X",
                markersize=MARKER_SIZE,
                color="red",
            )

        fig.tight_layout(pad=10 * FIG_PAD, w_pad=0.1)
        fig.savefig(
            f"{fig_path}figure_{fig_num}.{FIGURE_FORMAT}",
            format=FIGURE_FORMAT,
            dpi=DPI,
        )
        fig_num += 1
        # endregion

        # region Number of iterations
        logger.info(f"{del_log}Plotting iteration numbers for second example ..")
        fig = plt.figure(figsize=(FIGURE_WIDTH, ASPECT_RATIO * FIGURE_WIDTH))
        axis = fig.add_subplot(1, 1, 1)
        axis.set_box_aspect(1)
        axis.set_ylabel(f"p [{PRESSURE_SCALE_NAME}]")
        if EXAMPLE_2_flash_type == "p-h":
            axis.set_xlabel(f"h [{X_SCALE_NAME}]")
        else:
            axis.set_xlabel("T [K]")

        img = plot_abs_error_pT(axis, p_geo, x_geo * X_SCALE, num_iter_geo, norm=None)
        img_, leg_ = plot_max_iter_reached(
            axis, p_geo, x_geo * X_SCALE, max_iter_reached_geo
        )

        axis.legend(img_, leg_, loc="upper left", markerscale=MARKER_SCALE)

        cax = axis.inset_axes([1.04, 0.2, 0.05, 0.6])
        cb = fig.colorbar(
            img,
            ax=axis,
            cax=cax,
            orientation="vertical",
        )
        cbt = cb.get_ticks()
        cbt = cbt[cbt < num_iter_geo.max()]
        cbt = np.sort(np.hstack([cbt, np.array([num_iter_geo.max()])]))
        if np.abs(cbt[-1] - cbt[-2]) <= 4:
            cbt = np.hstack([cbt[:-2], cbt[-1:]])
        cbt = cbt[cbt <= num_iter_geo.max()]
        cb.set_ticks(cbt.astype(int))

        fig.tight_layout(pad=FIG_PAD)
        fig.savefig(
            f"{fig_path}figure_{fig_num}.{FIGURE_FORMAT}",
            format=FIGURE_FORMAT,
            dpi=DPI,
        )
        fig_num += 1
        # endregion
    else:
        fig_num += 2
