"""Script for plotting various figures for the comparison of the pT flash (H2O, CO2)
with thermo data.

This script follows the patterns introduced in ``calc_flash_h2o_co2.py`` and can
be performed on the files produced by it.

"""
import pathlib
import sys
from typing import Any, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

from thermo_comparison import (
    COMPONENTS,
    FAILED_ENTRY,
    FEED,
    MISSING_ENTRY,
    NAN_ENTRY,
    PHASES,
    T_HEADER,
    composition_HEADER,
    cond_end_HEADER,
    cond_start_HEADER,
    gas_frac_HEADER,
    get_result_headers,
    h_HEADER,
    is_supercrit_HEADER,
    liq_frac_HEADER,
    num_iter_HEADER,
    p_HEADER,
    path,
    phases_HEADER,
    read_headers,
    read_results,
    success_HEADER,
)

### General settings

# files containing data
THERMO_FILE: str = f"data/thermodata_pT2k_co2_1e-2.csv"
RESULT_FILE: str = f"data/results/results_pT2k_co2_1e-2_par.csv"
# Path to where figures should be stored
FIGURE_PATH: str = f"data/results/figures/"
# Indicate flash type 'pT' or 'ph' to plot respectively
FLASH_TYPE: str = "pT"
# Scaling of pressure and enthalpy from original units [Pa] and [J / mol]
P_FACTOR: float = 1e-3
P_UNIT: str = "[kPa]"
H_FACTOR: float = 1e-3
H_UNIT: str = "[J / mol]"
# flag to scale the pressure logarithmically in the plots
P_LOG_SCALE: bool = False
# image size information
FIG_WIDTH = 15  # in inches, 1080 / 1920 ratio applied to height
DPI: int = 500

# do not change this
if FLASH_TYPE == "pT":
    H_FACTOR = 1


def get_px_index_map(
    p: list[Any], x: list[Any]
) -> tuple[np.ndarray, np.ndarray, dict[tuple[Any, Any], int]]:
    """Return for un-meshed p-x data a map ``(pi, xi) -> i``, where ``i`` ranges from
    0 to length of passed lists.

    ``p`` and ``x`` are casted into numpy arrays with dtype float and returned
    with unique, sorted values.

    Use this to identify rows in data columns belonging to point ``(pi, xi)``.

    """
    assert len(p) == len(x), "Un-meshed data has unequal length."
    p_vec = np.array([float(p_) for p_ in p])
    x_vec = np.array([float(x_) for x_ in x])

    p_arr = np.unique(np.sort(p_vec))
    x_arr = np.unique(np.sort(x_vec))
    return (
        p_arr,
        x_arr,
        dict(
            [
                ((p_, x_), i)
                for p_, x_, i in zip(p_vec, x_vec, np.arange(p_vec.shape[0]))
            ]
        ),
    )


def _lump_liquid_phases(
    x_mesh: np.ndarray,
    p_mesh: np.ndarray,
    results: dict[str, list],
    component: str,
    px_map: dict,
) -> list[Any]:
    """Lumps the liquid fractions together at points where thermo indicates a LL split."""
    x_c = np.zeros(p_mesh.shape[0] * p_mesh.shape[1])
    liq1 = PHASES[1]
    liq2 = PHASES[2]

    for i in range(p_mesh.shape[0]):
        for j in range(p_mesh.shape[1]):
            p_ = p_mesh[i, j]
            x_ = x_mesh[i, j]
            px = (p_, x_)

            idx = px_map[px]

            L_count = results[phases_HEADER][idx].count("L")
            # G_count = results[phases_HEADER][idx].count('G')

            if L_count == 2:
                y1 = float(results[liq_frac_HEADER[0]][idx])
                y2 = float(results[liq_frac_HEADER[1]][idx])
                x1 = float(results[composition_HEADER[component][liq1]][idx])
                x2 = float(results[composition_HEADER[component][liq2]][idx])
                x_c[idx] = y1 * x1 + y2 * x2
            elif L_count == 1:
                x_c[idx] = float(results[composition_HEADER[component][liq1]][idx])
            elif L_count == 0:
                x_c[idx] = np.nan

    return [str(np.nan) if np.isnan(x_) else x_ for x_ in x_c]


def _plot_overshoot(
    axis: plt.Axes,
    vals: np.ndarray,
    x_mesh: np.ndarray,
    p_mesh: np.ndarray,
    val_name: str,
):
    over_shoot = vals > 1
    under_shoot = vals < 0
    legend_img = []
    legend_txt = []
    if np.any(under_shoot):
        img_u = axis.plot(
            (x_mesh[under_shoot] * H_FACTOR).flat,
            (p_mesh[under_shoot] * P_FACTOR).flat,
            "v",
            markersize=3,
            color="black",
        )
        legend_img.append(img_u[0])
        legend_txt.append(f"{val_name} < 0")
    if np.any(over_shoot):
        img_o = axis.plot(
            (x_mesh[under_shoot] * H_FACTOR).flat,
            (p_mesh[under_shoot] * P_FACTOR).flat,
            "^",
            markersize=3,
            color="red",
        )
        legend_img.append(img_o[0])
        legend_txt.append(f"{val_name} > 1")
    return legend_img, legend_txt


def _plot_crit_point_pT(axis: plt.Axes):
    """Plot critical pressure and temperature in p-T plot for components H2O and CO2."""

    pc_co2 = 7376460 * P_FACTOR
    Tc_co2 = 304.2
    pc_h2o = 22048320 * P_FACTOR
    Tc_h2o = 647.14

    img_h2o = axis.plot(Tc_h2o, pc_h2o, "*", markersize=7, color="cyan")
    img_co2 = axis.plot(Tc_co2, pc_co2, "*", markersize=7, color="lime")

    return [img_h2o[0], img_co2[0]], ["H2O Crit. Point", "CO2 Crit. Point"]


def _plot_supercrit(
    axis: plt.Axes,
    p_mesh: np.ndarray,
    x_mesh: np.ndarray,
    success: list[Any],
    supercrit: list[Any],
    px_map: dict,
):
    """Plot markers at points where the results indicate a supercritical mixture."""
    is_supercrit = np.zeros(p_mesh.shape, dtype=bool)
    ignore = [FAILED_ENTRY, str(NAN_ENTRY), MISSING_ENTRY]
    for i in range(p_mesh.shape[0]):
        for j in range(p_mesh.shape[1]):
            p_ = p_mesh[i, j]
            x_ = x_mesh[i, j]
            px = (p_, x_)

            idx = px_map[px]
            s = int(success[idx])
            if s == 1:
                v = bool(int(supercrit[idx]))
                is_supercrit[i, j] = v

    legend_img = []
    legend_txt = []
    if np.any(is_supercrit):
        img_u = axis.plot(
            (x_mesh[is_supercrit] * H_FACTOR).flat,
            (p_mesh[is_supercrit] * P_FACTOR).flat,
            "*",
            markersize=3,
            color="coral",
        )
        legend_img.append(img_u[0])
        legend_txt.append(f"supercritical")
    return legend_img, legend_txt


def _plot_liquid_phase_splits(
    axis: plt.Axes,
    p_mesh: np.ndarray,
    x_mesh: np.ndarray,
    vals: list[Any],
    px_map: dict,
):
    """Plot points where thermo indicates a split in two liquid phases"""
    GLL_mesh = np.zeros(p_mesh.shape)
    LL_mesh = np.zeros(p_mesh.shape)
    for i in range(p_mesh.shape[0]):
        for j in range(p_mesh.shape[1]):
            p_ = p_mesh[i, j]
            x_ = x_mesh[i, j]
            px = (p_, x_)

            v = vals[px_map[px]]

            if "GLL" in v:
                GLL_mesh[i, j] = 1
            if "LL" in v and not "G" in v:
                LL_mesh[i, j] = 1

    LL_split = LL_mesh == 1
    GLL_split = GLL_mesh == 1

    legend_img = []
    legend_txt = []
    if np.any(LL_split):
        img_u = axis.plot(
            (x_mesh[LL_split] * H_FACTOR).flat,
            (p_mesh[LL_split] * P_FACTOR).flat,
            "+",
            markersize=3,
            color="red",
        )
        legend_img.append(img_u[0])
        legend_txt.append(f"LL split")
    if np.any(GLL_split):
        img_o = axis.plot(
            x_mesh[GLL_split].flat * H_FACTOR,
            p_mesh[GLL_split].flat * P_FACTOR,
            "P",
            markersize=3,
            color="red",
        )
        legend_img.append(img_o[0])
        legend_txt.append(f"GLL split")
    return legend_img, legend_txt


def plot_success(
    axis: plt.Axes,
    p_mesh: np.ndarray,
    x_mesh: np.ndarray,
    success: list[Any],
    supercrit: list[Any],
    x_name: str,
    px_map: dict,
):
    """Plots a discrete success mesh plot for given p-x data.

    Use the index map created by ``get_px_index_map``.

    """

    v_mesh = np.zeros(p_mesh.shape)
    for i in range(p_mesh.shape[0]):
        for j in range(p_mesh.shape[1]):
            p_ = p_mesh[i, j]
            x_ = x_mesh[i, j]
            px = (p_, x_)

            v = success[px_map[px]]

            if v in [MISSING_ENTRY, str(NAN_ENTRY)]:
                v_mesh[i, j] = 0
            elif v == FAILED_ENTRY:
                v_mesh[i, j] = 1
            else:
                v = int(v)
                if v == 0:
                    v_mesh[i, j] = 1
                elif v == 1:
                    v_mesh[i, j] = 2
                else:
                    raise NotImplementedError(f"Unexpected entry in success-data: {v}")

    success_rate = np.count_nonzero(v_mesh == 2)

    cmap = mpl.colors.ListedColormap(["white", "red", "green"])
    img = axis.pcolormesh(
        x_mesh * H_FACTOR,
        p_mesh * P_FACTOR,
        v_mesh,
        cmap=cmap,
        vmin=0,
        vmax=2,
        shading="nearest",
    )
    img_sc, leg_sc = _plot_supercrit(axis, p_mesh, x_mesh, success, supercrit, px_map)
    # img_sc, leg_sc = [list(), list()]
    if "T" in x_name:
        img_c, leg_c = _plot_crit_point_pT(axis)
    else:
        img_c = list()
        leg_c = list()
    if img_sc + img_c:
        axis.legend(img_sc + img_c, leg_sc + leg_c, loc="upper left")
    rate = (success_rate) / (p_mesh.shape[0] * p_mesh.shape[1]) * 100
    axis.set_title(f"Success rate: {'{:.2f}'.format(rate)} %")
    axis.set_xlabel(x_name)
    axis.set_ylabel(f"p {P_UNIT}")
    if P_LOG_SCALE:
        axis.set_yscale("log")

    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(img, cax=cax, orientation="vertical")
    cb.set_ticks([0, 1, 2])
    cb.set_ticklabels(["N/A", "failed", "succeeded"])


def plot_phase_regions_thermo(
    axis: plt.Axes,
    p_mesh: np.ndarray,
    x_mesh: np.ndarray,
    vals: list[Any],
    x_name: str,
    px_map: dict,
):
    """Plots a discrete phase regions map."""
    v_mesh = np.zeros(p_mesh.shape)
    for i in range(p_mesh.shape[0]):
        for j in range(p_mesh.shape[1]):
            p_ = p_mesh[i, j]
            x_ = x_mesh[i, j]
            px = (p_, x_)

            v = vals[px_map[px]]

            if v == "G":
                v_mesh[i, j] = 2
            elif "L" in v and not "G" in v:
                v_mesh[i, j] = 1
            elif "GL" in v:
                v_mesh[i, j] = 3

    cmap = mpl.colors.ListedColormap(["white", "blue", "red", "yellow"])
    img = axis.pcolormesh(
        x_mesh * H_FACTOR,
        p_mesh * P_FACTOR,
        v_mesh,
        cmap=cmap,
        vmin=0,
        vmax=3,
        shading="nearest",
    )
    if "T" in x_name:
        img_c, leg_c = _plot_crit_point_pT(axis)
    else:
        img_c = list()
        leg_c = list()
    img_ps, leg_ps = _plot_liquid_phase_splits(axis, p_mesh, x_mesh, vals, px_map)
    axis.legend(img_c + img_ps, leg_c + leg_ps, loc="upper left")
    axis.set_title("Phase regions (thermo)")
    axis.set_xlabel(x_name)
    axis.set_ylabel(f"p {P_UNIT}")
    if P_LOG_SCALE:
        axis.set_yscale("log")

    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(img, cax=cax, orientation="vertical")
    cb.set_ticks([0, 1, 2, 3])
    cb.set_ticklabels(["N/A", "liquid", "gas", "2-phase"])


def plot_phase_regions_from_y(
    axis: plt.Axes,
    p_mesh: np.ndarray,
    x_mesh: np.ndarray,
    success: list[Any],
    vals: list[Any],
    x_name: str,
    px_map: dict,
):
    """Plots a discrete phase regions map from gas fraction values."""
    v_mesh = np.zeros(p_mesh.shape)
    for i in range(p_mesh.shape[0]):
        for j in range(p_mesh.shape[1]):
            p_ = p_mesh[i, j]
            x_ = x_mesh[i, j]
            px = (p_, x_)
            idx = px_map[px]
            v = vals[idx]

            s = int(success[idx])
            if s == 1:
                v = float(v)
                if v >= 1:
                    v_mesh[i, j] = 2
                elif v <= 0:
                    v_mesh[i, j] = 1
                elif 0 < v < 1:
                    v_mesh[i, j] = 3

    cmap = mpl.colors.ListedColormap(["white", "blue", "red", "yellow"])
    img = axis.pcolormesh(
        x_mesh * H_FACTOR,
        p_mesh * P_FACTOR,
        v_mesh,
        cmap=cmap,
        vmin=0,
        vmax=3,
        shading="nearest",
    )
    if "T" in x_name:
        img_c, leg_c = _plot_crit_point_pT(axis)
        axis.legend(img_c, leg_c, loc="upper left")

    axis.set_title("Phase regions (based on y)")
    axis.set_xlabel(x_name)
    axis.set_ylabel(f"p {P_UNIT}")
    if P_LOG_SCALE:
        axis.set_yscale("log")

    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(img, cax=cax, orientation="vertical")
    cb.set_ticks([0, 1, 2, 3])
    cb.set_ticklabels(["N/A", "liquid", "gas", "2-phase"])


def plot_any_vals(
    axis: plt.Axes,
    p_mesh: np.ndarray,
    x_mesh: np.ndarray,
    success: list[Any],
    vals: list[Any],
    val_type: Any,
    title: str,
    x_name: str,
    px_map: dict,
    norm: Optional[str] = None,
):
    """Plots any values.

    Specify ``norm`` using matplotlib if you wish to scale the values, otherwise
    it will scaled linear between min and max value.
    Available norms: ``'log'``
    """
    val_mesh = np.zeros(p_mesh.shape)
    ignore = [FAILED_ENTRY, str(NAN_ENTRY), MISSING_ENTRY]
    for i in range(p_mesh.shape[0]):
        for j in range(p_mesh.shape[1]):
            p_ = p_mesh[i, j]
            x_ = x_mesh[i, j]
            px = (p_, x_)

            idx = px_map[px]

            v = vals[idx]
            s = int(success[idx])
            # assuming always present
            if s == 1:
                if v not in ignore:
                    v = val_type(v)
                    val_mesh[i, j] = v

    vmin, vmax = val_mesh.min(), val_mesh.max()
    cmap = "Greys"

    if norm:
        if norm == "log":
            norm = mpl.colors.SymLogNorm(
                linthresh=1e-3, linscale=0.5, vmin=vmin, vmax=vmax
            )
        else:
            raise ValueError(f"Unknown norm option: {norm}")
        img = axis.pcolormesh(
            x_mesh * H_FACTOR,
            p_mesh * P_FACTOR,
            val_mesh,
            cmap=cmap,
            norm=norm,
            shading="nearest",
        )
    else:
        img = axis.pcolormesh(
            x_mesh * H_FACTOR,
            p_mesh * P_FACTOR,
            val_mesh,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="nearest",
        )

    if "T" in x_name:
        img_c, leg_c = _plot_crit_point_pT(axis)
        axis.legend(img_c, leg_c, loc="upper left")
    axis.set_title(f"{title}")
    axis.set_xlabel(x_name)
    axis.set_ylabel(f"p {P_UNIT}")
    if P_LOG_SCALE:
        axis.set_yscale("log")
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(
        img,
        cax=cax,
        orientation="vertical",
    )
    cb.set_label(
        f"Min.: {'{:.3e}'.format(vmin)}"
        + f" ; Max.: {'{:.3e}'.format(vmax)}"
        + f"\nMean: {'{:.3e}'.format(np.mean(val_mesh))}"
    )


def plot_fraction_values(
    axis: plt.Axes,
    p_mesh: np.ndarray,
    x_mesh: np.ndarray,
    success: list[Any],
    vals: list[Any],
    val_name: str,
    x_name: str,
    px_map: dict,
    norm: Optional[str] = None,
):
    """Plots fractional values between 0 and 1, with over and undershoots.

    Specify ``norm`` using matplotlib if you wish to scale the values, otherwise
    it will scaled linear between min and max value.
    Available norms: ``'log'``
    """
    frac_mesh = np.zeros(p_mesh.shape)
    ignore = [FAILED_ENTRY, str(NAN_ENTRY), MISSING_ENTRY]
    for i in range(p_mesh.shape[0]):
        for j in range(p_mesh.shape[1]):
            p_ = p_mesh[i, j]
            x_ = x_mesh[i, j]
            px = (p_, x_)

            idx = px_map[px]

            v = vals[idx]

            s = int(success[idx])
            # if successful, check for nan entry or anything else
            if s == 1:
                if v not in ignore:
                    v = float(v)
                    frac_mesh[i, j] = v

    vmin, vmax = frac_mesh.min(), frac_mesh.max()
    cmap = "coolwarm"

    if norm:
        if norm == "log":
            norm = mpl.colors.SymLogNorm(
                linthresh=1e-3, linscale=0.5, vmin=vmin, vmax=vmax
            )
        else:
            raise ValueError(f"Unknown norm option: {norm}")
        img = axis.pcolormesh(
            x_mesh * H_FACTOR,
            p_mesh * P_FACTOR,
            frac_mesh,
            cmap=cmap,
            norm=norm,
            shading="nearest",
        )
    else:
        img = axis.pcolormesh(
            x_mesh * H_FACTOR,
            p_mesh * P_FACTOR,
            frac_mesh,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="nearest",
        )

    img_uo, leg_uo = _plot_overshoot(axis, frac_mesh, x_mesh, p_mesh, val_name)
    if "T" in x_name:
        img_c, leg_c = _plot_crit_point_pT(axis)
    else:
        img_c = list()
        leg_c = list()
    if img_c + img_uo:
        axis.legend(img_uo + img_c, leg_uo + leg_c, loc="upper left")
    axis.set_title(f"Fraction values: {val_name}")
    axis.set_xlabel(x_name)
    axis.set_ylabel(f"p {P_UNIT}")
    if P_LOG_SCALE:
        axis.set_yscale("log")
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(
        img,
        cax=cax,
        orientation="vertical",
    )
    cb.set_label(f"Min.: {vmin}\nMax.: {vmax}")
    cb.set_label(f"Min.: {'{:.6f}'.format(vmin)}\nMax.: {'{:.6f}'.format(vmax)}")


def plot_abs_error_values(
    axis: plt.Axes,
    p_mesh: np.ndarray,
    x_mesh: np.ndarray,
    success: list[Any],
    vals: list[Any],
    target_vals: list[Any],
    val_name: str,
    x_name: str,
    px_map: dict,
    results: dict,
    is_rel_frac: bool = False,
    comp: Optional[str] = None,
    norm: Optional[str] = None,
):
    """Plots absolute error between ``vals`` and ``target_vals``.

    Specify ``norm`` using matplotlib if you wish to scale the values, otherwise
    it will scaled linear between min and max value.
    Available norms: ``'log'``
    """
    err_mesh = np.zeros(p_mesh.shape)
    ignore = [FAILED_ENTRY, str(NAN_ENTRY), MISSING_ENTRY]
    for i in range(p_mesh.shape[0]):
        for j in range(p_mesh.shape[1]):
            p_ = p_mesh[i, j]
            x_ = x_mesh[i, j]
            px = (p_, x_)

            idx = px_map[px]

            v = vals[idx]
            v_target = target_vals[idx]

            s = int(success[idx])
            # if successful, check for nan entry or anything else
            if s == 1:
                # work around for relative fractions to filter out error
                # introduced by extended fractions
                if is_rel_frac:
                    y = float(results[gas_frac_HEADER][idx])
                    if (
                        (y <= 0.0 or y >= 1.0)
                        and v_target not in ignore
                        and comp is not None
                    ):
                        v_target = FEED[comp]
                if (v not in ignore) and (v_target not in ignore):
                    v = float(v)
                    v_target = float(v_target)
                    err_mesh[i, j] = np.abs(v - v_target)

    vmin, vmax = err_mesh.min(), err_mesh.max()
    cmap = "Greys"

    if norm:
        if norm == "log":
            norm = mpl.colors.SymLogNorm(
                linthresh=1e-3, linscale=0.5, vmin=vmin, vmax=vmax
            )
        else:
            raise ValueError(f"Unknown norm option: {norm}")
        img = axis.pcolormesh(
            x_mesh * H_FACTOR,
            p_mesh * P_FACTOR,
            err_mesh,
            cmap=cmap,
            norm=norm,
            shading="nearest",
        )
    else:
        img = axis.pcolormesh(
            x_mesh * H_FACTOR,
            p_mesh * P_FACTOR,
            err_mesh,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="nearest",
        )

    if "T" in x_name:
        img_c, leg_c = _plot_crit_point_pT(axis)
        axis.legend(img_c, leg_c, loc="upper left")

    axis.set_title(f"Absolute error: {val_name}")
    axis.set_xlabel(x_name)
    axis.set_ylabel(f"p {P_UNIT}")
    if P_LOG_SCALE:
        axis.set_yscale("log")
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(
        img,
        cax=cax,
        orientation="vertical",
    )
    cb.set_label(
        "Max. abs. error: "
        + "{:.0e}".format(float(err_mesh.max()))
        + "\nL2-error: "
        + "{:.0e}".format(float(np.sqrt(np.sum(np.square(err_mesh)))))
    )


def plot_duality_gap(
    axis: plt.Axes,
    p_mesh: np.ndarray,
    x_mesh: np.ndarray,
    results: dict[str, list],
    phase_name: str,
    x_name: str,
    px_map: dict,
):
    """Plot the duality gap for a given phase."""
    gap_mesh = np.zeros(p_mesh.shape)
    ignore = [FAILED_ENTRY, str(NAN_ENTRY), MISSING_ENTRY]
    h2o, co2 = COMPONENTS
    for i in range(p_mesh.shape[0]):
        for j in range(p_mesh.shape[1]):
            p_ = p_mesh[i, j]
            x_ = x_mesh[i, j]
            px = (p_, x_)

            idx = px_map[px]

            s = int(results[success_HEADER][idx])

            # if successful, check for nan entry or anything else
            if s == 1:
                x_h2o = results[composition_HEADER[h2o][phase_name]][idx]
                x_co2 = results[composition_HEADER[co2][phase_name]][idx]
                x_h2o = float(x_h2o)
                x_co2 = float(x_co2)
                gap_mesh[i, j] = 1 - x_co2 - x_h2o

    vmin, vmax = gap_mesh.min(), gap_mesh.max()
    cmap = "Greys"

    img = axis.pcolormesh(
        x_mesh * H_FACTOR,
        p_mesh * P_FACTOR,
        gap_mesh,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="nearest",
    )

    if "T" in x_name:
        img_c, leg_c = _plot_crit_point_pT(axis)
        axis.legend(img_c, leg_c, loc="upper left")

    axis.set_title(f"Duality gap: phase {phase_name}")
    axis.set_xlabel(x_name)
    axis.set_ylabel(f"p {P_UNIT}")
    if P_LOG_SCALE:
        axis.set_yscale("log")
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(
        img,
        cax=cax,
        orientation="vertical",
    )


# num_levels = 20
# midpoint = 0
# levels = np.hstack([np.linspace(vmin, 0.25, num_levels), np.linspace(0.25, vmax, 10, )])
# levels = np.sort(np.unique(levels))
# midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
# vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
# colors = mcm.get_cmap('coolwarm_r')(vals)  # RdYlGn
# cmap, norm = from_levels_and_colors(levels, colors)

if __name__ == "__main__":
    figwidth = 15
    scipt_path = path()
    fname = RESULT_FILE[RESULT_FILE.rfind("/") + 1 : RESULT_FILE.rfind(".csv")]

    headers = get_result_headers()
    results = read_results([RESULT_FILE], headers)

    thermo_headers = read_headers(THERMO_FILE)
    thermo_results = read_results([THERMO_FILE], thermo_headers)

    p_points = thermo_results[p_HEADER]
    if FLASH_TYPE == "pT":
        x_name = "T [K]"
        x_points = thermo_results[T_HEADER]
    elif FLASH_TYPE == "ph":
        x_name = f"h {H_UNIT}"
        x_points = thermo_results[h_HEADER]
    else:
        raise NotImplementedError(f"Unsupported flash type: {FLASH_TYPE}")

    p_vec, x_vec, px_map = get_px_index_map(p_points, x_points)

    x_mesh, p_mesh = np.meshgrid(x_vec, p_vec)

    liq = PHASES[1]
    gas = PHASES[0]
    h2o, co2 = COMPONENTS

    # region Plot 0: Phase regions
    print("Plotting: Phase regions", flush=True)
    fig = plt.figure(figsize=(FIG_WIDTH, 1080 / 1920 * FIG_WIDTH))
    gs = fig.add_gridspec(1, 2)
    fig.suptitle(f"Overview: VLE with H2O and CO2")
    axis = fig.add_subplot(gs[0, 0])
    plot_phase_regions_thermo(
        axis, p_mesh, x_mesh, thermo_results[phases_HEADER], x_name, px_map
    )

    axis = fig.add_subplot(gs[0, 1])
    plot_phase_regions_from_y(
        axis,
        p_mesh,
        x_mesh,
        results[success_HEADER],
        results[gas_frac_HEADER],
        x_name,
        px_map,
    )

    fig.tight_layout()
    fig.savefig(
        f"{str(scipt_path)}/{FIGURE_PATH}0_regions__{fname}.png",
        format="png",
        dpi=DPI,
    )
    # endregion

    # region Plot 1: Success and number of iterations
    print("Plotting: Succes and num iter", flush=True)
    fig = plt.figure(figsize=(FIG_WIDTH, 1080 / 1920 * FIG_WIDTH))
    gs = fig.add_gridspec(1, 2)
    fig.suptitle(f"Overview on numeric procedure.")
    axis = fig.add_subplot(gs[0, 0])
    plot_success(
        axis,
        p_mesh,
        x_mesh,
        results[success_HEADER],
        results[is_supercrit_HEADER],
        x_name,
        px_map,
    )

    axis = fig.add_subplot(gs[0, 1])
    plot_any_vals(
        axis,
        p_mesh,
        x_mesh,
        results[success_HEADER],
        results[num_iter_HEADER],
        int,
        "Number of iterations",
        x_name,
        px_map,
        norm=None,
    )

    fig.tight_layout()
    fig.savefig(
        f"{str(scipt_path)}/{FIGURE_PATH}1_overview__{fname}.png",
        format="png",
        dpi=DPI,
    )
    # endregion

    # region Plot 2: Gas fraction values
    print("Plotting: Gas fraction data", flush=True)
    fig = plt.figure(figsize=(FIG_WIDTH, 1080 / 1920 * FIG_WIDTH))
    gs = fig.add_gridspec(1, 2)
    fig.suptitle(f"Gas fraction")
    axis = fig.add_subplot(gs[0, 0])

    plot_fraction_values(
        axis,
        p_mesh,
        x_mesh,
        results[success_HEADER],
        results[gas_frac_HEADER],
        "y",
        x_name,
        px_map,
        norm="log",
    )

    axis = fig.add_subplot(gs[0, 1])
    plot_abs_error_values(
        axis,
        p_mesh,
        x_mesh,
        results[success_HEADER],
        results[gas_frac_HEADER],
        thermo_results[gas_frac_HEADER],
        "y",
        x_name,
        px_map,
        results,
        is_rel_frac=False,
        comp=None,
        norm="log",
    )

    fig.tight_layout()
    fig.savefig(
        f"{str(scipt_path)}/{FIGURE_PATH}2_gasfrac__{fname}.png",
        format="png",
        dpi=DPI,
    )
    # endregion

    # region Plot 3: Fraction values in Liquid phase
    print("Plotting: Liquid phase data", flush=True)
    fig = plt.figure(figsize=(FIG_WIDTH, 1080 / 1920 * FIG_WIDTH))
    gs = fig.add_gridspec(2, 2)
    fig.suptitle(f"Liquid phase composition")

    frac_header = composition_HEADER[h2o][liq]
    axis = fig.add_subplot(gs[0, 0])
    plot_fraction_values(
        axis,
        p_mesh,
        x_mesh,
        results[success_HEADER],
        results[frac_header],
        frac_header,
        x_name,
        px_map,
        norm=None,
    )
    axis = fig.add_subplot(gs[0, 1])
    x_c = _lump_liquid_phases(x_mesh, p_mesh, thermo_results, h2o, px_map)
    plot_abs_error_values(
        axis,
        p_mesh,
        x_mesh,
        results[success_HEADER],
        results[frac_header],
        x_c,
        frac_header,
        x_name,
        px_map,
        results,
        is_rel_frac=True,
        comp=h2o,
        norm=None,
    )

    frac_header = composition_HEADER[co2][liq]
    axis = fig.add_subplot(gs[1, 0])
    plot_fraction_values(
        axis,
        p_mesh,
        x_mesh,
        results[success_HEADER],
        results[frac_header],
        frac_header,
        x_name,
        px_map,
        norm=None,
    )
    axis = fig.add_subplot(gs[1, 1])
    x_c = _lump_liquid_phases(x_mesh, p_mesh, thermo_results, co2, px_map)
    plot_abs_error_values(
        axis,
        p_mesh,
        x_mesh,
        results[success_HEADER],
        results[frac_header],
        x_c,
        frac_header,
        x_name,
        px_map,
        results,
        is_rel_frac=True,
        comp=co2,
        norm=None,
    )

    fig.tight_layout()
    fig.savefig(
        f"{str(scipt_path)}/{FIGURE_PATH}3_liqcomp__{fname}.png",
        format="png",
        dpi=DPI,
    )
    # endregion

    # region Plot 4: Fraction values in Gas phase
    print("Plotting: Gas phase data", flush=True)
    fig = plt.figure(figsize=(FIG_WIDTH, 1080 / 1920 * FIG_WIDTH))
    gs = fig.add_gridspec(2, 2)
    fig.suptitle(f"Gas phase composition")

    frac_header = composition_HEADER[h2o][gas]
    axis = fig.add_subplot(gs[0, 0])
    plot_fraction_values(
        axis,
        p_mesh,
        x_mesh,
        results[success_HEADER],
        results[frac_header],
        frac_header,
        x_name,
        px_map,
        norm=None,
    )
    axis = fig.add_subplot(gs[0, 1])
    plot_abs_error_values(
        axis,
        p_mesh,
        x_mesh,
        results[success_HEADER],
        results[frac_header],
        thermo_results[frac_header],
        frac_header,
        x_name,
        px_map,
        results,
        is_rel_frac=True,
        comp=h2o,
        norm=None,
    )

    frac_header = composition_HEADER[co2][gas]
    axis = fig.add_subplot(gs[1, 0])
    plot_fraction_values(
        axis,
        p_mesh,
        x_mesh,
        results[success_HEADER],
        results[frac_header],
        frac_header,
        x_name,
        px_map,
        norm=None,
    )
    axis = fig.add_subplot(gs[1, 1])
    plot_abs_error_values(
        axis,
        p_mesh,
        x_mesh,
        results[success_HEADER],
        results[frac_header],
        thermo_results[frac_header],
        frac_header,
        x_name,
        px_map,
        results,
        is_rel_frac=True,
        comp=co2,
        norm=None,
    )

    fig.tight_layout()
    fig.savefig(
        f"{str(scipt_path)}/{FIGURE_PATH}4_gascomp__{fname}.png",
        format="png",
        dpi=DPI,
    )
    # endregion

    # region Plot 5: Duality gaps
    print("Plotting: Duality gaps", flush=True)
    fig = plt.figure(figsize=(FIG_WIDTH, 1080 / 1920 * FIG_WIDTH))
    gs = fig.add_gridspec(1, 2)
    fig.suptitle(f"Duality gaps")

    phase_name = PHASES[1]
    axis = fig.add_subplot(gs[0, 0])
    plot_duality_gap(
        axis,
        p_mesh,
        x_mesh,
        results,
        phase_name,
        x_name,
        px_map,
    )
    phase_name = PHASES[0]
    axis = fig.add_subplot(gs[0, 1])
    plot_duality_gap(
        axis,
        p_mesh,
        x_mesh,
        results,
        phase_name,
        x_name,
        px_map,
    )

    fig.tight_layout()
    fig.savefig(
        f"{str(scipt_path)}/{FIGURE_PATH}5_duality__{fname}.png",
        format="png",
        dpi=DPI,
    )
    # endregion

    # region Plot 6: Condition numbers
    print("Plotting: Condition numbers", flush=True)
    fig = plt.figure(figsize=(FIG_WIDTH, 1080 / 1920 * FIG_WIDTH))
    gs = fig.add_gridspec(1, 2)
    fig.suptitle(f"Condition numbers")
    axis = fig.add_subplot(gs[0, 0])
    plot_any_vals(
        axis,
        p_mesh,
        x_mesh,
        results[success_HEADER],
        results[cond_start_HEADER],
        float,
        "Condition number: Initial guess",
        x_name,
        px_map,
        norm="log",
    )

    axis = fig.add_subplot(gs[0, 1])
    plot_any_vals(
        axis,
        p_mesh,
        x_mesh,
        results[success_HEADER],
        results[cond_end_HEADER],
        float,
        "Condition number: Converged state",
        x_name,
        px_map,
        norm="log",
    )

    fig.tight_layout()
    fig.savefig(
        f"{str(scipt_path)}/{FIGURE_PATH}6_condition__{fname}.png",
        format="png",
        dpi=DPI,
    )

    # endregion

    print("Plotting: Done", flush=True)
