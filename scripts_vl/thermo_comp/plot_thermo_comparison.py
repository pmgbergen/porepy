"""Script for plotting various figures for the comparison of the pT flash (H2O, CO2)
with thermo data.

This script follows the patterns introduced in ``calc_flash_h2o_co2.py`` and can
be performed on the files produced by it.

"""
import sys
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Any

sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

from thermo_comparison import (
    path,
    read_results,
    read_headers,
    get_result_headers,
    p_HEADER,
    T_HEADER,
    h_HEADER,
    success_HEADER,
    phases_HEADER,
    FAILED_ENTRY,
    NAN_ENTRY,
    MISSING_ENTRY
)

### General settings

# files containing data
THERMO_FILE: str = f"data/thermodata_pT10K.csv"
RESULT_FILE: str = f"data/results/results_pT10k_par_wo-reg.csv"
# Path to where figures should be stored
FIGURE_PATH: str = f"data/results/figures/"
# Indicate flash type 'pT' or 'ph' to plot respectively
FLASH_TYPE: str = 'pT'
# Scaling of pressure and enthalpy from original units [Pa] and [J / mol]
P_FACTOR: float = 1e-3
P_UNIT: str = "[kPa]"
H_FACTOR: float = 1e-3
H_UNIT: str = "[J / mol]"
# flag to scale the pressure logarithmically in the plots
P_LOG_SCALE: bool = True
# image size information
FIG_WIDTH = 15  # in inches, 1080 / 1920 ratio applied to height
DPI: int = 500

# do not change this
if FLASH_TYPE == 'pT':
    H_FACTOR = 1


def get_px_index_map(p: list[Any], x: list[Any]) -> tuple[np.ndarray, np.ndarray, dict[tuple[Any, Any], int]]:
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
    return p_arr, x_arr, dict([((p_, x_), i) for p_, x_, i in zip(p_vec, x_vec, np.arange(p_vec.shape[0]))])


def _plot_overshoot(
    axis: plt.Axes, vals: np.ndarray, T: np.ndarray, p: np.ndarray, name: str
):
    over_shoot = vals > 1
    under_shoot = vals < 0
    legend_img = []
    legend_txt = []
    if np.any(under_shoot):
        img_u = axis.plot(
            T[under_shoot].flat, p[under_shoot].flat, "v", markersize=3, color="black"
        )
        legend_img.append(img_u[0])
        legend_txt.append(f"{name} < 0")
    if np.any(over_shoot):
        img_o = axis.plot(
            T[over_shoot].flat, p[over_shoot].flat, "^", markersize=3, color="red"
        )
        legend_img.append(img_o[0])
        legend_txt.append(f"{name} > 1")
    return legend_img, legend_txt


def _plot_crit_point(axis):
    """Plot critical pressure and temperature in p-T plot for components H2O and CO2."""

    pc_co2 = 7376460 * P_FACTOR
    Tc_co2 = 304.2
    pc_h2o = 22048320 * P_FACTOR
    Tc_h2o = 647.14

    img_h2o = axis.plot(Tc_h2o, pc_h2o, "*", markersize=10, color="blue")
    img_co2 = axis.plot(Tc_co2, pc_co2, "*", markersize=10, color="black")

    return [img_h2o[0], img_co2[0]], ["H2O Crit. Point", "CO2 Crit. Point"]


def _plot_liquid_phase_splits(
        axis: plt.Axes, p_mesh: np.ndarray, x_mesh: np.ndarray, vals: list[Any],
        px_map: dict
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

            if 'GLL' in v:
                GLL_mesh[i, j] = 1
            if 'LL' in v and not 'G' in v:
                LL_mesh[i, j] = 1

    LL_split = LL_mesh == 1
    GLL_split = GLL_mesh == 1

    legend_img = []
    legend_txt = []
    if np.any(LL_split):
        img_u = axis.plot(
            x_mesh[LL_split].flat * H_FACTOR, p_mesh[LL_split].flat * P_FACTOR,
            "+", markersize=3, color="red"
        )
        legend_img.append(img_u[0])
        legend_txt.append(f"LL split")
    if np.any(GLL_split):
        img_o = axis.plot(
            x_mesh[GLL_split].flat * H_FACTOR, p_mesh[GLL_split].flat * P_FACTOR,
            "P", markersize=3, color="red"
        )
        legend_img.append(img_o[0])
        legend_txt.append(f"GLL split")
    return legend_img, legend_txt


def plot_success(
        axis: plt.Axes, p_mesh: np.ndarray, x_mesh: np.ndarray, vals: list[Any],
        x_name: str, px_map: dict
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

            v = vals[px_map[px]]

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
        x_mesh * H_FACTOR, p_mesh * P_FACTOR, v_mesh,
        cmap=cmap, vmin=0, vmax=2, shading="nearest"
    )
    if 'T' in x_name:
        img_c, leg_c = _plot_crit_point(axis)
        axis.legend(img_c, leg_c, loc="upper left")
    axis.set_title(f"Success rate: {(success_rate) / (p_mesh.shape[0] * p_mesh.shape[1]) * 100} %")
    axis.set_xlabel(x_name)
    axis.set_ylabel(f"p {P_UNIT}")
    if P_LOG_SCALE:
        axis.set_yscale("log")

    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(img, cax=cax, orientation="vertical")
    cb.set_ticks([0, 1, 2])
    cb.set_ticklabels(["missing/nan", "failed", "succeeded"])


def plot_phase_regions(
        axis: plt.Axes, p_mesh: np.ndarray, x_mesh: np.ndarray, vals: list[Any],
        x_name: str, px_map: dict
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
            elif 'L' in v and not 'G' in v:
                v_mesh[i, j] = 1
            elif 'GL' in v:
                v_mesh[i, j] = 3

    cmap = mpl.colors.ListedColormap(["white", "blue", "red", "yellow"])
    img = axis.pcolormesh(
        x_mesh * H_FACTOR, p_mesh * P_FACTOR, v_mesh,
        cmap=cmap, vmin=0, vmax=3, shading="nearest"
    )
    if 'T' in x_name:
        img_c, leg_c = _plot_crit_point(axis)
    else:
        img_c = list()
        leg_c = list()
    img_ps, leg_ps = _plot_liquid_phase_splits(axis, p_mesh, x_mesh, vals, px_map)
    axis.legend(img_c + img_ps, leg_c + leg_ps, loc="upper left")
    axis.set_title("Phase regions")
    axis.set_xlabel(x_name)
    axis.set_ylabel(f"p {P_UNIT}")
    if P_LOG_SCALE:
        axis.set_yscale("log")

    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(img, cax=cax, orientation="vertical")
    cb.set_ticks([0, 1, 2, 3])
    cb.set_ticklabels(["N/A", "liquid", "gas", "2-phase"])


# region Plot 2: Absolute error in gas fraction per pT point
y_err_mesh = np.zeros((nx, ny))
y_mesh = np.zeros((nx, ny))

for i in range(nx):
    for j in range(ny):

        p = p_mesh[i, j]
        T = T_mesh[i, j]

        pT = (p, T)

        # If data for point is available
        if pT in pT_id:
            results = result_data[pT]
            identifier = pT_id[pT]

            y_result = results["y"]
            y_thermo = thermo_data[identifier[1]][pT]["y"]

            abs_err = np.abs(y_result - y_thermo)

            y_err_mesh[i, j] = abs_err
            y_mesh[i, j] = y_result

# filter out nans where the flash failed
y_err_mesh[np.isnan(y_err_mesh)] = 0.0
y_mesh[np.isnan(y_mesh)] = 0.0

figwidth = 15
fig = plt.figure(figsize=(figwidth, 1080 / 1920 * figwidth))
gs = fig.add_gridspec(1, 2)
fig.suptitle(f"Gas fraction values and absolute error: {version}")

vmin, vmax = y_mesh.min(), y_mesh.max()

# num_levels = 20
# midpoint = 0
# levels = np.hstack([np.linspace(vmin, 0.25, num_levels), np.linspace(0.25, vmax, 10, )])
# levels = np.sort(np.unique(levels))
# midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
# vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
# colors = mcm.get_cmap('coolwarm_r')(vals)  # RdYlGn
# cmap, norm = from_levels_and_colors(levels, colors)
cmap = "coolwarm"
linthresh = 1e-3
norm = mpl.colors.SymLogNorm(linthresh=linthresh, linscale=0.5, vmin=vmin, vmax=vmax)
# norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

axis = fig.add_subplot(gs[0, 0])
img = axis.pcolormesh(
    T_mesh,
    p_mesh_f,
    y_mesh,
    cmap=cmap,
    norm=norm,
    # vmin=y_mesh.min(), vmax=y_mesh.max(),
    shading="nearest",
)
# plot over and undershooting
img_uo, leg_uo = _plot_overshoot(axis, y_mesh, T_mesh, p_mesh_f, "y")
img_c, leg_c = _plot_crit_point(axis)
axis.legend(img_uo + img_c, leg_uo + leg_c, loc="upper left")
axis.set_title("Gas fraction: values")
axis.set_xlabel("T")
axis.set_ylabel(f"p {P_UNIT}")
if P_LOG_SCALE:
    axis.set_yscale("log")
divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="5%", pad=0.1)
ticks = np.linspace(0, 1, 6)
ticks = np.hstack([np.array([vmin]), ticks, np.array([vmax])])
cb = fig.colorbar(
    img,
    cax=cax,
    orientation="vertical",
    ticks=ticks,
)
cb.set_label(f"Min.: {vmin}\nMax.: {vmax}")

axis = fig.add_subplot(gs[0, 1])
img = axis.pcolormesh(
    T_mesh,
    p_mesh_f,
    y_err_mesh,
    cmap="Greys",
    vmin=y_err_mesh.min(),
    vmax=y_err_mesh.max(),
    shading="nearest",
)
img_c, leg_c = _plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Absolute error: Gas fraction")
axis.set_xlabel("T")
axis.set_ylabel(f"p {P_UNIT}")
if P_LOG_SCALE:
    axis.set_yscale("log")
divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="5%", pad=0.1)
cb = fig.colorbar(img, cax=cax, orientation="vertical")
cb.set_label(
    "Max abs. error: "
    + "{:.0e}".format(float(y_err_mesh.max()))
    + "\nL2-error: "
    + "{:.0e}".format(float(np.sqrt(np.sum(np.square(y_err_mesh)))))
)

fig.tight_layout()
fig.savefig(
    f"{str(path)}/{figure_path}2_gas_fraction__{result_fnam_stripped}.png",
    format="png",
    dpi=500,
)
fig.show()

# endregion

# region Plot 3: Absolute error in Liquid phase
x_h2o_L_err_mesh = np.zeros((nx, ny))
x_h2o_L_mesh = np.zeros((nx, ny))
x_co2_L_err_mesh = np.zeros((nx, ny))
x_co2_L_mesh = np.zeros((nx, ny))

for i in range(nx):
    for j in range(ny):

        p = p_mesh[i, j]
        T = T_mesh[i, j]

        pT = (p, T)

        # If data for point is available
        if pT in pT_id:
            results = result_data[pT]
            identifier = pT_id[pT]

            x_h2o_L = results["x_h2o_L"]
            x_co2_L = results["x_co2_L"]

            x_h2o_L_mesh[i, j] = x_h2o_L
            x_co2_L_mesh[i, j] = x_co2_L

            # If thermo data contains Liquid or Gas, calculate error
            # leave zero otherwise
            if "L" in identifier[0]:
                x_h2o_L_thermo = thermo_data[identifier[1]][pT]["x_h2o_L"]
                x_co2_L_thermo = thermo_data[identifier[1]][pT]["x_co2_L"]

                x_h2o_L_err_mesh[i, j] = np.abs(x_h2o_L - x_h2o_L_thermo)
                x_co2_L_err_mesh[i, j] = np.abs(x_co2_L - x_co2_L_thermo)


# filter out nans where the flash failed
x_h2o_L_err_mesh[np.isnan(x_h2o_L_err_mesh)] = 0.0
x_h2o_L_mesh[np.isnan(x_h2o_L_err_mesh)] = 0.0
x_co2_L_err_mesh[np.isnan(x_co2_L_err_mesh)] = 0.0
x_co2_L_mesh[np.isnan(x_co2_L_err_mesh)] = 0.0

figwidth = 15
fig = plt.figure(figsize=(figwidth, 1080 / 1920 * figwidth))
gs = fig.add_gridspec(2, 2)
fig.suptitle(f"Liquid phase composition and absolute error: {version}")

vmin = x_h2o_L_mesh.min()
vmax = x_h2o_L_mesh.max()
linthresh = 1e-3
# norm = mpl.colors.SymLogNorm(linthresh=linthresh, linscale=0.5, vmin=vmin, vmax=vmax)
# norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
axis = fig.add_subplot(gs[0, 0])
img = axis.pcolormesh(
    T_mesh,
    p_mesh_f,
    x_h2o_L_mesh,
    cmap="coolwarm",  # norm=norm,
    vmin=vmin,
    vmax=vmax,
    shading="nearest",
)
# plot over and undershooting
img_uo, leg_uo = _plot_overshoot(axis, x_h2o_L_mesh, T_mesh, p_mesh_f, "x_h2o_L")
img_c, leg_c = _plot_crit_point(axis)
axis.legend(img_uo + img_c, leg_uo + leg_c, loc="upper left")
axis.set_title("Fraction H2O in Liquid: values")
axis.set_xlabel("T")
axis.set_ylabel(f"p {P_UNIT}")
if P_LOG_SCALE:
    axis.set_yscale("log")
divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="5%", pad=0.1)
ticks = np.linspace(0, 1, 6)
ticks = np.hstack([np.array([vmin]), ticks, np.array([vmax])])
cb = fig.colorbar(
    img,
    cax=cax,
    orientation="vertical",
    ticks=ticks,
)
cb.set_label(f"Min.: {vmin}\nMax.: {vmax}")

vmin = x_co2_L_mesh.min()
vmax = x_co2_L_mesh.max()
linthresh = 1e-3
# norm = mpl.colors.SymLogNorm(linthresh=linthresh, linscale=0.5, vmin=vmin, vmax=vmax)
# norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
axis = fig.add_subplot(gs[0, 1])
img = axis.pcolormesh(
    T_mesh,
    p_mesh_f,
    x_co2_L_mesh,
    cmap="coolwarm",  # norm=norm,
    vmin=vmin,
    vmax=vmax,
    shading="nearest",
)
# plot over and undershooting
img_uo, leg_uo = _plot_overshoot(axis, x_co2_L_mesh, T_mesh, p_mesh_f, "x_co2_L")
img_c, leg_c = _plot_crit_point(axis)
axis.legend(img_uo + img_c, leg_uo + leg_c, loc="upper left")
axis.set_title("Fraction CO2 in Liquid: values")
axis.set_xlabel("T")
axis.set_ylabel(f"p {P_UNIT}")
if P_LOG_SCALE:
    axis.set_yscale("log")
divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="5%", pad=0.1)
ticks = np.linspace(0, 1, 6)
ticks = np.hstack([np.array([vmin]), ticks, np.array([vmax])])
cb = fig.colorbar(
    img,
    cax=cax,
    orientation="vertical",
    ticks=ticks,
)
cb.set_label(f"Min.: {vmin}\nMax.: {vmax}")

axis = fig.add_subplot(gs[1, 0])
img = axis.pcolormesh(
    T_mesh,
    p_mesh_f,
    x_h2o_L_err_mesh,
    cmap="Greys",
    vmin=x_h2o_L_err_mesh.min(),
    vmax=x_h2o_L_err_mesh.max(),
    shading="nearest",
)
img_c, leg_c = _plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Absolute error: H2O fraction in Liquid")
axis.set_xlabel("T")
axis.set_ylabel(f"p {P_UNIT}")
if P_LOG_SCALE:
    axis.set_yscale("log")
divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="5%", pad=0.1)
cb = fig.colorbar(img, cax=cax, orientation="vertical")
cb.set_label(
    "Max abs. error: "
    + "{:.0e}".format(float(x_h2o_L_err_mesh.max()))
    + "\nL2-error: "
    + "{:.0e}".format(float(np.sqrt(np.sum(np.square(x_h2o_L_err_mesh)))))
)

axis = fig.add_subplot(gs[1, 1])
img = axis.pcolormesh(
    T_mesh,
    p_mesh_f,
    x_co2_L_err_mesh,
    cmap="Greys",
    vmin=x_co2_L_err_mesh.min(),
    vmax=x_co2_L_err_mesh.max(),
    shading="nearest",
)
img_c, leg_c = _plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Absolute error: CO2 fraction in Liquid")
axis.set_xlabel("T")
axis.set_ylabel(f"p {P_UNIT}")
if P_LOG_SCALE:
    axis.set_yscale("log")
divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="5%", pad=0.1)
cb = fig.colorbar(img, cax=cax, orientation="vertical")
cb.set_label(
    "Max abs. error: "
    + "{:.0e}".format(float(x_co2_L_err_mesh.max()))
    + "\nL2-error: "
    + "{:.0e}".format(float(np.sqrt(np.sum(np.square(x_co2_L_err_mesh)))))
)

fig.tight_layout()
fig.savefig(
    f"{str(path)}/{figure_path}3_liquid_composition_error__{result_fnam_stripped}.png",
    format="png",
    dpi=500,
)
fig.show()
# endregion

# region Plot4: Absolute error in Gas phase
x_h2o_G_err_mesh = np.zeros((nx, ny))
x_h2o_G_mesh = np.zeros((nx, ny))
x_co2_G_err_mesh = np.zeros((nx, ny))
x_co2_G_mesh = np.zeros((nx, ny))

for i in range(nx):
    for j in range(ny):

        p = p_mesh[i, j]
        T = T_mesh[i, j]

        pT = (p, T)

        # If data for point is available
        if pT in pT_id:
            results = result_data[pT]
            identifier = pT_id[pT]

            x_h2o_G = results["x_h2o_G"]
            x_co2_G = results["x_co2_G"]

            x_h2o_G_mesh[i, j] = x_h2o_G
            x_co2_G_mesh[i, j] = x_co2_G

            # If thermo data contains Liquid or Gas, calculate error
            # leave zero otherwise
            if "G" in identifier[0]:
                x_h2o_G_thermo = thermo_data[identifier[1]][pT]["x_h2o_G"]
                x_co2_G_thermo = thermo_data[identifier[1]][pT]["x_co2_G"]

                x_h2o_G_err_mesh[i, j] = np.abs(x_h2o_G - x_h2o_G_thermo)
                x_co2_G_err_mesh[i, j] = np.abs(x_co2_G - x_co2_G_thermo)


# filter out nans where the flash failed
x_h2o_G_err_mesh[np.isnan(x_h2o_G_err_mesh)] = 0.0
x_h2o_G_mesh[np.isnan(x_h2o_G_err_mesh)] = 0.0
x_co2_G_err_mesh[np.isnan(x_co2_G_err_mesh)] = 0.0
x_co2_G_mesh[np.isnan(x_co2_G_err_mesh)] = 0.0

figwidth = 15
fig = plt.figure(figsize=(figwidth, 1080 / 1920 * figwidth))
gs = fig.add_gridspec(2, 2)
fig.suptitle(f"Gas phase composition and absolute error: {version}")

vmin = x_h2o_G_mesh.min()
vmax = x_h2o_G_mesh.max()
linthresh = 1e-3
# norm = mpl.colors.SymLogNorm(linthresh=linthresh, linscale=0.5, vmin=vmin, vmax=vmax)
# norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
axis = fig.add_subplot(gs[0, 0])
img = axis.pcolormesh(
    T_mesh,
    p_mesh_f,
    x_h2o_G_mesh,
    cmap="coolwarm",  # norm=norm,
    vmin=vmin,
    vmax=vmax,
    shading="nearest",
)
# plot over and undershooting
img_uo, leg_uo = _plot_overshoot(axis, x_h2o_G_mesh, T_mesh, p_mesh_f, "x_h2o_L")
img_c, leg_c = _plot_crit_point(axis)
axis.legend(img_uo + img_c, leg_uo + leg_c, loc="upper left")
axis.set_title("Fraction H2O in Gas: values")
axis.set_xlabel("T")
axis.set_ylabel(f"p {P_UNIT}")
if P_LOG_SCALE:
    axis.set_yscale("log")
divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="5%", pad=0.1)
ticks = np.linspace(0, 1, 6)
ticks = np.hstack([np.array([vmin]), ticks, np.array([vmax])])
cb = fig.colorbar(
    img,
    cax=cax,
    orientation="vertical",
    ticks=ticks,
)
cb.set_label(f"Min.: {vmin}\nMax.: {vmax}")

vmin = x_co2_G_mesh.min()
vmax = x_co2_G_mesh.max()
linthresh = 1e-3
# norm = mpl.colors.SymLogNorm(linthresh=linthresh, linscale=0.5, vmin=vmin, vmax=vmax)
# norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
axis = fig.add_subplot(gs[0, 1])
img = axis.pcolormesh(
    T_mesh,
    p_mesh_f,
    x_co2_G_mesh,
    cmap="coolwarm",  # norm=norm,
    vmin=vmin,
    vmax=vmax,
    shading="nearest",
)
# plot over and undershooting
img_uo, leg_uo = _plot_overshoot(axis, x_co2_G_mesh, T_mesh, p_mesh_f, "x_co2_L")
img_c, leg_c = _plot_crit_point(axis)
axis.legend(img_uo + img_c, leg_uo + leg_c, loc="upper left")
axis.set_title("Fraction CO2 in Gas: values")
axis.set_xlabel("T")
axis.set_ylabel(f"p {P_UNIT}")
if P_LOG_SCALE:
    axis.set_yscale("log")
divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="5%", pad=0.1)
ticks = np.linspace(0, 1, 6)
ticks = np.hstack([np.array([vmin]), ticks, np.array([vmax])])
cb = fig.colorbar(
    img,
    cax=cax,
    orientation="vertical",
    ticks=ticks,
)
cb.set_label(f"Min.: {vmin}\nMax.: {vmax}")

axis = fig.add_subplot(gs[1, 0])
img = axis.pcolormesh(
    T_mesh,
    p_mesh_f,
    x_h2o_G_err_mesh,
    cmap="Greys",
    vmin=x_h2o_G_err_mesh.min(),
    vmax=x_h2o_G_err_mesh.max(),
    shading="nearest",
)
img_c, leg_c = _plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Absolute error: H2O fraction in Gas")
axis.set_xlabel("T")
axis.set_ylabel(f"p {P_UNIT}")
if P_LOG_SCALE:
    axis.set_yscale("log")
divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="5%", pad=0.1)
cb = fig.colorbar(img, cax=cax, orientation="vertical")
cb.set_label(
    "Max abs. error: "
    + "{:.0e}".format(float(x_h2o_G_err_mesh.max()))
    + "\nL2-error: "
    + "{:.0e}".format(float(np.sqrt(np.sum(np.square(x_h2o_G_err_mesh)))))
)

axis = fig.add_subplot(gs[1, 1])
img = axis.pcolormesh(
    T_mesh,
    p_mesh_f,
    x_co2_G_err_mesh,
    cmap="Greys",
    vmin=x_co2_G_err_mesh.min(),
    vmax=x_co2_G_err_mesh.max(),
    shading="nearest",
)
img_c, leg_c = _plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Absolute error: CO2 fraction in Gas")
axis.set_xlabel("T")
axis.set_ylabel(f"p {P_UNIT}")
if P_LOG_SCALE:
    axis.set_yscale("log")
divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="5%", pad=0.1)
cb = fig.colorbar(img, cax=cax, orientation="vertical")
cb.set_label(
    "Max abs. error: "
    + "{:.0e}".format(float(x_co2_G_err_mesh.max()))
    + "\nL2-error: "
    + "{:.0e}".format(float(np.sqrt(np.sum(np.square(x_co2_G_err_mesh)))))
)

fig.tight_layout()
fig.savefig(
    f"{str(path)}/{figure_path}4_gas_composition_error__{result_fnam_stripped}.png",
    format="png",
    dpi=500,
)
fig.show()
# endregion

# region Plot 5: Duality gap
dual_gap_L_mesh = np.zeros((nx, ny))
dual_gap_G_mesh = np.zeros((nx, ny))

for i in range(nx):
    for j in range(ny):

        p = p_mesh[i, j]
        T = T_mesh[i, j]

        pT = (p, T)

        # If data for point is available
        if pT in pT_id:
            results = result_data[pT]
            identifier = pT_id[pT]

            x_h2o_L = results["x_h2o_L"]
            x_co2_L = results["x_co2_L"]
            x_h2o_G = results["x_h2o_G"]
            x_co2_G = results["x_co2_G"]

            dual_gap_L_mesh[i, j] = 1 - x_h2o_L - x_co2_L
            dual_gap_G_mesh[i, j] = 1 - x_h2o_G - x_co2_G

dual_gap_L_mesh[np.isnan(dual_gap_L_mesh)] = 0.0
dual_gap_G_mesh[np.isnan(dual_gap_G_mesh)] = 0.0

figwidth = 15
fig = plt.figure(figsize=(figwidth, 1080 / 1920 * figwidth))
gs = fig.add_gridspec(1, 2)
fig.suptitle(f"Duality gaps: {version}")

axis = fig.add_subplot(gs[0, 0])
img = axis.pcolormesh(
    T_mesh,
    p_mesh_f,
    dual_gap_L_mesh,
    cmap="Greys",
    vmin=dual_gap_L_mesh.min(),
    vmax=dual_gap_L_mesh.max(),
    shading="nearest",
)
img_c, leg_c = _plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Duality Gap: Liquid")
axis.set_xlabel("T")
axis.set_ylabel(f"p {P_UNIT}")
if P_LOG_SCALE:
    axis.set_yscale("log")
divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="5%", pad=0.1)
cb = fig.colorbar(img, cax=cax, orientation="vertical")

axis = fig.add_subplot(gs[0, 1])
img = axis.pcolormesh(
    T_mesh,
    p_mesh_f,
    dual_gap_G_mesh,
    cmap="Greys",
    vmin=dual_gap_G_mesh.min(),
    vmax=dual_gap_G_mesh.max(),
    shading="nearest",
)
img_c, leg_c = _plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Duality Gap: Gas")
axis.set_xlabel("T")
axis.set_ylabel(f"p {P_UNIT}")
if P_LOG_SCALE:
    axis.set_yscale("log")
divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="5%", pad=0.1)
cb = fig.colorbar(img, cax=cax, orientation="vertical")

fig.tight_layout()
fig.savefig(
    f"{str(path)}/{figure_path}5_duality_gap__{result_fnam_stripped}.png",
    format="png",
    dpi=500,
)
fig.show()

# endregion

# region Plot 6: Condition numbers and numbers of iterations
cond_start_mesh = np.zeros((nx, ny))
cond_end_mesh = np.zeros((nx, ny))
num_iter_mesh = np.zeros((nx, ny))

for i in range(nx):
    for j in range(ny):

        p = p_mesh[i, j]
        T = T_mesh[i, j]

        pT = (p, T)

        # If data for point is available
        if pT in pT_id:
            results = result_data[pT]
            identifier = pT_id[pT]

            num_iter = results["num_iter"]
            cond_start = results["cond_start"]
            cond_end = results["cond_end"]

            cond_start_mesh[i, j] = cond_start
            cond_end_mesh[i, j] = cond_end
            num_iter_mesh[i, j] = num_iter

# i,j = np.unravel_index(cond_start_mesh.argmax(), cond_start_mesh.shape)
# print(f"COND START MAX AT: p={p_mesh[i,j]} ; T={T_mesh[i,j]}\n\tVal: {np.max(cond_start_mesh)}")
# i,j = np.unravel_index(cond_end_mesh.argmax(), cond_end_mesh.shape)
# print(f"COND CONVERGED MAX AT: p={p_mesh[i,j]} ; T={T_mesh[i,j]}\n\tVal: {np.max(cond_end_mesh)}")

figwidth = 15
fig = plt.figure(figsize=(figwidth, 1080 / 1920 * figwidth))
gs = fig.add_gridspec(2, 2)
fig.suptitle(f"Condition and iteration numbers: {version}")

vmin = cond_start_mesh.min()
vmax = cond_start_mesh.max()
linthresh = 1e-3
norm = mpl.colors.SymLogNorm(linthresh=linthresh, linscale=0.5, vmin=vmin, vmax=vmax)
# norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
axis = fig.add_subplot(gs[0, 0])
img = axis.pcolormesh(
    T_mesh,
    p_mesh_f,
    cond_start_mesh,
    cmap="Greys",
    norm=norm,
    # vmin=vmin,
    # vmax=vmax,
    shading="nearest",
)
# plot over and undershooting
img_c, leg_c = _plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Condition number: Start of iterations")
axis.set_xlabel("T")
axis.set_ylabel(f"p {P_UNIT}")
if P_LOG_SCALE:
    axis.set_yscale("log")
divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="5%", pad=0.1)
# ticks = np.linspace(0, 1, 6)
# ticks = np.hstack([np.array([vmin]), ticks, np.array([vmax])])
cb = fig.colorbar(
    img,
    cax=cax,
    orientation="vertical",
    # ticks=ticks,
)
cb.set_label(
    f"Max.: {'{:.4e}'.format(vmax)}\nMean: {'{:.4e}'.format(np.mean(cond_start_mesh))}"
)

vmin = cond_end_mesh.min()
vmax = cond_end_mesh.max()
linthresh = 1e-3
norm = mpl.colors.SymLogNorm(linthresh=linthresh, linscale=0.5, vmin=vmin, vmax=vmax)
axis = fig.add_subplot(gs[0, 1])
img = axis.pcolormesh(
    T_mesh,
    p_mesh_f,
    cond_end_mesh,
    cmap="Greys",
    norm=norm,
    # vmin=vmin,
    # vmax=vmax,
    shading="nearest",
)
# plot over and undershooting
img_c, leg_c = _plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Condition number: At converged state")
axis.set_xlabel("T")
axis.set_ylabel(f"p {P_UNIT}")
if P_LOG_SCALE:
    axis.set_yscale("log")
divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="5%", pad=0.1)
# ticks = np.linspace(0, 1, 6)
# ticks = np.hstack([np.array([vmin]), ticks, np.array([vmax])])
cb = fig.colorbar(
    img,
    cax=cax,
    orientation="vertical",
    # ticks=ticks,
)
cb.set_label(
    f"Max.: {'{:.4e}'.format(vmax)}\nMean: {'{:.4e}'.format(np.mean(cond_end_mesh))}"
)

vmin = num_iter_mesh.min()
vmax = num_iter_mesh.max()
linthresh = 1e-3
# norm = mpl.colors.SymLogNorm(linthresh=linthresh, linscale=0.5, vmin=vmin, vmax=vmax)
axis = fig.add_subplot(gs[1, 0])
img = axis.pcolormesh(
    T_mesh,
    p_mesh_f,
    num_iter_mesh,
    cmap="Greys",  # norm=norm,
    vmin=vmin,
    vmax=vmax,
    shading="nearest",
)
# plot over and undershooting
img_c, leg_c = _plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Number of iterations")
axis.set_xlabel("T")
axis.set_ylabel(f"p {P_UNIT}")
if P_LOG_SCALE:
    axis.set_yscale("log")
divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="5%", pad=0.1)
# ticks = np.linspace(0, 1, 6)
# ticks = np.hstack([np.array([vmin]), ticks, np.array([vmax])])
cb = fig.colorbar(
    img,
    cax=cax,
    orientation="vertical",
    # ticks=ticks,
)
cb.set_label(f"Max.: {vmax}\nMean: {np.mean(num_iter_mesh)}")

fig.tight_layout()
fig.savefig(
    f"{str(path)}/{figure_path}6_cond_iter_nums__{result_fnam_stripped}.png",
    format="png",
    dpi=500,
)
fig.show()

# endregion

if __name__ == '__main__':
    figwidth = 15
    scipt_path = path()
    fname = RESULT_FILE[
        RESULT_FILE.rfind("/") + 1 : RESULT_FILE.rfind(".csv")
    ]

    headers = get_result_headers()
    results = read_results([RESULT_FILE], headers)

    thermo_headers = read_headers(THERMO_FILE)
    thermo_results = read_results([THERMO_FILE], thermo_headers)

    p_points = results[p_HEADER]
    if FLASH_TYPE == 'pT':
        x_name = 'T [K]'
        x_points = results[T_HEADER]
    elif FLASH_TYPE == 'ph':
        x_name = f'h {H_UNIT}'
        x_points = results[h_HEADER]
    else:
        raise NotImplementedError(f"Unsupported flash type: {FLASH_TYPE}")

    p_vec, x_vec, px_map = get_px_index_map(p_points, x_points)

    x_mesh, p_mesh = np.meshgrid(x_vec, p_vec)
    p_mesh_f = p_mesh * P_FACTOR
    if FLASH_TYPE == 'pT':
        x_mesh_f = x_mesh
    elif FLASH_TYPE == 'ph':
        x_mesh_f = x_mesh * H_FACTOR

    # Plot 1: success rate and phase regions
    
    fig = plt.figure(figsize=(FIG_WIDTH, 1080 / 1920 * FIG_WIDTH))
    gs = fig.add_gridspec(1, 2)
    fig.suptitle(f"Overview: VLE with H2O and CO2")
    axis = fig.add_subplot(gs[0, 0])
    plot_phase_regions(axis, p_mesh, x_mesh, thermo_results[phases_HEADER], x_name, px_map)

    axis = fig.add_subplot(gs[0, 1])
    plot_success(axis, p_mesh, x_mesh, results[success_HEADER], x_name, px_map)


    fig.tight_layout()
    fig.savefig(
        f"{str(scipt_path)}/{FIGURE_PATH}1_overview__{fname}.png",
        format="png",
        dpi=DPI,
    )
