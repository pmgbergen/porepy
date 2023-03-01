"""Script for plotting various figures for the comparison of the pT flash (H2O, CO2)
with thermo data.

This script follows the patterns introduced in ``calc_flash_h2o_co2.py`` and can
be performed on the files produced by it.

"""
import csv
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

### General settings

# files containing thermo data
# list of (filename, mode),
# where mode indicates if only L (liquid), G (gas) or both (GL) data are available
thermo_files = [
    ("data/pr_data_thermo_isothermal_G_easy.csv", "G"),
    ("data/pr_data_thermo_isothermal_G_hard.csv", "G"),
    ("data/pr_data_thermo_isothermal_L_easy.csv", "L"),
    ("data/pr_data_thermo_isothermal_L_hard.csv", "L"),
    ("data/pr_data_thermo_isothermal_GL_easy.csv", "GL"),
    ("data/pr_data_thermo_isothermal_GL_hard.csv", "GL"),
]
# Files containing results and identification for p T values
version = "w-o-reg-par-cond"
results_file = f"data/results/pr_result_VL_{version}.csv"
identifier_file = f"data/results/pr_result_VL_{version}_ID.csv"
figure_path = "data/results/figures/"  # path to where figures should be stored
path = pathlib.Path(__file__).parent.resolve()  # path to script for file i/o

# Thermo data is provided in Pascal for pressure
# Scale to Mpa or kPa here
# Flag if log scale should be applied to pressure
p_factor = 1e-3
p_unit = "[kPa]"
p_scale_log = True

# thermo data storage
# file name -> dictionary containing per p-T point another dictionary with floats per data name
thermo_data: dict[str, dict[tuple[float, float], dict[str, float]]] = dict()

# lists containing pressure and Temperature data for test cases
p_points: list[float] = list()
T_points: list[float] = list()
# to identify file and row per pT point
# (p, T) -> (mode, file name, row id)
pT_id: dict[tuple[float, float], tuple[str, str, int]] = dict()

result_fnam_stripped = results_file[
    results_file.rfind("/") + 1 : results_file.rfind(".csv")
]

print("Reading thermo data ...", flush=True)
for filename, mode in thermo_files:
    with open(f"{path}/{filename}") as file:
        file_reader = csv.reader(file, delimiter=",")
        header = next(file_reader)

        # get column indices in csv file
        row_id_idx = header.index("id")
        p_idx = header.index("p [Pa]")
        T_idx = header.index("T [K]")
        y_idx = header.index("y")

        # get gas data indices if available
        if "G" in mode:
            x_h2o_G_idx = header.index("x_h2o_G")
            x_co2_G_idx = header.index("x_co2_G")
            Z_G_idx = header.index("Z_G")
        else:
            x_h2o_G_idx = None
            x_co2_G_idx = None
            Z_G_idx = None
        # get liquid data indices if available
        if "L" in mode:
            x_h2o_L_idx = header.index("x_h2o_L")
            x_co2_L_idx = header.index("x_co2_L")
            Z_L_idx = header.index("Z_L")
        else:
            x_h2o_L_idx = None
            x_co2_L_idx = None
            Z_L_idx = None

        # prepare storage
        thermo_data[filename] = dict()

        for datarow in file_reader:

            row_id = int(datarow[0])
            p = float(datarow[p_idx])
            T = float(datarow[T_idx])

            pT = (p, T)
            # get only unique points
            if pT not in pT_id:
                identifier = (mode, filename, row_id)
                pT_id.update({pT: identifier})

                p_points.append(p)
                T_points.append(T)

                # store data
                thermo_data[filename][pT] = dict()
                thermo_data[filename][pT].update({"y": float(datarow[y_idx])})
                if "G" in mode:
                    thermo_data[filename][pT].update(
                        {
                            "x_h2o_G": float(datarow[x_h2o_G_idx]),
                            "x_co2_G": float(datarow[x_co2_G_idx]),
                            "Z_G": float(datarow[Z_G_idx]),
                        }
                    )
                if "L" in mode:
                    thermo_data[filename][pT].update(
                        {
                            "x_h2o_L": float(datarow[x_h2o_L_idx]),
                            "x_co2_L": float(datarow[x_co2_L_idx]),
                            "Z_L": float(datarow[Z_L_idx]),
                        }
                    )

# storage for result data
# (p,T) -> dict containing values per data name
result_data: dict[tuple[float, float], dict[str, float]] = dict()

print("Reading result data ...", flush=True)
with open(f"{path}/{results_file}") as file:
    file_reader = csv.reader(file, delimiter=",")
    header = next(file_reader)

    # get column indices in csv file
    success_idx = header.index("success")
    p_idx = header.index("p [Pa]")
    T_idx = header.index("T [K]")
    y_idx = header.index("y")
    x_h2o_G_idx = header.index("x_h2o_G")
    x_co2_G_idx = header.index("x_co2_G")
    Z_G_idx = header.index("Z_G")
    x_h2o_L_idx = header.index("x_h2o_L")
    x_co2_L_idx = header.index("x_co2_L")
    Z_L_idx = header.index("Z_L")
    num_iter_idx = header.index("num_iter")
    cond_start_idx = header.index("cond_start")
    cond_end_idx = header.index("cond_end")

    for datarow in file_reader:
        p = float(datarow[p_idx])
        T = float(datarow[T_idx])
        success = int(datarow[success_idx])
        # success = 1
        y = float(datarow[y_idx])
        Z_L = float(datarow[Z_L_idx])
        y = float(datarow[y_idx])
        Z_G = float(datarow[Z_G_idx])
        x_h2o_L = float(datarow[x_h2o_L_idx])
        x_co2_L = float(datarow[x_co2_L_idx])
        x_h2o_G = float(datarow[x_h2o_G_idx])
        x_co2_G = float(datarow[x_co2_G_idx])
        num_iter = int(datarow[num_iter_idx])
        cond_start = float(datarow[cond_start_idx])
        cond_end = float(datarow[cond_end_idx])

        pT = (p, T)

        assert pT in pT_id, f"Point {pT} not in thermo data files."

        result_data[pT] = dict()
        result_data[pT].update(
            {
                "success": success,
                "p": p,
                "T": p,
                "y": y,
                "Z_L": Z_L,
                "Z_G": Z_G,
                "x_h2o_L": x_h2o_L,
                "x_co2_L": x_co2_L,
                "x_h2o_G": x_h2o_G,
                "x_co2_G": x_co2_G,
                "num_iter": num_iter,
                "cond_start": cond_start,
                "cond_end": cond_end,
            }
        )

### Helper functions


def plot_overshoot(axis, vals, T, p, name):
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


def plot_crit_point(axis):

    pc_co2 = 7376460 * p_factor
    Tc_co2 = 304.2
    pc_h2o = 22048320 * p_factor
    Tc_h2o = 647.14

    img_h2o = axis.plot(Tc_h2o, pc_h2o, "*", markersize=10, color="blue")
    img_co2 = axis.plot(Tc_co2, pc_co2, "*", markersize=10, color="black")

    return [img_h2o[0], img_co2[0]], ["H2O Crit. Point", "CO2 Crit. Point"]


### Creating various plots

print("Creating plots ...", flush=True)

# creating basic mesh grid from pT points
p_vec = np.sort(np.unique(np.array(p_points)))
T_vec = np.sort(np.unique(np.array(T_points)))
# Temperature as first axis, pressure as second axis
T_mesh, p_mesh = np.meshgrid(T_vec, p_vec)
p_mesh_f = p_mesh * p_factor
nx, ny = T_mesh.shape

# region Plot 1: success rate and phase regions
success_mesh = np.zeros((nx, ny))
region_mesh = np.zeros((nx, ny))

for i in range(nx):
    for j in range(ny):

        p = p_mesh[i, j]
        T = T_mesh[i, j]

        # If data for point is available
        if (p, T) in pT_id:
            results = result_data[(p, T)]
            identifier = pT_id[(p, T)]

            # value of 1 indicates failure, value of 2 indicates success
            success = results["success"] + 1
            # value map:
            # 1 -> liquid only
            # 2 -> gas only
            # 3 -> 2-phase
            if identifier[0] == "G":
                mode = 2
            elif identifier[0] == "L":
                mode = 1
            elif identifier[0] == "GL":
                mode = 3

            success_mesh[i, j] = success
            region_mesh[i, j] = mode

figwidth = 15
fig = plt.figure(figsize=(figwidth, 1080 / 1920 * figwidth))
gs = fig.add_gridspec(1, 2)
fig.suptitle(f"Phase regions and success rate: {version}")

# plotting phase regions
axis = fig.add_subplot(gs[0, 0])
cmap = mpl.colors.ListedColormap(["white", "blue", "yellow", "red"])
img = axis.pcolormesh(
    T_mesh, p_mesh_f, region_mesh, cmap=cmap, vmin=0, vmax=3, shading="nearest"
)
img_c, leg_c = plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Phase regions")
axis.set_xlabel("T")
axis.set_ylabel(f"p {p_unit}")
if p_scale_log:
    axis.set_yscale("log")

divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="5%", pad=0.1)
cb = fig.colorbar(img, cax=cax, orientation="vertical")
cb.set_ticks([0, 1, 2, 3])
cb.set_ticklabels(["no data", "liquid", "gas", "2-phase"])

# plotting success
axis = fig.add_subplot(gs[0, 1])
cmap = mpl.colors.ListedColormap(["white", "red", "green"])
img = axis.pcolormesh(
    T_mesh, p_mesh_f, success_mesh, cmap=cmap, vmin=0, vmax=2, shading="nearest"
)
axis.set_title("Flash success")
axis.set_xlabel("T")
axis.set_ylabel(f"p {p_unit}")
if p_scale_log:
    axis.set_yscale("log")

divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="5%", pad=0.1)
cb = fig.colorbar(img, cax=cax, orientation="vertical")
cb.set_ticks([1, 2])
cb.set_ticklabels(["failed", "succeeded"])

fig.tight_layout()
fig.savefig(
    f"{str(path)}/{figure_path}1_success_and_regions__{result_fnam_stripped}.png",
    format="png",
    dpi=500,
)
fig.show()

# endregion

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
img_uo, leg_uo = plot_overshoot(axis, y_mesh, T_mesh, p_mesh_f, "y")
img_c, leg_c = plot_crit_point(axis)
axis.legend(img_uo + img_c, leg_uo + leg_c, loc="upper left")
axis.set_title("Gas fraction: values")
axis.set_xlabel("T")
axis.set_ylabel(f"p {p_unit}")
if p_scale_log:
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
img_c, leg_c = plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Absolute error: Gas fraction")
axis.set_xlabel("T")
axis.set_ylabel(f"p {p_unit}")
if p_scale_log:
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
img_uo, leg_uo = plot_overshoot(axis, x_h2o_L_mesh, T_mesh, p_mesh_f, "x_h2o_L")
img_c, leg_c = plot_crit_point(axis)
axis.legend(img_uo + img_c, leg_uo + leg_c, loc="upper left")
axis.set_title("Fraction H2O in Liquid: values")
axis.set_xlabel("T")
axis.set_ylabel(f"p {p_unit}")
if p_scale_log:
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
img_uo, leg_uo = plot_overshoot(axis, x_co2_L_mesh, T_mesh, p_mesh_f, "x_co2_L")
img_c, leg_c = plot_crit_point(axis)
axis.legend(img_uo + img_c, leg_uo + leg_c, loc="upper left")
axis.set_title("Fraction CO2 in Liquid: values")
axis.set_xlabel("T")
axis.set_ylabel(f"p {p_unit}")
if p_scale_log:
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
img_c, leg_c = plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Absolute error: H2O fraction in Liquid")
axis.set_xlabel("T")
axis.set_ylabel(f"p {p_unit}")
if p_scale_log:
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
img_c, leg_c = plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Absolute error: CO2 fraction in Liquid")
axis.set_xlabel("T")
axis.set_ylabel(f"p {p_unit}")
if p_scale_log:
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
img_uo, leg_uo = plot_overshoot(axis, x_h2o_G_mesh, T_mesh, p_mesh_f, "x_h2o_L")
img_c, leg_c = plot_crit_point(axis)
axis.legend(img_uo + img_c, leg_uo + leg_c, loc="upper left")
axis.set_title("Fraction H2O in Gas: values")
axis.set_xlabel("T")
axis.set_ylabel(f"p {p_unit}")
if p_scale_log:
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
img_uo, leg_uo = plot_overshoot(axis, x_co2_G_mesh, T_mesh, p_mesh_f, "x_co2_L")
img_c, leg_c = plot_crit_point(axis)
axis.legend(img_uo + img_c, leg_uo + leg_c, loc="upper left")
axis.set_title("Fraction CO2 in Gas: values")
axis.set_xlabel("T")
axis.set_ylabel(f"p {p_unit}")
if p_scale_log:
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
img_c, leg_c = plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Absolute error: H2O fraction in Gas")
axis.set_xlabel("T")
axis.set_ylabel(f"p {p_unit}")
if p_scale_log:
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
img_c, leg_c = plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Absolute error: CO2 fraction in Gas")
axis.set_xlabel("T")
axis.set_ylabel(f"p {p_unit}")
if p_scale_log:
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
img_c, leg_c = plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Duality Gap: Liquid")
axis.set_xlabel("T")
axis.set_ylabel(f"p {p_unit}")
if p_scale_log:
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
img_c, leg_c = plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Duality Gap: Gas")
axis.set_xlabel("T")
axis.set_ylabel(f"p {p_unit}")
if p_scale_log:
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
img_c, leg_c = plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Condition number: Start of iterations")
axis.set_xlabel("T")
axis.set_ylabel(f"p {p_unit}")
if p_scale_log:
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
img_c, leg_c = plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Condition number: At converged state")
axis.set_xlabel("T")
axis.set_ylabel(f"p {p_unit}")
if p_scale_log:
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
img_c, leg_c = plot_crit_point(axis)
axis.legend(img_c, leg_c, loc="upper left")
axis.set_title("Number of iterations")
axis.set_xlabel("T")
axis.set_ylabel(f"p {p_unit}")
if p_scale_log:
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

print("Done")
