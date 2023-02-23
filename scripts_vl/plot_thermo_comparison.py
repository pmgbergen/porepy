"""Script for plotting various figures for the comparison of the pT flash (H2O, CO2)
with thermo data.

This script follows the patterns introduced in ``calc_flash_h2o_co2.py`` and can
be performed on the files produced by it.

"""
import csv
import pathlib

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

### General settings

# files containing thermo data
# list of (filename, mode),
# where mode indicates if only L (liquid), G (gas) or both (GL) data are available
thermo_files = [
    ('data/pr_data_thermo_isothermal_G_easy.csv', 'G'),
    ('data/pr_data_thermo_isothermal_G_hard.csv', 'G'),
    ('data/pr_data_thermo_isothermal_L_easy.csv', 'L'),
    ('data/pr_data_thermo_isothermal_L_hard.csv', 'L'),
    ('data/pr_data_thermo_isothermal_GL_easy.csv', 'GL'),
    ('data/pr_data_thermo_isothermal_GL_hard.csv', 'GL'),
]
# Files containing results and identification for p T values
results_file = 'data/results/pr_result_VL_wo-reg.csv'
identifier_file = 'data/results/pr_result_VL_wo-reg_ID.csv'
figure_path = 'data/results/'  # path to where figures should be stored
path = pathlib.Path(__file__).parent.resolve()  # path to script for file i/o


# thermo data storage
# file name -> dictionary containing per p-T point another dictionary with floats per data name
thermo_data: dict[str, dict[tuple[float, float], dict[str, float]]] = dict()

# lists containing pressure and Temperature data for test cases
p_points: list[float] = list()
T_points: list[float] = list()
# to identify file and row per pT point
# (p, T) -> (mode, file name, row id)
pT_id: dict[tuple[float, float], tuple[str, str, int]] = dict()

print("Reading thermo data ...", flush=True)
for filename, mode in thermo_files:
      with open(f"{path}/{filename}") as file:
        file_reader = csv.reader(file, delimiter=",")
        header = next(file_reader)

        # get column indices in csv file
        row_id_idx = header.index('id')
        p_idx = header.index('p [Pa]')
        T_idx = header.index('T [K]')
        y_idx = header.index('y')

        # get gas data indices if available
        if 'G' in mode:
            x_h2o_G_idx = header.index('x_h2o_G')
            x_co2_G_idx = header.index('x_co2_G')
            Z_G_idx = header.index('Z_G')
        else:
            x_h2o_G_idx = None
            x_co2_G_idx = None
            Z_G_idx = None
        # get liquid data indices if available
        if 'L' in mode:
            x_h2o_L_idx = header.index('x_h2o_L')
            x_co2_L_idx = header.index('x_co2_L')
            Z_L_idx = header.index('Z_L')
        else:
            x_h2o_L_idx = None
            x_co2_L_idx = None
            Z_L_idx = None

        # prepare storage
        thermo_data[filename] = dict()

        for datarow in file_reader:

            row_id = int(datarow[0])
            p = float(datarow[p_idx]) * 1e-6  # scale from Pa to MPa
            T = float(datarow[T_idx])

            pT = (p, T)
            # get only unique points
            if pT not in pT_id:
                identifier = (mode, filename, row_id)
                pT_id.update({
                    pT: identifier
                })

                p_points.append(p)
                T_points.append(T)

                # store data
                thermo_data[filename][pT] = dict()
                if 'G' in mode:
                    thermo_data[filename][pT].update(
                        {
                            'x_h2o_G': datarow[x_h2o_G_idx],
                            'x_co2_G': datarow[x_co2_G_idx],
                            'Z_G': datarow[Z_G_idx],
                        }
                    )
                if 'L' in mode:
                    thermo_data[filename][pT].update(
                        {
                            'x_h2o_L': datarow[x_h2o_L_idx],
                            'x_co2_L': datarow[x_co2_L_idx],
                            'Z_L': datarow[Z_L_idx],
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
    success_idx = header.index('success')
    p_idx = header.index('p [MPa]')
    T_idx = header.index('T [K]')
    y_idx = header.index('y')
    x_h2o_G_idx = header.index('x_h2o_G')
    x_co2_G_idx = header.index('x_co2_G')
    Z_G_idx = header.index('Z_G')
    x_h2o_L_idx = header.index('x_h2o_L')
    x_co2_L_idx = header.index('x_co2_L')
    Z_L_idx = header.index('Z_L')

    for datarow in file_reader:
        p = float(datarow[p_idx])
        T = float(datarow[T_idx])
        success = int(datarow[success_idx])
        y = float(datarow[y_idx])
        Z_L = float(datarow[Z_L_idx])
        y = float(datarow[y_idx])
        Z_G = float(datarow[Z_G_idx])
        x_h2o_L = float(datarow[x_h2o_L_idx])
        x_co2_L = float(datarow[x_co2_L_idx])
        x_h2o_G = float(datarow[x_h2o_G_idx])
        x_co2_G = float(datarow[x_co2_G_idx])

        assert (p, T) in pT_id.keys(), f"Point {(p, T)} not in thermo data files."

        result_data[(p, T)] = dict()
        result_data[(p, T)].update(
            {
                'success': success,
                'p': p,
                'T': p,
                'y': y,
                'Z_L': Z_L,
                'Z_G': Z_G,
                'x_h2o_L': x_h2o_L,
                'x_co2_L': x_co2_L,
                'x_h2o_G': x_h2o_G,
                'x_co2_G': x_co2_G,
            }
        )

### Creating various plots

print("Creating plots ...", flush=True)

# creating basic mesh grid from pT points
p_vec = np.sort(np.unique(np.array(p_points)))
T_vec = np.sort(np.unique(np.array(T_points)))
# Temperature as first axis, pressure as second axis
T_mesh, p_mesh = np.meshgrid(T_vec, p_vec)
nx, ny = T_mesh.shape

## Plot 1: success rate and phase regions
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
            success = results['success'] + 1
            # value map:
            # 1 -> liquid only
            # 2 -> gas only
            # 3 -> 2-phase
            if identifier[0] == 'G':
                mode = 2
            elif identifier[0] == 'L':
                mode = 1
            elif identifier[0] == 'GL':
                mode = 3

            success_mesh[i, j] = success
            region_mesh[i, j] = mode

figwidth = 15
fig = plt.figure(figsize=(figwidth, 1080 / 1920 * figwidth))
gs = fig.add_gridspec(1, 2)

# plotting phase regions
ax_regions = fig.add_subplot(gs[0, 0])
cmap = mpl.colors.ListedColormap(["white", "blue", "yellow", "red"])
img_regions = ax_regions.pcolormesh(
    p_mesh, T_mesh, region_mesh, cmap=cmap, vmin=0, vmax=3
)
ax_regions.set_title("Phase regions")
ax_regions.set_xlabel("T")
ax_regions.set_ylabel("p [MPa]")

divider = make_axes_locatable(ax_regions)
cax = divider.append_axes("right", size="5%", pad=0.1)
cb_rr = fig.colorbar(img_regions, cax=cax, orientation="vertical")
cb_rr.set_ticks([1, 2, 3])
cb_rr.set_ticklabels(["liquid", "gas", "2-phase"])

# plotting success
ax_sucess = fig.add_subplot(gs[0, 1])
cmap = mpl.colors.ListedColormap(["white", "red", "green"])
img_success = ax_sucess.pcolormesh(
    p_mesh, T_mesh, success_mesh, cmap=cmap, vmin=0, vmax=2
)
ax_sucess.set_title("Flash success")
ax_sucess.set_xlabel("T")
ax_sucess.set_ylabel("p [MPa]")

divider = make_axes_locatable(ax_sucess)
cax = divider.append_axes("right", size="5%", pad=0.1)
cb_rr = fig.colorbar(img_success, cax=cax, orientation="vertical")
cb_rr.set_ticks([1, 2])
cb_rr.set_ticklabels(["failed", "succeeded"])

fig.tight_layout()
fig.savefig(f"{str(path)}/{figure_path}success_and_regions.png", format="png", dpi=500)
fig.show()
