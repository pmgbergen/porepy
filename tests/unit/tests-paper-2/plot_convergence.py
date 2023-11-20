import scipy as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Callable, Optional, Type, Literal, Sequence, Union
import porepy as pp
import os
import pdb

os.system("clear")

root_path = "./convergence_results/"
cell_sizes = np.loadtxt(root_path + "cell_sizes")  # last one is the ref
err_list = np.array([])


# fine (ref): ------------------------------------------------------
cell_size = cell_sizes[-1]
variable_num_dofs = np.loadtxt(root_path + "variable_num_dofs_" + str(cell_size))
volumes_2d_ref = np.loadtxt(root_path + "volumes_2d_" + str(cell_size))
volumes_2d_ref = np.concatenate((volumes_2d_ref, volumes_2d_ref))
volumes_1d_ref = np.loadtxt(root_path + "volumes_1d_" + str(cell_size))
volumes_1d_ref = np.concatenate((volumes_1d_ref, volumes_1d_ref))


pressure = np.loadtxt(root_path + "pressure_" + str(cell_size))
saturation = np.loadtxt(root_path + "saturation_" + str(cell_size))

id_2d = np.arange(variable_num_dofs[0], dtype=np.int32)  # HARDCODED
id_1d = np.arange(
    variable_num_dofs[0], variable_num_dofs[0] + variable_num_dofs[1], dtype=np.int32
)  # HARDCODED

pressure_2d = pressure[id_2d]
pressure_1d = pressure[id_1d]
saturation_2d = saturation[id_2d]
saturation_1d = saturation[id_1d]

sol_2d_ref = np.concatenate((pressure_2d, saturation_2d))
sol_1d_ref = np.concatenate((pressure_1d, saturation_1d))


for cell_size in cell_sizes[:-1]:
    # coarse: --------------------------------------------------------
    variable_num_dofs = np.loadtxt(root_path + "variable_num_dofs_" + str(cell_size))
    # volumes_1d = np.loadtxt(root_path + "volumes_1d_" + str(cell_size)) # not used it you project vars onto ref
    # volumes_2d = np.loadtxt(root_path + "volumes_2d_" + str(cell_size))

    coarse_to_fine_2d = sp.sparse.load_npz(
        root_path + "mapping_matrix_2d_" + str(cell_size) + ".npz"
    )
    coarse_to_fine_1d = sp.sparse.load_npz(
        root_path + "mapping_matrix_1d_" + str(cell_size) + ".npz"
    )

    pressure = np.loadtxt(root_path + "pressure_" + str(cell_size))
    saturation = np.loadtxt(root_path + "saturation_" + str(cell_size))

    id_2d = np.arange(variable_num_dofs[0], dtype=np.int32)  # HARDCODED
    id_1d = np.arange(
        variable_num_dofs[0],
        variable_num_dofs[0] + variable_num_dofs[1],
        dtype=np.int32,
    )  # HARDCODED

    pressure_2d = coarse_to_fine_2d @ pressure[id_2d]
    pressure_1d = coarse_to_fine_1d @ pressure[id_1d]
    saturation_2d = coarse_to_fine_2d @ saturation[id_2d]
    saturation_1d = coarse_to_fine_1d @ saturation[id_1d]

    sol_2d = np.concatenate((pressure_2d, saturation_2d))
    sol_1d = np.concatenate((pressure_1d, saturation_1d))

    err = np.linalg.norm(
        (sol_2d - sol_2d_ref) * volumes_2d_ref, ord=2
    ) + np.linalg.norm((sol_1d - sol_1d_ref) * volumes_1d_ref, ord=2)

    err_list = np.append(err_list, err)


print("\nerr_list = ", err_list)

######################################################

fontsize = 28
my_orange = "darkorange"
my_blu = [0.1, 0.1, 0.8]

x_ticks = np.linspace(cell_sizes[0], cell_sizes[-2], 3)

save_folder = root_path

#####################################################

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=fontsize)

params = {"text.latex.preamble": r"\usepackage{bm}\usepackage{amsmath}"}
plt.rcParams.update(params)
matplotlib.rcParams["axes.linewidth"] = 1.5

fig, ax_1 = plt.subplots()

for label in ax_1.get_xticklabels() + ax_1.get_yticklabels():
    label.set_fontsize(fontsize)

ax_1.loglog(
    cell_sizes[:-1],
    err_list,
    label="$err$",
    linestyle="-",
    color=[0, 0, 0],
    marker="",
)
ax_1.loglog(
    cell_sizes[:-1],
    cell_sizes[:-1],
    label="$h^1$",
    linestyle="--",
    color=[0, 0, 0],
    marker="",
)

ax_1.set_xlabel("h", fontsize=fontsize)
ax_1.set_xticks(x_ticks)

ax_1.grid(linestyle="--", alpha=0.5)

plt.savefig(
    save_folder + "/convergence_err.pdf",
    dpi=150,
    bbox_inches="tight",
    pad_inches=0.2,
)


# legend:
handles_all, labels_all = [
    (a + b)
    for a, b in zip(ax_1.get_legend_handles_labels(), ax_1.get_legend_handles_labels())
]

handles = np.ravel(np.reshape(handles_all[:2], (1, 2)), order="F")
labels = np.ravel(np.reshape(labels_all[:2], (1, 2)), order="F")
fig, ax = plt.subplots(figsize=(25, 10))

for h, l in zip(handles, labels):
    ax.plot(np.zeros(1), label=l)

ax.legend(
    handles,
    labels,
    fontsize=fontsize,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(-0.1, -0.65),
)

filename = save_folder + "/convergence_err" + "_legend.pdf"
fig.savefig(filename, bbox_inches="tight")
plt.gcf().clear()

os.system("pdfcrop --margins '0 -800 0 0' " + filename + " " + filename)
os.system("pdfcrop " + filename + " " + filename)
