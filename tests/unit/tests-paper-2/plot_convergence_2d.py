import scipy as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Callable, Optional, Type, Literal, Sequence, Union
import porepy as pp
import os
import pdb

os.system("clear")

# root_path = "./case_0/hu/convergence_results/"
root_path = "./case_0/hu/convergence_results-disc/"

cell_sizes = np.loadtxt(root_path + "cell_sizes")  # last one is the ref
err_list_p_2d = np.array([])
err_list_s_2d = np.array([])

# fine (ref): ------------------------------------------------------
cell_size = cell_sizes[-1]
variable_num_dofs = np.loadtxt(root_path + "variable_num_dofs_" + str(cell_size))

volumes_2d_ref = np.load(
    root_path + "volumes_2d_" + str(cell_size) + ".npy", allow_pickle=True
)
id_2d = np.arange(variable_num_dofs[0], dtype=np.int32)  # HARDCODED

pressure = np.load(root_path + "pressure_" + str(cell_size) + ".npy", allow_pickle=True)
saturation = np.load(
    root_path + "saturation_" + str(cell_size) + ".npy", allow_pickle=True
)

pressure_2d_ref = pressure[id_2d]
saturation_2d_ref = saturation[id_2d]

for cell_size in cell_sizes[:-1]:
    # coarse: --------------------------------------------------------
    variable_num_dofs = np.loadtxt(root_path + "variable_num_dofs_" + str(cell_size))

    coarse_to_fine_2d = sp.sparse.load_npz(
        root_path + "mapping_matrix_2d_" + str(cell_size) + ".npz"
    )

    pressure = np.load(root_path + "pressure_" + str(cell_size) + ".npy")
    saturation = np.load(root_path + "saturation_" + str(cell_size) + ".npy")

    id_2d = np.arange(variable_num_dofs[0], dtype=np.int32)  # HARDCODED

    pressure_2d_proj = coarse_to_fine_2d @ pressure[id_2d]
    saturation_2d_proj = coarse_to_fine_2d @ saturation[id_2d]

    err_p_2d = np.linalg.norm(
        (pressure_2d_proj - pressure_2d_ref) * volumes_2d_ref, ord=2
    ) / np.linalg.norm(pressure_2d_ref * volumes_2d_ref, ord=2)

    err_s_2d = np.linalg.norm(
        (saturation_2d_proj - saturation_2d_ref) * volumes_2d_ref, ord=2
    ) / np.linalg.norm(saturation_2d_ref * volumes_2d_ref, ord=2)

    err_list_p_2d = np.append(err_list_p_2d, err_p_2d)
    err_list_s_2d = np.append(err_list_s_2d, err_s_2d)

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


items = zip(
    [
        "err_list_p_2d",
        "err_list_s_2d",
    ],
    [
        err_list_p_2d,
        err_list_s_2d,
    ],
)

for name, val in items:
    fig, ax_1 = plt.subplots()

    for label in ax_1.get_xticklabels() + ax_1.get_yticklabels():
        label.set_fontsize(fontsize)

    ax_1.loglog(
        cell_sizes[:-1],
        val,
        label=name,
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
        save_folder + "/convergence_" + name + ".pdf",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.2,
    )

    # legend:
    handles_all, labels_all = [
        (a + b)
        for a, b in zip(
            ax_1.get_legend_handles_labels(), ax_1.get_legend_handles_labels()
        )
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

    filename = save_folder + "/convergence_" + name + "_legend.pdf"
    fig.savefig(filename, bbox_inches="tight")
    plt.gcf().clear()

    os.system("pdfcrop --margins '0 -800 0 0' " + filename + " " + filename)
    os.system("pdfcrop " + filename + " " + filename)
