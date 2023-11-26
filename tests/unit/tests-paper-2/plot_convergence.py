import scipy as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Callable, Optional, Type, Literal, Sequence, Union
import porepy as pp
import os
import pdb

os.system("clear")

# root_paths = [
#     "./case_1/slanted_hu_Kn0.1/convergence_results/",
#     "./case_1/slanted_ppu_Kn0.1/convergence_results/",
# ]
root_paths = [
    "./case_1/non-conforming/slanted_hu_Kn0.1/convergence_results/",
    "./case_1/non-conforming/slanted_ppu_Kn0.1/convergence_results/",
]

err_list_p_2d = [np.array([])] * 2
err_list_p_1d = [np.array([])] * 2
err_list_s_2d = [np.array([])] * 2
err_list_s_1d = [np.array([])] * 2
err_list_mortar_0 = [np.array([])] * 2
err_list_mortar_1 = [np.array([])] * 2


for index, root_path in enumerate(root_paths):
    cell_sizes = np.loadtxt(root_path + "cell_sizes")  # last one is the ref

    # fine (ref): ------------------------------------------------------
    cell_size = cell_sizes[-1]
    variable_num_dofs = np.loadtxt(root_path + "variable_num_dofs_" + str(cell_size))

    volumes_2d_ref = np.load(
        root_path + "volumes_2d_" + str(cell_size) + ".npy", allow_pickle=True
    )
    volumes_1d_ref = np.loadtxt(root_path + "volumes_1d_" + str(cell_size))
    volumes_mortar = np.concatenate((volumes_1d_ref, volumes_1d_ref))

    id_2d = np.arange(variable_num_dofs[0], dtype=np.int32)  # HARDCODED
    id_1d = np.arange(
        variable_num_dofs[0],
        variable_num_dofs[0] + variable_num_dofs[1],
        dtype=np.int32,
    )  # HARDCODED

    pressure = np.load(
        root_path + "pressure_" + str(cell_size) + ".npy", allow_pickle=True
    )
    saturation = np.load(
        root_path + "saturation_" + str(cell_size) + ".npy", allow_pickle=True
    )
    mortar_phase_0_ref = np.loadtxt(root_path + "mortar_phase_0_" + str(cell_size))
    mortar_phase_1_ref = np.loadtxt(root_path + "mortar_phase_1_" + str(cell_size))

    pressure_2d_ref = pressure[id_2d]
    saturation_2d_ref = saturation[id_2d]
    pressure_1d_ref = pressure[id_1d]
    saturation_1d_ref = saturation[id_1d]

    for cell_size in cell_sizes[:-1]:
        # coarse: --------------------------------------------------------
        variable_num_dofs = np.loadtxt(
            root_path + "variable_num_dofs_" + str(cell_size)
        )

        coarse_to_fine_2d = sp.sparse.load_npz(
            root_path + "mapping_matrix_2d_" + str(cell_size) + ".npz"
        )
        coarse_to_fine_1d = sp.sparse.load_npz(
            root_path + "mapping_matrix_1d_" + str(cell_size) + ".npz"
        )  #

        coarse_to_fine_intf = sp.sparse.kron(
            sp.sparse.eye(2), coarse_to_fine_1d
        )  # sorry, I want to use kron...
        # intf is conforming with 1d

        pressure = np.load(root_path + "pressure_" + str(cell_size) + ".npy")
        saturation = np.load(root_path + "saturation_" + str(cell_size) + ".npy")
        mortar_phase_0 = np.loadtxt(root_path + "mortar_phase_0_" + str(cell_size))
        mortar_phase_1 = np.loadtxt(root_path + "mortar_phase_1_" + str(cell_size))

        id_2d = np.arange(variable_num_dofs[0], dtype=np.int32)  # HARDCODED
        id_1d = np.arange(
            variable_num_dofs[0],
            variable_num_dofs[0] + variable_num_dofs[1],
            dtype=np.int32,
        )  # HARDCODED

        pressure_2d_proj = coarse_to_fine_2d @ pressure[id_2d]
        pressure_1d_proj = coarse_to_fine_1d @ pressure[id_1d]
        saturation_2d_proj = coarse_to_fine_2d @ saturation[id_2d]
        saturation_1d_proj = coarse_to_fine_1d @ saturation[id_1d]

        mortar_phase_0_proj = coarse_to_fine_intf @ mortar_phase_0
        mortar_phase_1_proj = coarse_to_fine_intf @ mortar_phase_1

        err_p_2d = np.linalg.norm(
            (pressure_2d_proj - pressure_2d_ref) * volumes_2d_ref, ord=2
        ) / np.linalg.norm(pressure_2d_ref * volumes_2d_ref, ord=2)

        err_p_1d = np.linalg.norm(
            (pressure_1d_proj - pressure_1d_ref) * volumes_1d_ref, ord=2
        ) / np.linalg.norm(pressure_1d_ref * volumes_1d_ref, ord=2)

        err_s_2d = np.linalg.norm(
            (saturation_2d_proj - saturation_2d_ref) * volumes_2d_ref, ord=2
        ) / np.linalg.norm(saturation_2d_ref * volumes_2d_ref, ord=2)

        err_s_1d = np.linalg.norm(
            (saturation_1d_proj - saturation_1d_ref) * volumes_1d_ref, ord=2
        ) / np.linalg.norm(saturation_1d_ref * volumes_1d_ref, ord=2)

        err_mortar_0 = np.linalg.norm(
            (mortar_phase_0_proj - mortar_phase_0_ref) * volumes_mortar, ord=2
        ) / np.linalg.norm(mortar_phase_0_ref * volumes_mortar, ord=2)

        err_mortar_1 = np.linalg.norm(
            (mortar_phase_1_proj - mortar_phase_1_ref) * volumes_mortar, ord=2
        ) / np.linalg.norm(mortar_phase_1_ref * volumes_mortar, ord=2)

        err_list_p_2d[index] = np.append(err_list_p_2d[index], err_p_2d)
        err_list_p_1d[index] = np.append(err_list_p_1d[index], err_p_1d)
        err_list_s_2d[index] = np.append(err_list_s_2d[index], err_s_2d)
        err_list_s_1d[index] = np.append(err_list_s_1d[index], err_s_1d)
        err_list_mortar_0[index] = np.append(err_list_mortar_0[index], err_mortar_0)
        err_list_mortar_1[index] = np.append(err_list_mortar_1[index], err_mortar_1)

######################################################

fontsize = 28
my_orange = "darkorange"
my_blu = [0.1, 0.1, 0.8]

x_ticks = np.linspace(cell_sizes[0], cell_sizes[-2], 3)

# save_folder = "./case_1/slanted_ppu_hu_"
save_folder = "./case_1/non-conforming/slanted_ppu_hu_"

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
        "err_list_p_1d",
        "err_list_s_2d",
        "err_list_s_1d",
        "err_list_mortar_0",
        "err_list_mortar_1",
    ],
    [
        err_list_p_2d,
        err_list_p_1d,
        err_list_s_2d,
        err_list_s_1d,
        err_list_mortar_0,
        err_list_mortar_1,
    ],
)

for name, val in items:
    fig, ax_1 = plt.subplots()

    for label in ax_1.get_xticklabels() + ax_1.get_yticklabels():
        label.set_fontsize(fontsize)

    ax_1.loglog(
        cell_sizes[:-1],
        val[0],
        label=name,
        linestyle="-",
        color=[0, 0, 0],
        marker="",
    )
    ax_1.loglog(
        cell_sizes[:-1],
        val[1],
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
        save_folder + "convergence_" + name + ".pdf",
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

    filename = save_folder + "convergence_" + name + "_legend.pdf"
    fig.savefig(filename, bbox_inches="tight")
    plt.gcf().clear()

    os.system("pdfcrop --margins '0 -800 0 0' " + filename + " " + filename)
    os.system("pdfcrop " + filename + " " + filename)
