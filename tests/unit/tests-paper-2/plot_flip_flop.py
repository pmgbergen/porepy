import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
import pdb


os.system("clear")


def load_output_flip_flop(filename):
    """ """
    data = np.loadtxt(filename, delimiter=",")
    return (
        data[:, 0],
        data[:, [1, 2]].T,
        data[:, [3, 4]].T,
    )


#####################################################

output_file_ppu = "./case_1/slanted_ppu_Kn0.1"
output_file_hu = "./case_1/slanted_hu_Kn0.1"

fontsize = 28
my_orange = "darkorange"
my_blu = [0.1, 0.1, 0.8]

# x_ticks = np.array([0, 10, 20, 30, 40, 50])

save_folder = "./case_1"

#####################################################

time_ppu, cumulative_flips_ppu, global_cumulative_flips_ppu = load_output_flip_flop(
    output_file_ppu + "/FLIPS"
)

time_hu, cumulative_flips_hu, global_cumulative_flips_hu = load_output_flip_flop(
    output_file_hu + "/FLIPS"
)


plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=fontsize)

params = {"text.latex.preamble": r"\usepackage{bm}\usepackage{amsmath}"}
plt.rcParams.update(params)
matplotlib.rcParams["axes.linewidth"] = 1.5

fig, ax_1 = plt.subplots()

for label in ax_1.get_xticklabels() + ax_1.get_yticklabels():
    label.set_fontsize(fontsize)

ax_1.plot(
    time_ppu,
    global_cumulative_flips_ppu[0],
    label="$darcy phase 0$",
    linestyle="--",
    color=my_orange,
    marker="",
)
ax_1.plot(
    time_ppu,
    global_cumulative_flips_ppu[1],
    label="$darcy phase 1$",
    linestyle="--",
    color=my_orange,
    marker="",
)
ax_1.plot(
    time_hu,
    global_cumulative_flips_hu[0],
    label="$q_T$",
    linestyle="-",
    color=my_blu,
    marker="",
)
ax_1.plot(
    time_hu,
    global_cumulative_flips_hu[1],
    label="$omega_0$",
    linestyle="-",
    color=my_blu,
    marker="",
)
ax_1.set_xlabel("time $[s]$", fontsize=fontsize)
# ax_1.set_xticks(x_ticks)

ax_1.grid(linestyle="--", alpha=0.5)

plt.savefig(
    save_folder + "/flip_flop.pdf",
    dpi=150,
    bbox_inches="tight",
    pad_inches=0.2,
)


# legend:
handles_all, labels_all = [
    (a + b)
    for a, b in zip(ax_1.get_legend_handles_labels(), ax_1.get_legend_handles_labels())
]

handles = np.ravel(np.reshape(handles_all[:4], (1, 4)), order="F")
labels = np.ravel(np.reshape(labels_all[:4], (1, 4)), order="F")
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

filename = save_folder + "/flip_flop" + "_legend.pdf"
fig.savefig(filename, bbox_inches="tight")
plt.gcf().clear()

os.system("pdfcrop --margins '0 -800 0 0' " + filename + " " + filename)
os.system("pdfcrop " + filename + " " + filename)

print("\nDone!")
