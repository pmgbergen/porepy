import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({"figure.max_open_warning": 0})

# ------------------------------------------------------------------------------#

regions = np.arange(22)

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=15)


def plot(pre, post, name):

    for region in regions:
        data = np.loadtxt(pre + str(region) + post, delimiter=",")
        plt.figure(region)
        plt.plot(data[:, 0], data[:, 1], label=name)
        title = "$c$ over time for region " + str(region) + " - $k_{f, *} = 1e-4$"
        plt.title(title)
        plt.xlabel("$t$")
        plt.ylabel("$c$")
        plt.grid(True)
        plt.legend()


# ------------------------------------------------------------------------------#
# Insert here your data for the plotting, see the file 'color_regions.vtu'
# for the coloring code of each region.

name = "UiB-MVEM"
plot("./UiB/MVEM/1e_4/c_", ".csv", name)

# ------------------------------------------------------------------------------#

folder = "./plots/1e_4/"
file_name = "c_o_t_1e_4_"

if not os.path.exists(folder):
    os.makedirs(folder)

for region in regions:
    plt.figure(region)
    plt.savefig(folder + file_name + str(region) + ".pdf", bbox_inches="tight")

# ------------------------------------------------------------------------------#
