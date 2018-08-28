import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------------------#

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=15)


def plot(file_name, ID, simulation_id):

    data = np.loadtxt(file_name, delimiter=",")
    plt.figure(simulation_id)
    plt.plot(data[:, 0], data[:, 1], label=ID)
    plt.title("pressure over line $k_{f, t} = k_{f, n} = 1e-4$")
    plt.xlabel("archlenght")
    plt.ylabel("$p$")
    plt.grid(True)
    plt.legend()


def save(simulation_id):
    folder = "./plots/"
    file_name = "pol_1e_4_" + str(simulation_id)

    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.figure(simulation_id)
    plt.savefig(folder + file_name, bbox_inches="tight")


# ------------------------------------------------------------------------------#
# Insert here your data for the plotting, see the file 'color_regions.vtu'
# for the coloring code of each region.

ID = "UiB-MVEM"
simulation_id = 0
plot("./UiB/MVEM/1e_4/pol_0.csv", ID, simulation_id)

# ------------------------------------------------------------------------------#

simulation_id = 0
save(simulation_id)

# ------------------------------------------------------------------------------#
