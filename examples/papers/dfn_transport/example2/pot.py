import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------------------#

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=15)

main_folder = "./plots/"


def plot(file_name, legend, title, num_frac):

    data = np.loadtxt(file_name, delimiter=",")
    for frac_id in np.arange(num_frac):
        plt.figure(frac_id)
        plt.plot(data[:, 0], data[:, frac_id + 1], label=legend)
        plt.title(title + " - " + str(frac_id))
        plt.xlabel("$t$")
        plt.ylabel("$c$")
        plt.grid(True)
        plt.legend()


# ------------------------------------------------------------------------------#


def save(filename, num_frac):

    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    folder = main_folder + "/"

    if not os.path.exists(folder):
        os.makedirs(folder)

    for frac_id in np.arange(num_frac):
        plt.figure(frac_id)
        name = filename + "_frac_" + str(frac_id)
        plt.savefig(folder + name, bbox_inches="tight")
        plt.gcf().clear()


# ------------------------------------------------------------------------------#

num_frac = 10

data = "./solution/dot_avg.csv"
label = "LABEL"
title = "average cot"
plot(data, label, title, num_frac)

name = "cot_avg"
save(name, num_frac)

###########

data = "./solution/dot_min.csv"
label = "LABEL"
title = "min cot"
plot(data, label, title, num_frac)

name = "cot_min"
save(name, num_frac)

###########

data = "./solution/dot_max.csv"
label = "LABEL"
title = "max cot"
plot(data, label, title, num_frac)

name = "cot_max"
save(name, num_frac)

# ------------------------------------------------------------------------------#
