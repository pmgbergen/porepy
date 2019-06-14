import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------------------#

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=15)


def plot_single(file_name, legend, title):

    data = np.loadtxt(file_name, delimiter=",")

    plt.figure(0)
    plt.plot(data[:, 0], data[:, 1], label=legend)
    plt.title(title)
    plt.xlabel("$t$")
    plt.ylabel("$\\theta$")
    plt.grid(True)
    plt.legend()


# ------------------------------------------------------------------------------#


def plot_multiple(file_name, legend, title, num_frac):

    data = np.loadtxt(file_name, delimiter=",")

    for frac_id in np.arange(num_frac):
        plt.figure(frac_id)
        plt.plot(data[:, 0], data[:, frac_id + 1], label=legend)
        plt_title = (
            title[0] + " on " + "$\\Omega_{" + str(frac_id) + "}$" + " " + title[1]
        )
        plt.title(plt_title)
        plt.xlabel("$t$")
        plt.ylabel("$\\theta$")
        plt.grid(True)
        plt.legend()


# ------------------------------------------------------------------------------#


def save_single(filename, folder, figure_id=0):

    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.figure(figure_id)
    plt.savefig(folder + filename, bbox_inches="tight")
    plt.gcf().clear()


# ------------------------------------------------------------------------------#


def save_multiple(filename, num_frac, folder):

    if not os.path.exists(folder):
        os.makedirs(folder)

    for frac_id in np.arange(num_frac):
        plt.figure(frac_id)
        name = filename + "_frac_" + str(frac_id)
        plt.savefig(folder + name, bbox_inches="tight")
        plt.gcf().clear()


# ------------------------------------------------------------------------------#


def main():

    num_frac = 10

    master_folder = "/home/elle/Dropbox/Work/PresentazioniArticoli/2019/Articles/tipetut++/Results/example2/"

    methods_stefano_1 = ["OPTxfem", "OPTfem"]
    methods_stefano_2 = ["GCmfem"]
    methods_alessio = ["MVEM_UPWIND", "Tpfa_UPWIND", "RT0_UPWIND"]
    methods_andrea = ["MVEM_VEMSUPG", "MVEM_VEMSUPG_POWERTAU"]

    grids = {
        "grid_0": ("3k", "200", "3", "9e-05"),
        "grid_1": ("40k", "2600", "40", "0.0015"),
    }
    grids_label = {"grid_0": "coarse", "grid_1": "fine"}

    for grid_name, grid in grids.items():
        grid_label = grids_label[grid_name]

        folder_in = master_folder
        folder_out = folder_in + "img/"

        title = ["avg $\\theta$", grid_label]
        # Alessio
        for method in methods_alessio:
            data = folder_in + method + "/" + "Cmean_" + grid[0] + ".csv"
            plot_multiple(data, method.replace("_", " "), title, num_frac)

        # Stefano
        for method in methods_stefano_1:
            data = folder_in + method + "/" + method + "_Cmean_" + grid[1] + ".csv"
            plot_multiple(data, method, title, num_frac)

        for method in methods_stefano_2:
            data = folder_in + method + "/" + method + "_Cmean_" + grid[2] + ".csv"
            plot_multiple(data, method, title, num_frac)

        # Andrea
        for method in methods_andrea:
            data = folder_in + method + "/" + "Cmean_" + grid[3] + ".csv"
            plot_multiple(data, method.replace("_", " "), title, num_frac)

        # save
        name = grid_label + "_cot_avg"
        save_multiple(name, num_frac, folder_out)

        ###########

        title = ["min $\\theta$", grid_label]
        # Alessio
        for method in methods_alessio:
            data = folder_in + method + "/" + "Cmin_" + grid[0] + ".csv"
            plot_multiple(data, method.replace("_", " "), title, num_frac)

        # Stefano
        for method in methods_stefano_1:
            data = folder_in + method + "/" + method + "_Cmin_" + grid[1] + ".csv"
            plot_multiple(data, method, title, num_frac)

        for method in methods_stefano_2:
            data = folder_in + method + "/" + method + "_Cmin_" + grid[2] + ".csv"
            plot_multiple(data, method, title, num_frac)

        # Andrea
        for method in methods_andrea:
            data = folder_in + method + "/" + "Cmin_" + grid[3] + ".csv"
            plot_multiple(data, method.replace("_", " "), title, num_frac)

        # save
        name = grid_label + "_cot_min"
        save_multiple(name, num_frac, folder_out)

        ###########

        title = ["max $\\theta$", grid_label]
        # Alessio
        for method in methods_alessio:
            data = folder_in + method + "/" + "Cmax_" + grid[0] + ".csv"
            plot_multiple(data, method.replace("_", " "), title, num_frac)

        # Stefano
        for method in methods_stefano_1:
            data = folder_in + method + "/" + method + "_Cmax_" + grid[1] + ".csv"
            plot_multiple(data, method, title, num_frac)

        for method in methods_stefano_2:
            data = folder_in + method + "/" + method + "_Cmax_" + grid[2] + ".csv"
            plot_multiple(data, method, title, num_frac)

        # Andrea
        for method in methods_andrea:
            data = folder_in + method + "/" + "Cmax_" + grid[3] + ".csv"
            plot_multiple(data, method.replace("_", " "), title, num_frac)

        # save
        name = grid_label + "_cot_max"
        save_multiple(name, num_frac, folder_out)

        ###########

        title = "production on " + grid_label
        # Alessio
        for method in methods_alessio:
            data = folder_in + method + "/" + "production_" + grid[0] + ".csv"
            plot_single(data, method.replace("_", " "), title)

        # Stefano
        for method in methods_stefano_1:
            data = folder_in + method + "/" + method + "_production_" + grid[1] + ".csv"
            plot_single(data, method, title)

        for method in methods_stefano_2:
            data = folder_in + method + "/" + method + "_production_" + grid[2] + ".csv"
            plot_single(data, method, title)

        # Andrea
        for method in methods_andrea:
            data = folder_in + method + "/" + "production_" + grid[3] + ".csv"
            plot_single(data, method.replace("_", " "), title)

        # save
        name = grid_label + "_outflow"
        save_single(name, folder_out)


# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()
