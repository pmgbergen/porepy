import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------------------#

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=15)


def plot_single(file_name, legend, title, **kwargs):

    data = np.loadtxt(file_name, delimiter=",")
    reference = kwargs.get("reference", None)

    fig = plt.figure(0)
    ax = fig.add_subplot(111)

    # if the data is a reference
    if reference:
        data_p = data[:, 1] + data[:, 1] * reference / 100
        plt.plot(data[:, 0], data_p, label=legend, linestyle="--", color="gray")
        text = "ref + " + str(reference) + "\%"
        pos = (np.median(data[:, 0]), np.median(data_p))
        pos_t = (pos[0], pos[1]+5*pos[1]/100)
        ax.annotate(text, xy=pos, xytext=pos_t)

        data_m = data[:, 1] - data[:, 1] * reference / 100
        plt.plot(data[:, 0], data_m, label=legend, linestyle="--", color="gray")
        text = "ref - " + str(reference) + "\%"
        pos = (np.median(data[:, 0]), np.median(data_m))
        pos_t = (pos[0], pos[1]-5*pos[1]/100)
        ax.annotate(text, xy=pos, xytext=pos_t)

    else:
        plt.plot(data[:, 0], data[:, 1], label=legend)

    plt.title(title)
    plt.xlabel("$t$")
    plt.ylabel("$\\theta$")
    plt.grid(True)
    plt.legend()


# ------------------------------------------------------------------------------#


def plot_multiple(file_name, legend, title, num_frac, **kwargs):

    data = np.loadtxt(file_name, delimiter=",")
    frac_label = {0: "$\\Omega_l$", 1: "$\\Omega_m$", 2: "$\\Omega_r$"}

    reference = kwargs.get("reference", None)

    for frac_id in np.arange(num_frac):
        fig = plt.figure(frac_id)
        ax = fig.add_subplot(111)

        # if the data is a reference
        if reference:
            data_p = data[:, frac_id + 1] + data[:, frac_id + 1] * reference / 100
            plt.plot(data[:, 0], data_p, label=legend, linestyle="--", color="gray")
            text = "ref + " + str(reference) + "\%"
            pos = (np.median(data[:, 0]), np.median(data_p))
            pos_t = (pos[0], pos[1]+5*pos[1]/100)
            ax.annotate(text, xy=pos, xytext=pos_t)

            data_m = data[:, frac_id + 1] - data[:, frac_id + 1] * reference / 100
            plt.plot(data[:, 0], data_m, label=legend, linestyle="--", color="gray")
            text = "ref - " + str(reference) + "\%"
            pos = (np.median(data[:, 0]), np.median(data_m))
            pos_t = (pos[0], pos[1]-5*pos[1]/100)
            ax.annotate(text, xy=pos, xytext=pos_t)

        else:
            plt.plot(data[:, 0], data[:, frac_id + 1], label=legend)

        plt_title = (
            title[0]
            + " on "
            + frac_label[frac_id]
            + " "
            + title[1]
            + " - "
            + " config "
            + str(title[2])
        )
        plt.title(plt_title)
        plt.xlabel("$t$")
        plt.ylabel("$\\theta$")
        plt.grid(True)
        plt.legend()


# ------------------------------------------------------------------------------#


def plot_num_cells(data, legend, title):

    data = np.loadtxt(data, delimiter=",")
    data = np.atleast_2d(data)

    plt.figure(0)
    plt.plot(np.arange(data.shape[0]), data[:, -1], label=legend)
    plt.title(title)
    plt.xlabel("config.")
    plt.ylabel("num. cells")
    plt.grid(True)
    plt.legend()
    # useful to plot the legend as flat
    # ncol = 5 # number of methods
    # plt.legend(bbox_to_anchor=(1, -0.2), ncol=5)


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

    num_simul = 21
    num_frac = 3

    master_folder = "/home/elle/Dropbox/Work/PresentazioniArticoli/2019/Articles/tipetut++/Results/example1/"

    methods_stefano_1 = ["OPTxfem", "OPTfem"]
    methods_stefano_2 = ["GCmfem"]
    methods_alessio = ["MVEM_UPWIND", "Tpfa_UPWIND", "RT0_UPWIND"]
    methods_andrea = ["MVEM_VEMSUPG"]

    method_reference = "GCmfem"
    reference = {"grid_0": 10, "grid_1": 5, "grid_2": 3.5}

    grids = {
        "grid_0": ("1k", "220", "1", "0.005"),
        "grid_1": ("3k", "650", "3", "0.0015"),
        "grid_2": ("10k", "2100", "10", "0.00045"),
    }
    grids_label = {"grid_0": "coarse", "grid_1": "medium", "grid_2": "fine"}


    for grid_name, grid in grids.items():
        grid_label = grids_label[grid_name]
        for simul in np.arange(num_simul):

            folder_in = master_folder
            folder_out = folder_in + "img/"

            title = ["avg $\\theta$", grid_label, simul]

            # Reference
            data = (
                folder_in
                + method_reference
                + "/"
                + method_reference
                + "_Cmean_"
                + str(simul + 1)
                + "_big"
                + ".csv"
            )
            plot_multiple(data, None, title, num_frac, reference=reference[grid_name])

            # Alessio
            for method in methods_alessio:
                data = (
                    folder_in
                    + method
                    + "/"
                    + "Cmean_"
                    + str(simul + 1)
                    + "_"
                    + grid[0]
                    + ".csv"
                )
                plot_multiple(data, method.replace("_", " "), title, num_frac)

            # Stefano
            for method in methods_stefano_1:
                data = (
                    folder_in
                    + method
                    + "/"
                    + method
                    + "_Cmean_"
                    + str(simul + 1)
                    + "_"
                    + grid[1]
                    + ".csv"
                )
                plot_multiple(data, method, title, num_frac)

            for method in methods_stefano_2:
                data = (
                    folder_in
                    + method
                    + "/"
                    + method
                    + "_Cmean_"
                    + str(simul + 1)
                    + "_"
                    + grid[2]
                    + ".csv"
                )
                plot_multiple(data, method, title, num_frac)

            # Andrea
            for method in methods_andrea:
                data = (
                    folder_in
                    + method
                    + "/"
                    + "Cmean_"
                    + str(simul + 1)
                    + "_"
                    + grid[3]
                    + ".csv"
                )
                plot_multiple(data, method.replace("_", " "), title, num_frac)

            # save
            name = grid_label + "_cot_avg_" + str(simul)
            save_multiple(name, num_frac, folder_out)

            ###########

            title = ["min $\\theta$", grid_label, simul]

            # Reference
            data = (
                folder_in
                + method_reference
                + "/"
                + method_reference
                + "_Cmin_"
                + str(simul + 1)
                + "_big"
                + ".csv"
            )
            plot_multiple(data, None, title, num_frac, reference=reference[grid_name])

            # Alessio
            for method in methods_alessio:
                data = (
                    folder_in
                    + method
                    + "/"
                    + "Cmin_"
                    + str(simul + 1)
                    + "_"
                    + grid[0]
                    + ".csv"
                )
                plot_multiple(data, method.replace("_", " "), title, num_frac)

            # Stefano
            for method in methods_stefano_1:
                data = (
                    folder_in
                    + method
                    + "/"
                    + method
                    + "_Cmin_"
                    + str(simul + 1)
                    + "_"
                    + grid[1]
                    + ".csv"
                )
                plot_multiple(data, method, title, num_frac)

            # Stefano
            for method in methods_stefano_2:
                data = (
                    folder_in
                    + method
                    + "/"
                    + method
                    + "_Cmin_"
                    + str(simul + 1)
                    + "_"
                    + grid[2]
                    + ".csv"
                )
                plot_multiple(data, method, title, num_frac)

            # Andrea
            for method in methods_andrea:
                data = (
                    folder_in
                    + method
                    + "/"
                    + "Cmin_"
                    + str(simul + 1)
                    + "_"
                    + grid[3]
                    + ".csv"
                )
                plot_multiple(data, method.replace("_", " "), title, num_frac)

            # save
            name = grid_label + "_cot_min_" + str(simul)
            save_multiple(name, num_frac, folder_out)

            ###########

            title = ["max $\\theta$", grid_label, simul]

            # Reference
            data = (
                folder_in
                + method_reference
                + "/"
                + method_reference
                + "_Cmax_"
                + str(simul + 1)
                + "_big"
                + ".csv"
            )
            plot_multiple(data, None, title, num_frac, reference=reference[grid_name])

            # Alessio
            for method in methods_alessio:
                data = (
                    folder_in
                    + method
                    + "/"
                    + "Cmax_"
                    + str(simul + 1)
                    + "_"
                    + grid[0]
                    + ".csv"
                )
                plot_multiple(data, method.replace("_", " "), title, num_frac)

            # Stefano
            for method in methods_stefano_1:
                data = (
                    folder_in
                    + method
                    + "/"
                    + method
                    + "_Cmax_"
                    + str(simul + 1)
                    + "_"
                    + grid[1]
                    + ".csv"
                )
                plot_multiple(data, method, title, num_frac)

            for method in methods_stefano_2:
                data = (
                    folder_in
                    + method
                    + "/"
                    + method
                    + "_Cmax_"
                    + str(simul + 1)
                    + "_"
                    + grid[2]
                    + ".csv"
                )
                plot_multiple(data, method, title, num_frac)

            # Andrea
            for method in methods_andrea:
                data = (
                    folder_in
                    + method
                    + "/"
                    + "Cmax_"
                    + str(simul + 1)
                    + "_"
                    + grid[3]
                    + ".csv"
                )
                plot_multiple(data, method.replace("_", " "), title, num_frac)

            # save
            name = grid_label + "_cot_max_" + str(simul)
            save_multiple(name, num_frac, folder_out)

            ###########

            title = "production on " + grid_label + " - config " + str(simul)

            # Reference
            data = (
                folder_in
                + method_reference
                + "/"
                + method_reference
                + "_production_"
                + str(simul + 1)
                + "_big"
                + ".csv"
            )
            plot_single(data, None, title, reference=reference[grid_name])

            # Alessio
            for method in methods_alessio:
                data = (
                    folder_in
                    + method
                    + "/"
                    + "production_"
                    + str(simul + 1)
                    + "_"
                    + grid[0]
                    + ".csv"
                )
                plot_single(data, method.replace("_", " "), title)

            # Stefano
            for method in methods_stefano_1:
                data = (
                    folder_in
                    + method
                    + "/"
                    + method
                    + "_production_"
                    + str(simul + 1)
                    + "_"
                    + grid[1]
                    + ".csv"
                )
                plot_single(data, method, title)

            for method in methods_stefano_2:
                data = (
                    folder_in
                    + method
                    + "/"
                    + method
                    + "_production_"
                    + str(simul + 1)
                    + "_"
                    + grid[2]
                    + ".csv"
                )
                plot_single(data, method, title)

            # Andrea
            for method in methods_andrea:
                data = (
                    folder_in
                    + method
                    + "/"
                    + "production_"
                    + str(simul + 1)
                    + "_"
                    + grid[3]
                    + ".csv"
                )
                plot_single(data, method.replace("_", " "), title)

            # save
            name = grid_label + "_outflow_" + str(simul)
            save_single(name, folder_out)

            ########

        title = "number of cells - " + grid_label
        # Alessio
        for method in methods_alessio:
            data = folder_in + method + "/" + "num_cells_" + grid[0] + ".csv"
            plot_num_cells(data, method.replace("_", " "), title)

        # Stefano
        for method in methods_stefano_1:
            data = folder_in + method + "/" + "num_cells_" + grid[1] + ".csv"
            plot_num_cells(data, method.replace("_", " "), title)

        for method in methods_stefano_2:
            data = folder_in + method + "/" + method + "_cells_" + grid[2] + ".csv"
            plot_num_cells(data, method.replace("_", " "), title)

        # Andrea
        for method in methods_andrea:
            data = folder_in + method + "/" + "num_cells_" + grid[3] + ".csv"
            plot_num_cells(data, method.replace("_", " "), title)

        name = grid_label + "_num_cells"
        save_single(name, folder_out)


# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()
