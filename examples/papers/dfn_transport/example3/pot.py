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

    num_frac = 86-7

    master_folder = "/home/elle/Dropbox/Work/PresentazioniArticoli/2019/Articles/tipetut++/Results/example3/"

    methods_stefano = ["OPTfem", "OPTxfem", "GCmfem"]
    methods_alessio = ["MVEM_UPWIND", "Tpfa_UPWIND", "RT0_UPWIND"]
    methods_andrea = []  # ["MVEM_VEMSUPG"]

    cases = {"case_0": ("different", "different", "0.005"), "case_1": ("same", "same", "0.001")}
    cases_label = {"case_0": "different", "case_1": "same"}

    for case_name, case in cases.items():
        case_label = cases_label[case_name]

        folder_in = master_folder
        folder_out = folder_in + "img/"

        title = ["avg $\\theta$", case_label]
        # Alessio
        for method in methods_alessio:
            data = folder_in + method + "/" + "Cmean_" + case[0] + ".csv"
            plot_multiple(data, method.replace("_", " "), title, num_frac)

        # Stefano
        for method in methods_stefano:
            data = folder_in + method + "/" + method + "_Cmean_" + case[1] + ".csv"
            plot_multiple(data, method, title, num_frac)

        # Andrea
        for method in methods_andrea:
            data = folder_in + method + "/" + "Cmean_" + case[2] + ".csv"
            plot_multiple(data, method.replace("_", " "), title, num_frac)

        # save
        name = case_label + "_cot_avg"
        save_multiple(name, num_frac, folder_out)

        ###########

        title = ["min $\\theta$", case_label]
        # Alessio
        for method in methods_alessio:
            data = folder_in + method + "/" + "Cmin_" + case[0] + ".csv"
            plot_multiple(data, method.replace("_", " "), title, num_frac)

        # Stefano
        for method in methods_stefano:
            data = folder_in + method + "/" + method + "_Cmin_" + case[1] + ".csv"
            plot_multiple(data, method, title, num_frac)

        # Andrea
        for method in methods_andrea:
            data = folder_in + method + "/" + "Cmin_" + case[2] + ".csv"
            plot_multiple(data, method.replace("_", " "), title, num_frac)

        # save
        name = case_label + "_cot_min"
        save_multiple(name, num_frac, folder_out)

        ###########

        title = ["max $\\theta$", case_label]
        # Alessio
        for method in methods_alessio:
            data = folder_in + method + "/" + "Cmax_" + case[0] + ".csv"
            plot_multiple(data, method.replace("_", " "), title, num_frac)

        # Stefano
        for method in methods_stefano:
            data = folder_in + method + "/" + method + "_Cmax_" + case[1] + ".csv"
            plot_multiple(data, method, title, num_frac)

        # Andrea
        for method in methods_andrea:
            data = folder_in + method + "/" + "Cmax_" + case[2] + ".csv"
            plot_multiple(data, method.replace("_", " "), title, num_frac)

        # save
        name = case_label + "_cot_max"
        save_multiple(name, num_frac, folder_out)

        ###########

        title = "production on " + case_label
        # Alessio
        for method in methods_alessio:
            data = folder_in + method + "/" + "production_" + case[0] + ".csv"
            plot_single(data, method.replace("_", " "), title)

        # Stefano
        for method in methods_stefano:
            data = folder_in + method + "/" + method + "_production_" + case[1] + ".csv"
            plot_single(data, method, title)

        # Andrea
        for method in methods_andrea:
            data = folder_in + method + "/" + "production_" + case[2] + ".csv"
            plot_single(data, method.replace("_", " "), title)

        # save
        name = case_label + "_outflow"
        save_single(name, folder_out)


# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()
