import matplotlib.pyplot as plt
import numpy as np
import os

#------------------------------------------------------------------------------#

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)

def plot_single(file_name, legend, title):

    data = np.loadtxt(file_name, delimiter=',')

    plt.figure(0)
    plt.plot(data[:, 0], data[:, 1], label=legend)
    plt.title(title)
    plt.xlabel('$t$')
    plt.ylabel('$c$')
    plt.grid(True)
    plt.legend()

#------------------------------------------------------------------------------#

def plot_multiple(file_name, legend, title, num_frac):

    data = np.loadtxt(file_name, delimiter=',')

    for frac_id in np.arange(num_frac):
        plt.figure(frac_id)
        plt.plot(data[:, 0], data[:, frac_id+1], label=legend)
        plt.title(title + " - " + str(frac_id))
        plt.xlabel('$t$')
        plt.ylabel('$c$')
        plt.grid(True)
        plt.legend()

#------------------------------------------------------------------------------#

def plot_num_cells(data, legend, title):

    for frac_id, cells in enumerate(data.T[:-1]):
        plt.figure(frac_id)
        plt.plot(np.arange(cells.size), cells, label=legend)
        plt.title(title + " - " + str(frac_id))
        plt.xlabel('simulation')
        plt.ylabel('number of cells')
        plt.grid(True)
        plt.legend()

    plt.figure(frac_id+1)
    plt.plot(np.arange(cells.size), data.T[-1], label=legend)
    plt.title(title + " - total")
    plt.xlabel('simulation')
    plt.ylabel('number of cells')
    plt.grid(True)
    plt.legend()

#------------------------------------------------------------------------------#

def save_single(filename, folder, figure_id=0):

    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.figure(figure_id)
    plt.savefig(folder+filename, bbox_inches='tight')
    plt.gcf().clear()

#------------------------------------------------------------------------------#

def save_multiple(filename, num_frac, folder):

    if not os.path.exists(folder):
        os.makedirs(folder)

    for frac_id in np.arange(num_frac):
        plt.figure(frac_id)
        name = filename + "_frac_" + str(frac_id)
        plt.savefig(folder+name, bbox_inches='tight')
        plt.gcf().clear()

#------------------------------------------------------------------------------#

def main():

    num_simul = 3

    n_step = 300
    num_frac = 3

    num_cells = np.zeros((num_simul, num_frac+1))
    master_folder = "/home/elle/Dropbox/Work/PresentazioniArticoli/2019/Articles/tipetut++/Results/example1/"

    methods_stefano = ["OPTxfem"]
    methods_alessio = ["MVEM", "Mpfa", "Tpfa"]
    grids = {"grid_0": ("1k", "220"), "grid_1": ("3k", "650"), "grid_2": ("10k", "2100")}

    for grid_name, grid in grids.items():
        for simul in np.arange(num_simul):

            folder_in = master_folder
            folder_out = folder_in + "img/"

            title = "average concentration over time - " + grid_name.replace("_", " ")
            # Alessio
            for method in methods_alessio:
                data = folder_in + method + "/" + method + "_Cmean_" + str(simul+1) + "_" + grid[0] + ".csv"
                plot_multiple(data, method, title, num_frac)

            # Stefano
            for method in methods_stefano:
                data = folder_in + method + "/" + method + "_Cmean_" + str(simul+1) + "_" + grid[1] + ".csv"
                plot_multiple(data, method, title, num_frac)

            # save
            name = grid_name + "_cot_avg_" + str(simul)
            save_multiple(name, num_frac, folder_out)

            ###########

            title = "minimum concentration over time - " + grid_name.replace("_", " ")
            # Alessio
            for method in methods_alessio:
                data = folder_in + method + "/" + method + "_Cmin_" + str(simul+1) + "_" + grid[0] + ".csv"
                plot_multiple(data, method, title, num_frac)

            # Stefano
            for method in methods_stefano:
                data = folder_in + method + "/" + method + "_Cmin_" + str(simul+1) + "_" + grid[1] + ".csv"
                plot_multiple(data, method, title, num_frac)

            # save
            name = grid_name + "_cot_min_" + str(simul)
            save_multiple(name, num_frac, folder_out)

            ###########

            title = "maximum concentration over time - " + grid_name.replace("_", " ")
            # Alessio
            for method in methods_alessio:
                data = folder_in + method + "/" + method + "_Cmax_" + str(simul+1) + "_" + grid[0] + ".csv"
                plot_multiple(data, method, title, num_frac)

            # Stefano
            for method in methods_stefano:
                data = folder_in + method + "/" + method + "_Cmax_" + str(simul+1) + "_" + grid[1] + ".csv"
                plot_multiple(data, method, title, num_frac)

            # save
            name = grid_name + "_cot_max_" + str(simul)
            save_multiple(name, num_frac, folder_out)

            ###########

            title = "outflow over time - " + grid_name.replace("_", " ")
            # Alessio
            for method in methods_alessio:
                data = folder_in + method + "/" + method + "_production_" + str(simul+1) + "_" + grid[0] + ".csv"
                plot_single(data, method, title)

            # Stefano
            for method in methods_stefano:
                data = folder_in + method + "/" + method + "_production_" + str(simul+1) + "_" + grid[1] + ".csv"
                plot_single(data, method, title)

            # save
            name = grid_name + "_outflow_" + str(simul)
            save_single(name, folder_out)

            ########

            #data = folder_in + "num_cells.csv"
            #num_cells[simul-1] = np.loadtxt(data, delimiter=',')
#
#    label = "LABEL"
#    title = "number of cells"
#    plot_num_cells(num_cells, label, title)
#
#    name = "num_cells"
#    folder_out = "./plot/"
#    save_multiple(name, num_frac, folder_out)
#    save_single(name, folder_out, num_frac)

#------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()
