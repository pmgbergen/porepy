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

    num_simul = 5

    n_step = 300
    num_frac = 3

    num_cells = np.zeros((num_simul, num_frac+1))

    for simul in np.arange(1, num_simul+1):

        folder_in = "./plot/solution_" + str(simul) + "/"
        folder_out = folder_in + "img/"

        data = folder_in + "dot_avg.csv"
        label = "LABEL"
        title = "average cot"
        plot_multiple(data, label, title, num_frac)

        name = "cot_avg"
        save_multiple(name, num_frac, folder_out)

        ###########

        data = folder_in + "dot_min.csv"
        label = "LABEL"
        title = "min cot"
        plot_multiple(data, label, title, num_frac)

        name = "cot_min"
        save_multiple(name, num_frac, folder_out)

        ###########

        data = folder_in + "dot_max.csv"
        label = "LABEL"
        title = "max cot"
        plot_multiple(data, label, title, num_frac)

        name = "cot_max"
        save_multiple(name, num_frac, folder_out)

        ###########

        data = folder_in + "outflow.csv"
        label = "LABEL"
        title = "outflow"
        plot_single(data, label, title)

        name = "outflow"
        save_single(name, folder_out)

        ########

        data = folder_in + "num_cells.csv"
        num_cells[simul-1] = np.loadtxt(data, delimiter=',')

    label = "LABEL"
    title = "number of cells"
    plot_num_cells(num_cells, label, title)

    name = "num_cells"
    folder_out = "./plot/"
    save_multiple(name, num_frac, folder_out)
    save_single(name, folder_out, num_frac)

#------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()
