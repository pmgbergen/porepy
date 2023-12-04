import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator


import os
import pdb


os.system("clear")

times = np.around(np.loadtxt("./case_1/slanted_hu_Kn10.0/BETA/BETA_TIME"), decimals=6)

for time in times:
    print("\ntime = ", time, " ----------------------")

    #####################################################
    output_file = "./case_1/slanted_hu_Kn10.0/BETA/BETA_" + str(time)

    fontsize = 28
    my_orange = "darkorange"
    my_blu = [0.1, 0.1, 0.8]

    x_ticks = np.array([0, 2, 4, 6, 8, 10])

    save_folder = "./case_1"

    #####################################################

    info = np.loadtxt(output_file)
    time = info[0][0]
    x = info[:, 1]
    y = info[:, 2]
    beta = info[:, 3]

    x_plot = np.linspace(min(x), max(x), 200)
    y_plot = np.linspace(min(y), max(y), 200)
    X, Y = np.meshgrid(x_plot, x_plot)
    interp = LinearNDInterpolator(list(zip(x, y)), beta)
    Z = interp(X, Y)
    plt.pcolormesh(X, Y, Z, shading="auto", vmin=0, vmax=1)
    plt.plot(x, y, "ok", label="input point", markersize=1)
    # plt.legend()
    plt.colorbar()
    plt.axis("equal")
    plt.show()


print("\nDone!")
