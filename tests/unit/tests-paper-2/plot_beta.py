import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator


import os
import pdb


os.system("clear")

# times = np.around(np.loadtxt("./case_1/horizontal_hu_Kn0.1/BETA/BETA_TIME"), decimals=6)
# times = np.around(np.loadtxt("./case_1/vertical_hu_Kn10.0/BETA/BETA_TIME"), decimals=6)
# times = np.around(np.loadtxt("./case_1/slanted_hu_Kn0.1/BETA/BETA_TIME"), decimals=6)
# times = np.around(
#     np.loadtxt("./case_1/slanted_hu_Kn0.1/non-conforming/BETA/BETA_TIME"), decimals=6
# )
# times = np.around(np.loadtxt("./case_1/horizontal_hu_beta/BETA/BETA_TIME"), decimals=6)
times = np.around(np.loadtxt("./case_2/hu/BETA/BETA_TIME"), decimals=6)


for time in times:
    print("\ntime = ", time, " ----------------------")

    #####################################################
    # root_name = "./case_1/horizontal_hu_Kn0.1/BETA"
    # root_name = "./case_1/vertical_hu_Kn10.0/BETA"
    # root_name = "./case_1/slanted_hu_Kn0.1/BETA"
    # root_name = "./case_1/slanted_hu_Kn0.1/non-conforming/BETA"
    # root_name = "./case_1/horizontal_hu_beta/BETA"
    root_name = "./case_2/hu/BETA"

    output_file = root_name + "/BETA_" + str(time)

    fontsize = 28
    my_orange = "darkorange"
    my_blu = [0.1, 0.1, 0.8]

    x_ticks = np.array([0, 2, 4, 6, 8, 10])

    save_folder = root_name

    #####################################################

    info = np.loadtxt(output_file)
    time = info[0][0]
    x = info[:, 1]
    y = info[:, 2]
    beta = info[:, 3]
    delta_potential = info[:, 4]

    x_plot = np.linspace(min(x), max(x), 200)
    y_plot = np.linspace(min(y), max(y), 200)
    X, Y = np.meshgrid(x_plot, x_plot)

    # beta: --------------------
    interp_1 = LinearNDInterpolator(list(zip(x, y)), beta)
    Z_1 = interp_1(X, Y)
    plt.pcolormesh(X, Y, Z_1, shading="auto", vmin=0.0, vmax=1.0)
    plt.plot(x, y, "ok", label="input point", markersize=1)
    # plt.legend()
    plt.colorbar()
    plt.axis("equal")
    # plt.show()

    plt.savefig(
        save_folder + "/beta_" + str(time) + ".pdf",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.2,
    )

    plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.scatter(x, y, beta, marker="o")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("beta")
    # plt.show()

    # # delta_potential: --------------
    # interp_2 = LinearNDInterpolator(list(zip(x, y)), delta_potential)
    # Z_2 = interp_2(X, Y)
    # plt.pcolormesh(X, Y, Z_2, shading="auto", vmin=0.0, vmax=1.0)
    # plt.plot(x, y, "ok", label="input point", markersize=1)
    # # plt.legend()
    # plt.colorbar()
    # plt.axis("equal")
    # # plt.show()

    # plt.savefig(
    #     save_folder + "/delta_potential.pdf",
    #     dpi=150,
    #     bbox_inches="tight",
    #     pad_inches=0.2,
    # )

    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.scatter(x, y, beta, marker="o")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("delta_potential")
    # plt.show()

    # pdb.set_trace()


print("\nDone!")
