import numpy as np
import matplotlib.pyplot as plt
import matplotlib

num_file = 44
# folder_name = ['example_2_1_mpfa/', 'example_2_1_vem/', 'example_2_1_vem_coarse/']
folder_name = ["example_2_2_mpfa/", "example_2_2_vem_coarse/"]
labels = ["MPFA", "VEM", "VEM-coarse"]
file_name = "/flow_rate.txt"
figure_name = "flow_rate.pdf"  # pdf

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=15)

cm = plt.get_cmap("gist_rainbow")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("simu")
ax.set_ylabel("flow rate")
folder_name = np.array(folder_name)
color_setup = plt.cm.copper(np.linspace(0, 1, folder_name.size, endpoint=False))
ax.set_prop_cycle("color", color_setup)

for folder, label in zip(folder_name, labels):
    data_to_plot = np.empty(num_file)

    for f in np.arange(1, num_file + 1):
        f_name = folder + str(f) + file_name
        data_to_plot[f - 1] = np.loadtxt(f_name, delimiter=" ", unpack=True)[1]
        if np.isnan(data_to_plot[f - 1]):
            print(f_name, data_to_plot[f - 1])

    ax.plot(np.arange(num_file), data_to_plot, label=label)

ax.legend()
plt.show()
fig.savefig(figure_name, bbox_inches="tight")
