import numpy as np
import matplotlib.pyplot as plt
import matplotlib

num_file = 20
folder_name = 'example_2_1_vem_coarse/'
file_name = '/plot_over_line.txt'
figure_name = 'plot_over_line_vem_coarse.pdf' # pdf

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)

cm = plt.get_cmap('gist_rainbow')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('arc length')
ax.set_ylabel('$p$')
ax.set_prop_cycle('color', plt.cm.copper(np.linspace(0,1,num_file,endpoint=False)))

for f in np.arange(1, num_file+1):
    f_name = folder_name + str(f) + file_name
    data = np.loadtxt(f_name, delimiter=' ', unpack=True)
    ax.plot(data[:, 0], data[:, 1])

#ax.legend()
plt.show()
fig.savefig(figure_name, bbox_inches='tight')
