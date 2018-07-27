import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages


file_name = ["example_5_3/production.txt", "example_5_3_coarse/production.txt"]
figure_name = "production.pdf"

color = ["b", "r"]
legend = ["tetrahedra", "coarse"]
year = np.pi * 1e7
scale = 7.3

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=15)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("year")
ax.set_ylabel("production")

for f, c, l in zip(file_name, color, legend):
    data = np.loadtxt(f, delimiter=",", unpack=True)
    ax.plot(data[:, 0] / year, data[:, 1] / scale, color=c, label=l)

ax.legend()
plt.show()
# plt.savefig(figure_name)

with PdfPages(figure_name) as pdf:
    pdf.savefig(fig)
