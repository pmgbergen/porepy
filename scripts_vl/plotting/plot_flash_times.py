import matplotlib.pyplot as plt
import numpy as np

labels = [
    'Point-wise',
    'Python\nmultiprocessing',
    'numba-compiled\nparallelization',
]

minimize_times = np.array([5760, 1204.0572671890259, 0.10272908210754395])
init_times = np.array([0., 0., 0.005086660385131836])
total_times = init_times + minimize_times
minimize_times[:2] = 0.


data = [(x,y, z) for x, y, z in zip(init_times, minimize_times, total_times)]

dim = len(data[0])
w = 0.75
dimw = w / dim

fig, ax = plt.subplots()
x = np.arange(len(data))
bars = list()
for i in range(len(data[0])):
    y = [d[i] for d in data]
    b = ax.bar(x + i * dimw, y, dimw, bottom=0.001)
    bars.append(b)

ax.legend(bars, ['initialization', 'minimization', 'total time'], loc="upper right")
ax.set_xticks(x + dimw / 2, labels=map(str, x))
ax.set_xticklabels(labels)

xticks = x + dimw / 2
ax.text(xticks[0] , total_times[0] + 50, f"~{np.ceil(total_times[0])}")
ax.text(xticks[1] , total_times[1] + 10, f"~{np.ceil(total_times[1])}")
ax.text(xticks[2] - 0.25, total_times[2] + 10, f"{0.005} + {0.103}")

ax.set_yscale('log')
ax.set_title("Comparison of computational times\n6400 p-T flash problems\n2 phases, 2 components")
ax.set_xlabel('Modi of computation')
ax.set_ylabel('Time [s]')

fig.show()
fig.tight_layout()

print('done')