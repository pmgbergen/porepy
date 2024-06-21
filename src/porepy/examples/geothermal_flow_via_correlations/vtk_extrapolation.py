import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from DriesnerBrineOBL import DriesnerBrineOBL

file_name = "sample_with_gradients.vtk"


def f(x, y, z):
    xs = 0.75 * x**4 - x**2 + 0.3
    ys = 0.75 * y**4 - y**2 + 0.3
    zs = 0.75 * z**4 - z**2 + 0.3
    return xs * ys * zs
    # return np.sin(0.25 * np.pi * x)*np.sin(0.25 * np.pi * y)* (z-1.0)


# Create the 3D NumPy array of spatially referenced data again.
nc = 100
nc_x = nc
nc_y = nc
nc_z = nc
dx = float(2.0 / nc_x)
dy = float(2.0 / nc_y)
dz = float(2.0 / nc_z)
x = np.linspace(-1, 1.0, nc_x + 1)
y = np.linspace(-1, 1.0, nc_y + 1)
z = np.linspace(-1, 1.0, nc_z + 1)
xv, yv, zv = np.meshgrid(x, y, z)


grid = pv.ImageData()
grid.dimensions = xv.shape

# Edit the spatial reference
grid.origin = (-1.0, -1.0, -1.0)  # The bottom left corner of the data set
grid.spacing = (dx, dy, dz)  # These are the cell sizes along each axis
grid.point_data["f"] = f(xv, yv, zv).flatten(order="F")  # Flatten the array!


fields = grid.array_names
gradients = {}
for field in fields:
    grad_field = grid.compute_derivative(field, gradient=True)
    gradients[field] = grad_field["gradient"]

for field in fields:
    grid.point_data.set_vectors(gradients[field], "grad_" + field)
grid.save(file_name, binary=True)

file_name = file_name
sampled_obj = DriesnerBrineOBL(file_name)

# sample and plot
dxi = 0.001
xi = np.arange(-2.0, 2.0 + dxi, dxi)
xv = xi
yv = xi
zv = 0.0 * np.ones_like(xi)

# p = np.arange(1.0e6, 20.0e6, 0.5e6)
# h = 3.2e6 * np.ones_like(p)
par_points = np.array((xv, yv, zv)).T
sampled_obj.sample_at(par_points)

fv = sampled_obj.sampled_could.point_data["f"]
plt.plot(
    xv,
    fv,
    label="f",
    color="blue",
    linestyle="-",
    marker="o",
    markerfacecolor="blue",
    markersize=5,
)
plt.show()
