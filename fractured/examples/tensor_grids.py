import numpy as np


from gridding.fractured import meshing
from viz.exporter import export_vtk


def tensor_grid_2d():
    f_1 = np.array([[2, 6], [5, 5]])
    f_2 = np.array([[4, 4], [2, 7]])
    # f_3 = np.array([[4, 4], [5, 7]])
    f = [f_1,  f_2]

    gb = meshing.tensor_grid(f, [10, 10], physdims=[10, 10])

    export_vtk(gb, 'tensor_grid_2d')


def tensor_grid_3d():
    f_1 = np.array([[2, 5, 5, 2], [2, 2, 5, 5], [5, 5, 5, 5]])
    f_2 = np.array([[2, 2, 5, 5], [5, 5, 5, 5], [2, 5, 5, 2]])
    f = [f_1, f_2]
    bucket = meshing.tensor_grid(f, np.array([10, 10, 10]))
    export_vtk(bucket, 'tensor_grid_3d')


def tensor_grid_3d_complicated():
    f_1 = np.array([[2, 5, 5, 2], [2, 2, 5, 5], [5, 5, 5, 5]])
    f_2 = np.array([[2, 2, 5, 5], [5, 5, 5, 5], [2, 5, 5, 2]])
    f_3 = np.array([[4, 4, 4, 4], [1, 1, 8, 8], [1, 8, 8, 1]])
    f_4 = np.array([[3, 3, 6, 6], [3, 3, 3, 3], [3, 7, 7, 3]])
    f = [f_1, f_2, f_3, f_4]
    bucket = meshing.tensor_grid(f, np.array([8, 8, 8]))
    export_vtk(bucket, 'tensor_grid_3d')


if __name__ == "__main__":
    tensor_grid_2d()
    tensor_grid_3d()
    tensor_grid_3d_complicated()
