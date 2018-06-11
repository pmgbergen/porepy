# How to use: generally in your paraview folder you have a sub-folder called bin
# inside you have an executable called pvpython, which is a python interpreter
# with paraview library. Use it to call this script.
#
# You can run the code as
# ${PARAVIEW_BIN}/pvpython file_in.pvd file_out.csv pressure_field_name
#
# example
# ${PARAVIEW_BIN}/pvpython ./vem/result.pvd ./vem/pol.csv pressure

#### import the simple module from the paraview
from __future__ import print_function

import paraview.simple as pv

import vtk
from vtk.util.numpy_support import vtk_to_numpy

import csv
from scipy.io import mmread
import numpy as np

#------------------------------------------------------------------------------#

def read_file(file_in):
    vtk_reader = vtk.vtkXMLUnstructuredGridReader()
    vtk_reader.SetFileName(file_in)
    vtk_reader.Update()
    return vtk_reader

#------------------------------------------------------------------------------#

def read_data(vtk_reader, field):
    data = vtk_reader.GetOutput().GetCellData().GetArray(field)
    return vtk_to_numpy(data)

#------------------------------------------------------------------------------#

def color(cell_centers):
    val = np.zeros(cell_centers.shape[1], dtype=np.int)
    x = cell_centers[0, :]
    y = cell_centers[1, :]
    z = cell_centers[2, :]

    val[np.logical_and.reduce((x>.5, y<.5, z<.5))] = 0
    val[np.logical_and.reduce((x<.5, y>.5, z<.5))] = 1
    val[np.logical_and.reduce((x>.5, y>.5, z<.5))] = 2
    val[np.logical_and.reduce((x<.5, y<.5, z>.5))] = 3
    val[np.logical_and.reduce((x>.5, y<.5, z>.5))] = 4
    val[np.logical_and.reduce((x<.5, y>.5, z>.5))] = 5

    val[np.logical_and.reduce((x>.75, y>.75, z>.75))] = 6
    val[np.logical_and.reduce((x>.75, y>.5, y<.75, z>.75))] = 7
    val[np.logical_and.reduce((x>.5, x<.75, y>.75, z>.75))] = 8
    val[np.logical_and.reduce((x>.5, x<.75, y>.5, y<.75, z>.75))] = 9
    val[np.logical_and.reduce((x>.75, y>.75, z>.5, z<.75))] = 10
    val[np.logical_and.reduce((x>.75, y>.5, y<.75, z>.5, z<.75))] = 11
    val[np.logical_and.reduce((x>.5, x<.75, y>.75, z>.5, z<.75))] = 12

    val[np.logical_and.reduce((x>.5, x<.625, y>.5, y<.625, z>.5, z<.625))] = 13
    val[np.logical_and.reduce((x>.625, x<.75, y>.5, y<.625, z>.5, z<.625))] = 14
    val[np.logical_and.reduce((x>.5, x<.625, y>.625, y<.75, z>.5, z<.625))] = 15
    val[np.logical_and.reduce((x>.625, x<.75, y>.625, y<.75, z>.5, z<.625))] = 16
    val[np.logical_and.reduce((x>.5, x<.625, y>.5, y<.625, z>.625, z<.75))] = 17
    val[np.logical_and.reduce((x>.625, x<.75, y>.5, y<.625, z>.625, z<.75))] = 18
    val[np.logical_and.reduce((x>.5, x<.625, y>.625, y<.75, z>.625, z<.75))] = 19
    val[np.logical_and.reduce((x>.625, x<.75, y>.625, y<.75, z>.625, z<.75))] = 20

    return val

#------------------------------------------------------------------------------#

def pot_block(field, file_in):

    vtk_reader = read_file(file_in)
    c = read_data(vtk_reader, field)
    cell_volumes = read_data(vtk_reader, "cell_volumes")
    cell_centers = read_data(vtk_reader, "cell_centers").T
    region_id = color(cell_centers)

    # we need to add a temporal loop
    c_av = np.zeros(21)
    for r_id in np.arange(21):
        mask = region_id == r_id
        c_av[r_id] = np.sum(c[mask]*cell_volumes[mask])/np.sum(cell_volumes[mask])

#------------------------------------------------------------------------------#

def plot_over_line(file_in, file_out, pts, resolution=50000):

    if file_in.lower().endswith('.pvd'):
        # create a new 'PVD Reader'
        sol = pv.PVDReader(FileName=file_in)
    elif file_in.lower().endswith('.vtu'):
        # create a new 'XML Unstructured Grid Reader'
        sol = pv.XMLUnstructuredGridReader(FileName=file_in)
    else:
        raise ValueError, "file format not yet supported"

    # create a new 'Plot Over Line'
    pol = pv.PlotOverLine(Input=sol, Source='High Resolution Line Source')

    # Properties modified on plotOverLine1.Source
    pol.Source.Point1 = pts[0]
    pol.Source.Point2 = pts[1]
    pol.Source.Resolution = resolution

    # save data
    pv.SaveData(file_out, proxy=pol, Precision=15)

#------------------------------------------------------------------------------#

def post_process(file_out, fields):
    # post-process the file by selecting only few columns
    data = list(list() for _ in fields)
    with open(file_out, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        [d.append(row[f]) for row in reader for f, d in zip(fields, data)]

    with open(file_out, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        [writer.writerow({f: d for f, d in zip(fields, dd)}) for dd in zip(*data)]

#------------------------------------------------------------------------------#

if __name__ == "__main__":

#    # 2) matrix information
#    file_in = "matrix.mtx"
#    A = mmread(file_in)
#
#    print("dof ", A.shape)
#    print("nnz ", A.nnz)
#    print("nnz/dof^2 ", A.nnz/float(A.shape[0]**2))
#
#    # for the condest use matlab/octave, no available option in scipy
#
#    # 3) for each mesh and for each matrix permeability, the pressure over
#    #    line from (0, 0, 0) to (1, 1, 1)
#
#    field = "pressure"
#    # file of both the matrix and the fracture
#    file_in = "./rt0_results/sol.pvd"
#    file_out = "./rt0_results/pol.csv"
#    pts = [[0, 0, 0], [1, 1, 1]]
#
#    plot_over_line(file_in, file_out, pts)
#    post_process(file_out, ['arc_length', field])

    # 4) for the coarsest mesh the averaged concentration on each matrix block
    field = "pressure" # should be c
    file_in = "./rt0_results_2_0/sol_3.vtu"

    pot_block(field, file_in)
