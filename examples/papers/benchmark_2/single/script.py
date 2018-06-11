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
import paraview.simple as pv

import vtk
from vtk.util.numpy_support import vtk_to_numpy

import csv
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

    # 1) $\int_{\Omega_f} \porosity c \, \mathrm{d}x$ $([-])$ vs. time
    field = "pressure" # should be c
    file_in = "./rt0_results/sol_3.vtu"

    vtk_reader = read_file(file_in)
    c = read_data(vtk_reader, field)
    cell_volumes = read_data(vtk_reader, "cell_volumes")
    #phi = read_data(vtk_reader, "phi")
    print(np.sum(c*cell_volumes))

    # 2) $\int_{\Omega_3} \porosity c \, \mathrm{d}x$ $([-])$ vs. time

    # 4) plot of the pressure head in the matrix, along
    #    (0, 100, 100)-(100, 0, 0)

    field = "pressure"
    # file of the matrix
    file_in = "./rt0_results/sol_3.vtu"
    file_out = "./rt0_results/pol.csv"
    pts = [[0, 100, 100], [100, 0, 0]]

    plot_over_line(file_in, file_out, pts)
    post_process(file_out, ['arc_length', field])

    # 5) plot of $c$ in the matrix, at the final simulation time, along
    #    (0, 100, 100)-(100, 0, 0)

    field = "c"
    # file of the matrix at final simulation time
    file_in = "./rt0_results/sol_3.vtu"
    file_out = "./rt0_results/pot_matrix.csv"
    pts = [[0, 100, 100], [100, 0, 0]]

    plot_over_line(file_in, file_out, pts)
    post_process(file_out, ['arc_length', field])

    # 6) plot of $c$ within the fracture at the final simulation time along
    #    (0, 100, 80)-(100, 0, 20)

    field = "c"
    # file of the fracture at final simulation time
    file_in = "./rt0_results/sol_2.vtu"
    file_out = "./rt0_results/pot_fracture.csv"
    pts = [[0, 100, 80], [100, 0, 20]]

    plot_over_line(file_in, file_out, pts)
    post_process(file_out, ['arc_length', field])
