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
#from scipy.io import mmread
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

def plot_over_line(file_in, file_out, pts, resolution=1000):

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

def read_csv(file_in, fields):
    # post-process the file by selecting only few columns
    data = list(list() for _ in fields)
    with open(file_in, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        [d.append(row[f]) for row in reader for f, d in zip(fields, data)]
    return data

#------------------------------------------------------------------------------#

def write_csv(file_out, fields, data):
    with open(file_out, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        #writer.writeheader()
        [writer.writerow({f: d for f, d in zip(fields, dd)}) for dd in zip(*data)]

#------------------------------------------------------------------------------#

def cot_domain(transport_root, file_in, step, field, fields, num_colors, padding=6):

    vtk_reader = read_file(file_in)
    weight = np.multiply.reduce([read_data(vtk_reader, f) for f in fields])
    color = read_data(vtk_reader, "color")

    cot = np.zeros((step, num_colors))
    for i in np.arange(step):
        file_in = transport_root+str(i).zfill(padding)+".vtu"
        vtk_reader = read_file(file_in)
        c = read_data(vtk_reader, field)
        for color_id in np.arange(num_colors):
            weight_loc = weight * (color == color_id).astype(np.float)
            cot[i, color_id] = np.sum(c*weight_loc)/np.sum(weight_loc)

    return cot

#------------------------------------------------------------------------------#

if __name__ == "__main__":

    solver_names = ['tpfa', 'vem', 'rt0', 'mpfa']
    cases = ['1', '2']
    indices = ['0', '1', '2', '3']

    #solver_names = ['vem'] ###
    cases = ['1']
    indices = ['0'] ###

    for case in cases:
        for idx in indices:
            for solver in solver_names:
                folder = "./" + solver + "_results_" + case + "_" + idx + "/"

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
                # 3) for each mesh and for each matrix permeability, the pressure over
                #    line from (0, 0, 0) to (1, 1, 1)

#                field = "pressure"
#                # file of both the matrix and the fracture
#                file_in = folder + "sol_3.vtu"
#                file_out = folder + "pol_matrix_"+ case + "_" + idx + ".csv"
#                pts = [[0, 0, 0], [1, 1, 1]]
#
#                plot_over_line(file_in, file_out, pts)
#                data = read_csv(file_out, ['arc_length', field])
#                write_csv(file_out, ['arc_length', field], data)

                # 4) for the coarsest mesh the averaged concentration on each matrix block

                transport_root = folder+"tracer_3_"
                file_in = folder+"sol_3.vtu"
                fields = ["cell_volumes"]
                field = "tracer"
                step = 101
                num_colors = 21
                cot_matrix = cot_domain(transport_root, file_in, step, field, fields, num_colors)

                # collect the data in a single file
                file_out = folder + "cot_" + case + "_" + idx + ".csv"
                times = np.arange(step)*0.25/100.
                labels = np.arange(num_colors).astype(np.str)
                labels = np.core.defchararray.add("cot_m_", labels)
                labels = np.insert(labels, 0, 'time')
                data = np.insert(cot_matrix, 0, times, axis=1).T
                write_csv(file_out, labels, data)

