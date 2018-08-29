import paraview.simple as pv

import vtk
from vtk.util.numpy_support import vtk_to_numpy

import csv
import numpy as np

#------------------------------------------------------------------------------#

def plot_over_line(file_in, file_out, pts, resolution=2000):

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

if __name__ == "__main__":

    solver_names = ['tpfa', 'vem', 'rt0', 'mpfa']

    for solver in solver_names:
        folder = "./"+solver+"_results/"

        # plot of the pressure head in the matrix, along

        # 3) (0, 100, 100)-(100, 0, 0)

        field = "pressure"
        # file of the matrix
        file_in = folder+"sol_3.vtu"

        file_out = folder+"dol_0.csv"
        pts = [[350, 100, -100], [-500, 1500, 500]]

        plot_over_line(file_in, file_out, pts)
        data = read_csv(file_out, ['arc_length', field])
        write_csv(file_out, ['arc_length', field], data)

        # 4) (0, 100, 100)-(100, 0, 0)

        file_out = folder+"dol_1.csv"
        pts = [[-500, 100, -100], [350, 1500, 500]]

        plot_over_line(file_in, file_out, pts)
        data = read_csv(file_out, ['arc_length', field])
        write_csv(file_out, ['arc_length', field], data)

