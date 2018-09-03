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

def cot_domain(transport_root, file_in, step, field, fields, padding=6):

    vtk_reader = read_file(file_in)
    weight = np.multiply.reduce([read_data(vtk_reader, f) for f in fields])

    cot = np.zeros(step)
    for i in np.arange(step):
        file_in = transport_root+str(i).zfill(padding)+".vtu"
        vtk_reader = read_file(file_in)
        c = read_data(vtk_reader, field)
        cot[i] = np.sum(c*weight)

    return cot

#------------------------------------------------------------------------------#

if __name__ == "__main__":

    solver_names = ['tpfa', 'vem', 'rt0', 'mpfa']
    refinements = ['0', '1', '2']

    for refinement in refinements:
        for solver in solver_names:
            folder = "./"+solver+"_results_"+refinement+"/"

            # 2) $\int_{\Omega_3,3} \porosity c \, \mathrm{d}x$ $([-])$ vs. time
            field = "tracer"
            step = 101

            transport_root = folder+"tracer_3_"

            # in this file the constant data are saved
            file_in = folder+"sol_3.vtu"
            fields = ["phi", "cell_volumes", "aperture", "bottom_domain"]
            cot_matrix = cot_domain(transport_root, file_in, step, field, fields)

            # 1) $\int_{\Omega_f} \epsilon \porosity c \, \mathrm{d}x$ $([-])$ vs. time

            transport_root = folder+"tracer_2_"

            # in this file the constant data are saved
            file_in = folder+"sol_2.vtu"
            fields = ["phi", "cell_volumes", "aperture"]
            cot_fracture = cot_domain(transport_root, file_in, step, field, fields)

            # 3)
            # collect the data in a single file
            file_out = folder+"dot_"+refinement+".csv"
            times = np.arange(step)*1e7

            write_csv(file_out, ['time', 'cot_m', 'cot_f'], [times, cot_matrix, cot_fracture])

            # 4) plot of the pressure head in the matrix, along
            #    (0, 100, 100)-(100, 0, 0)

            field_0 = "pressure"
            # file of the matrix
            file_in = folder+"sol_3.vtu"
            file_tmp = folder+"tmp0.csv"
            pts = [[0, 100, 100], [100, 0, 0]]

            plot_over_line(file_in, file_tmp, pts)
            data_0 = read_csv(file_tmp, ['arc_length', field_0])

            # 5) plot of $c$ in the matrix, at the final simulation time, along
            #    (0, 100, 100)-(100, 0, 0)

            field_1 = "tracer"
            # file of the matrix at final simulation time
            file_in = folder+"tracer_3_000100.vtu"
            file_tmp = folder+"tmp1.csv"
            pts = [[0, 100, 100], [100, 0, 0]]

            plot_over_line(file_in, file_tmp, pts)
            data_1 = read_csv(file_tmp, ['arc_length', field_1])

            # 6) plot of $c$ within the fracture at the final simulation time along
            #    (0, 100, 80)-(100, 0, 20)

            field_2 = "tracer"
            # file of the fracture at final simulation time
            file_in = folder+"tracer_2_000100.vtu"
            file_tmp = folder+"tmp2.csv"
            pts = [[0, 100, 80], [100, 0, 20]]

            plot_over_line(file_in, file_tmp, pts)
            data_2 = read_csv(file_tmp, ['arc_length', field_2])

            file_out = folder+"dol_refinement_"+refinement+".csv"
            write_csv(file_out,
                    ['arc_length', field_0, 'arc_length', field_1, 'arc_length', field_2],
                    [data_0, data_1, data_2])
