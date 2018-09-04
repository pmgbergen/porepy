import paraview.simple as pv

import vtk
from vtk.util.numpy_support import vtk_to_numpy

import csv
import numpy as np
from scipy.io import mmread

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

def read_csv(file_in, fields=None):

    # post-process the file by selecting only few columns
    if fields is not None:
        data = list(list() for _ in fields)
        with open(file_in, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            [d.append(row[f]) for row in reader for f, d in zip(fields, data)]
    else:
        with open(file_in, 'r') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)

    return data

#------------------------------------------------------------------------------#

def write_csv(file_out, fields, data):
    with open(file_out, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        #writer.writeheader()
        for dd in zip(*data):
            if np.isnan(np.array(dd)).any():
                print(dd)
            writer.writerow({f: d for f, d in zip(fields, dd)})

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

            # 1) matrix and grid information
            file_in = folder + "info.txt"
            data = read_csv(file_in)[0]
            data = map(int, map(float, data[:0:-1]))

            file_in = folder + "matrix.mtx"
            A = mmread(file_in)
            data.append(A.shape[0])
            data.append(A.nnz)

            with open(solver+"_results.csv", 'a+') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data)

            # 2) $\int_{\Omega_3,3} \porosity c \, \mathrm{d}x$ $([-])$ vs. time
            field = "tracer"
            step = 101

            transport_root = folder+"tracer_3_"

            # in this file the constant data are saved
            file_in = folder+"sol_3.vtu"
            fields = ["cell_volumes", "aperture", "bottom_domain"]
            phi = 0.25
            cot_matrix = phi * cot_domain(transport_root, file_in, step, field, fields)

            # 3) $\int_{\Omega_f} \epsilon \porosity c \, \mathrm{d}x$ $([-])$ vs. time

            transport_root = folder+"tracer_2_"

            # in this file the constant data are saved
            file_in = folder+"sol_2.vtu"
            fields = ["cell_volumes", "aperture"]
            phi = 0.4
            cot_fracture = phi * cot_domain(transport_root, file_in, step, field, fields)


            # 4) the integrated flux of c across the outlet boundary for each time step
            file_in = folder + "outflow.csv"
            outflow = read_csv(file_in)
            outflow = np.array(outflow, dtype=np.float).ravel()

            # )
            # collect the data in a single file
            file_out = folder+"dot_refinement_"+refinement+".csv"
            times = np.arange(step)*1e7

            data = [times, cot_matrix, cot_fracture, outflow]
            write_csv(file_out, ['time', 'cot_m', 'cot_f', 'outflow'], data)

            # 5) plot of the pressure head in the matrix, along
            #    (0, 100, 100)-(100, 0, 0)

            field_0 = "pressure"
            # file of the matrix
            file_in = folder+"sol_3.vtu"
            file_tmp = folder+"tmp0.csv"
            pts = [[0, 100, 100], [100, 0, 0]]

            plot_over_line(file_in, file_tmp, pts)
            data_0 = read_csv(file_tmp, ['arc_length', field_0])
            data_0 = np.array(data_0, dtype=np.float).T

            # 6) plot of $c$ in the matrix, at the final simulation time, along
            #    (0, 100, 100)-(100, 0, 0)

            field_1 = "tracer"
            # file of the matrix at final simulation time
            file_in = folder+"tracer_3_000100.vtu"
            file_tmp = folder+"tmp1.csv"
            pts = [[0, 100, 100], [100, 0, 0]]

            plot_over_line(file_in, file_tmp, pts)
            data_1 = read_csv(file_tmp, ['arc_length', field_1])
            data_1 = np.array(data_1, dtype=np.float).T

            # 7) plot of $c$ within the fracture at the final simulation time along
            #    (0, 100, 80)-(100, 0, 20)

            field_2 = "tracer"
            # file of the fracture at final simulation time
            file_in = folder+"tracer_2_000100.vtu"
            file_tmp = folder+"tmp2.csv"
            pts = [[0, 100, 80], [100, 0, 20]]

            plot_over_line(file_in, file_tmp, pts)
            data_2 = read_csv(file_tmp, ['arc_length', field_2])
            data_2 = np.array(data_2, dtype=np.float).T

            file_out = folder+"dol_refinement_"+refinement+".csv"
            data = np.asarray([data_0[:, 0], data_0[:, 1], data_1[:, 0],
                               data_1[:, 1], data_2[:, 0], data_2[:, 1]])

            with open(file_out, 'w') as csvfile:
                writer = csv.writer(csvfile)
                for row in data.T:
                    if np.isnan(row).any():
                        continue
                    writer.writerow(row)
