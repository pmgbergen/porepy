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

def cot_domain(file_in, step, field, num_frac, padding=6):

    cot_avg = np.zeros((step, num_frac))
    cot_min = np.zeros((step, num_frac))
    cot_max = np.zeros((step, num_frac))

    for i in np.arange(step):

        ifile = file_in+str(i).zfill(padding)+".vtu"
        vtk_reader = read_file(ifile)

        weight = read_data(vtk_reader, "cell_volumes")
        frac_num = read_data(vtk_reader, "frac_num")
        c = read_data(vtk_reader, field)

        for frac_id in np.arange(num_frac):
            is_loc = frac_num == frac_id
            weight_loc = weight[is_loc]

            cot_avg[i, frac_id] = np.sum(c[is_loc]*weight_loc)/np.sum(weight_loc)
            cot_min[i, frac_id] = np.amin(c[is_loc])
            cot_max[i, frac_id] = np.amax(c[is_loc])

    return cot_avg, cot_min, cot_max

#------------------------------------------------------------------------------#

if __name__ == "__main__":

    folder = "./solution/"
    field = "scalar"
    step = 1000
    num_frac = 10

    # in this file the constant data are saved
    file_in = folder+"solution_2_"

    cot_avg, cot_min, cot_max = cot_domain(file_in, step, field, num_frac)

    times = np.arange(step)
    labels = np.arange(num_frac).astype(np.str)
    labels = np.core.defchararray.add("cot_", labels)
    labels = np.insert(labels, 0, 'time')

    # create the output files
    file_out = folder + "dot_min.csv"
    data = np.insert(cot_min, 0, times, axis=1).T
    write_csv(file_out, labels, data)

    # create the output files
    file_out = folder + "dot_max.csv"
    data = np.insert(cot_max, 0, times, axis=1).T
    write_csv(file_out, labels, data)

    # create the output files
    file_out = folder + "dot_avg.csv"
    data = np.insert(cot_avg, 0, times, axis=1).T
    write_csv(file_out, labels, data)

