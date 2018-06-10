# How to use: generally in your paraview folder you have a sub-folder called bin
# inside you have an executable called pvpython, which is a python interpreter
# with paraview library. Use it to call this script.
#
# You can run the code as
# ${PARAVIEW_BIN}/pvpython input_file.pvd output_file.csv pressure_field_name
#
# example
# ${PARAVIEW_BIN}/pvpython ./vem/result.pvd ./vem/pol.csv pressure

#### import the simple module from the paraview
from paraview.simple import *
import csv, sys

#------------------------------------------------------------------------------#

def plot_over_line(input_file, output_file):

    if input_file.lower().endswith('.pvd'):
        # create a new 'PVD Reader'
        sol = PVDReader(FileName=input_file)
    elif input_file.lower().endswith('.vtu'):
        # create a new 'XML Unstructured Grid Reader'
        sol = XMLUnstructuredGridReader(FileName=input_file)
    else:
        raise ValueError, "file format not yet supported"

    # create a new 'Plot Over Line'
    pol = PlotOverLine(Input=sol, Source='High Resolution Line Source')

    # Properties modified on plotOverLine1.Source
    pol.Source.Point1 = [0.0, 100.0, 100.0]
    pol.Source.Point2 = [100.0, 0.0, 0.0]
    pol.Source.Resolution = 50000

    # save data
    SaveData(output_file, proxy=pol, Precision=15)

#------------------------------------------------------------------------------#

def post_process(output_file, fields):
    # post-process the file by selecting only few columns
    data = list(list() for _ in fields)
    with open(output_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        [d.append(row[f]) for row in reader for f, d in zip(fields, data)]

    with open(output_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        [writer.writerow({f: d for f, d in zip(fields, dd)}) for dd in zip(*data)]

#------------------------------------------------------------------------------#

if __name__ == "__main__":

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    pressure_field_name = sys.argv[3]

    data = plot_over_line(input_file, output_file)

    fields = ['arc_length', pressure_field_name]
    post_process(output_file, fields)
