import numpy as np
import scipy.sparse as sps
import pickle

from porepy.viz.exporter import Exporter

from porepy.numerics.fv import mpfa, fvutils, source

# import example_4_create_grid
import example_4_data

# ------------------------------------------------------------------------------#


def main(grid_name, direction):
    file_export = "solution"
    tol = 1e-4

    folder_grids = "/home/elle/Dropbox/Work/tipetut/"
    gb = pickle.load(open(folder_grids + grid_name, "rb"))

    folder_export = "./example_4_mpfa_" + grid_name + "_" + direction + "/"

    domain = {
        "xmin": -800,
        "xmax": 600,
        "ymin": 100,
        "ymax": 1500,
        "zmin": -100,
        "zmax": 1000,
    }

    example_4_data.add_data(gb, domain, direction, tol)

    # Choose and define the solvers and coupler
    solver_flux = mpfa.MpfaDFN(gb.dim_max(), "flow")
    A_flux, b_flux = solver_flux.matrix_rhs(gb)

    solver_source = source.IntegralDFN(gb.dim_max(), "flow")
    A_source, b_source = solver_source.matrix_rhs(gb)

    p = sps.linalg.spsolve(A_flux + A_source, b_flux + b_source)
    solver_flux.split(gb, "p", p)

    save = Exporter(gb, file_export, folder_export)
    save.write_vtk(["p"])

    # compute the flow rate
    fvutils.compute_discharges(gb, "flow")
    diam, flow_rate = example_4_data.compute_flow_rate(gb, direction, domain, tol)
    np.savetxt(folder_export + "flow_rate.txt", (diam, flow_rate))

    # compute the number of cells
    num_cells = gb.num_cells(lambda g: g.dim == 2)
    with open(folder_export + "cells.txt", "w") as f:
        f.write(str(num_cells))


# ------------------------------------------------------------------------------#

grids_name = [
    "gb_conf_24284.grid",
    "gb_conf_32614.grid",
    "gb_conf_42817.grid",
    "gb_conf_64215.grid",
    "gb_conf_79690.grid",
    "gb_conf_112371.grid",
    "gb_non_conf_19575.grid",
    "gb_non_conf_53603.grid",
    "gb_non_conf_60523.grid",
    "gb_non_conf_79308.grid",
    "gb_non_conf_118441.grid",
    "gb_non_conf_336205.grid",
]

for name in grids_name:
    main(name, "left_right")
    main(name, "bottom_top")
    main(name, "back_front")

# ------------------------------------------------------------------------------#
