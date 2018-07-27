import numpy as np
import scipy.sparse as sps
import pickle

from porepy.viz.exporter import Exporter

from porepy.grids.grid import FaceTag

# from porepy.grids import coarsening as co

from porepy.numerics.vem import vem_dual, vem_source

# import example_4_create_grid
import example_4_data

# ------------------------------------------------------------------------------#


def main(grid_name, direction):
    file_export = "solution"
    tol = 1e-4

    folder_grids = "/home/elle/Dropbox/Work/tipetut/"
    gb = pickle.load(open(folder_grids + grid_name, "rb"))

    folder_export = "./example_4_vem_" + grid_name + "_" + direction + "/"

    domain = {
        "xmin": -800,
        "xmax": 600,
        "ymin": 100,
        "ymax": 1500,
        "zmin": -100,
        "zmax": 1000,
    }

    internal_flag = FaceTag.FRACTURE
    [g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]

    example_4_data.add_data(gb, domain, direction, tol)

    # Choose and define the solvers and coupler
    solver_flow = vem_dual.DualVEMDFN(gb.dim_max(), "flow")
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = vem_source.IntegralDFN(gb.dim_max(), "flow")
    A_source, b_source = solver_source.matrix_rhs(gb)

    up = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)
    solver_flow.split(gb, "up", up)

    gb.add_node_props(["discharge", "p", "P0u"])
    solver_flow.extract_u(gb, "up", "discharge")
    solver_flow.extract_p(gb, "up", "p")
    solver_flow.project_u(gb, "discharge", "P0u")

    save = Exporter(gb, file_export, folder_export)
    save.write_vtk(["p", "P0u"])

    # compute the flow rate
    diam, flow_rate = example_4_data.compute_flow_rate_vem(gb, direction, domain, tol)
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
