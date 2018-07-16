import numpy as np
import scipy.sparse as sps

from porepy.viz.exporter import Exporter

from porepy.grids.grid import FaceTag

from porepy.numerics.vem import vem_dual, vem_source

import example_2_3_create_grid
import example_2_3_data

# ------------------------------------------------------------------------------#


def main(id_problem, tol=1e-5, N_pts=1000, if_export=False):

    mesh_size = 0.025  # 0.01 0.05
    folder_export = (
        "example_2_3_vem_coarse_" + str(mesh_size) + "/" + str(id_problem) + "/"
    )
    file_export = "vem"

    gb = example_2_3_create_grid.create(
        id_problem, is_coarse=True, mesh_size=mesh_size, tol=tol
    )

    internal_flag = FaceTag.FRACTURE
    [g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]

    # Assign parameters
    example_2_3_data.add_data(gb, tol)

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

    if if_export:
        save = Exporter(gb, file_export, folder_export)
        save.write_vtk(["p", "P0u"])

    b_box = gb.bounding_box()
    y_range = np.linspace(b_box[0][1] + tol, b_box[1][1] - tol, N_pts)
    pts = np.stack((0.35 * np.ones(N_pts), y_range, np.zeros(N_pts)))
    values = example_2_3_data.plot_over_line(gb, pts, "p", tol)

    arc_length = y_range - b_box[0][1]
    np.savetxt(folder_export + "plot_over_line.txt", (arc_length, values))

    # compute the flow rate
    diam, flow_rate = example_2_3_data.compute_flow_rate_vem(gb, tol)
    np.savetxt(folder_export + "flow_rate.txt", (diam, flow_rate))

    # compute the number of cells
    num_cells = gb.num_cells(lambda g: g.dim == 2)
    with open(folder_export + "cells.txt", "w") as f:
        f.write(str(num_cells))


# ------------------------------------------------------------------------------#

num_simu = 20
for i in np.arange(num_simu):
    main(i + 1, if_export=True)

# ------------------------------------------------------------------------------#
