import numpy as np
import scipy.sparse as sps

from porepy.viz.exporter import Exporter


from porepy.numerics.vem import vem_dual, vem_source

import example_2_2_create_grid
import example_2_2_data

# ------------------------------------------------------------------------------#


def main(id_problem, is_coarse=False, tol=1e-5, N_pts=1000, if_export=False):

    folder_export = "example_2_2_vem_coarse/" + str(id_problem) + "/"
    file_export = "vem"

    gb = example_2_2_create_grid.create(id_problem, is_coarse=is_coarse, tol=tol)

    # Assign parameters
    example_2_2_data.add_data(gb, tol)

    # Choose and define the solvers and coupler
    solver_flow = vem_dual.DualVEMDFN(gb.dim_max(), "flow")
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = vem_source.DualSourceDFN(gb.dim_max(), "flow")
    A_source, b_source = solver_source.matrix_rhs(gb)

    up = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)
    solver_flow.split(gb, "up", up)

    gb.add_node_props(["discharge", "pressure", "P0u"])
    solver_flow.extract_u(gb, "up", "discharge")
    solver_flow.extract_p(gb, "up", "pressure")
    solver_flow.project_u(gb, "discharge", "P0u")

    if if_export:
        save = Exporter(gb, file_export, folder_export)
        save.write_vtk(["pressure", "P0u"])

    b_box = gb.bounding_box()
    y_range = np.linspace(b_box[0][1] + tol, b_box[1][1] - tol, N_pts)
    pts = np.stack((1.5 * np.ones(N_pts), y_range, 0.5 * np.ones(N_pts)))
    values = example_2_2_data.plot_over_line(gb, pts, "pressure", tol)

    arc_length = y_range - b_box[0][1]
    np.savetxt(folder_export + "plot_over_line.txt", (arc_length, values))

    # compute the flow rate
    diam, flow_rate = example_2_2_data.compute_flow_rate_vem(gb, tol)
    np.savetxt(folder_export + "flow_rate.txt", (diam, flow_rate))


# ------------------------------------------------------------------------------#


num_simu = 44
is_coarse = True
for i in np.arange(num_simu):
    main(i + 1, is_coarse, if_export=True)

# ------------------------------------------------------------------------------#
