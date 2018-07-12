import numpy as np
import scipy.sparse as sps

from porepy.viz.exporter import Exporter

from porepy.numerics.fv import tpfa, fvutils, source

import example_2_1_create_grid
import example_2_1_data

# ------------------------------------------------------------------------------#


def main(id_problem, tol=1e-5, N_pts=1000, if_export=False):

    mesh_size = 0.15
    folder_export = "example_2_1_tpfa_" + str(mesh_size) + "/" + str(id_problem) + "/"
    file_export = "tpfa"

    gb = example_2_1_create_grid.create(id_problem, mesh_size=mesh_size, tol=tol)

    # Assign parameters
    example_2_1_data.add_data(gb, tol)

    # Choose and define the solvers and coupler
    solver_flux = tpfa.TpfaDFN(gb.dim_max(), "flow")
    A_flux, b_flux = solver_flux.matrix_rhs(gb)
    solver_source = source.IntegralDFN(gb.dim_max(), "flow")
    A_source, b_source = solver_source.matrix_rhs(gb)

    p = sps.linalg.spsolve(A_flux + A_source, b_flux + b_source)
    solver_flux.split(gb, "pressure", p)

    if if_export:
        save = Exporter(gb, file_export, folder_export)
        save.write_vtk(["pressure"])

    b_box = gb.bounding_box()
    z_range = np.linspace(b_box[0][2], b_box[1][2], N_pts)
    pts = np.stack((0.5 * np.ones(N_pts), 0.5 * np.ones(N_pts), z_range))
    values = example_2_1_data.plot_over_line(gb, pts, "pressure", tol)

    arc_length = z_range - b_box[0][2]
    np.savetxt(folder_export + "plot_over_line.txt", (arc_length, values))

    # compute the flow rate
    fvutils.compute_discharges(gb, "flow")
    diam, flow_rate = example_2_1_data.compute_flow_rate(gb, tol)
    np.savetxt(folder_export + "flow_rate.txt", (diam, flow_rate))


# ------------------------------------------------------------------------------#


num_simu = 21
for i in np.arange(num_simu):
    main(i + 1, if_export=True)

# ------------------------------------------------------------------------------#
