import numpy as np
import scipy.sparse as sps

from porepy.viz.exporter import Exporter

from porepy.numerics.fv import tpfa, fvutils, source

import example_2_2_create_grid
import example_2_2_data

#------------------------------------------------------------------------------#


def main(id_problem, tol=1e-5, N_pts=1000, if_export=False):

    folder_export = 'example_2_2_tpfa/' + str(id_problem) + "/"
    file_export = 'tpfa'

    gb = example_2_2_create_grid.create(id_problem, tol=tol)

    # Assign parameters
    example_2_2_data.add_data(gb, tol)

    # Choose and define the solvers and coupler
    solver_flux = tpfa.TpfaDFN(gb.dim_max(), 'flow')
    A_flux, b_flux = solver_flux.matrix_rhs(gb)
    solver_source = source.IntegralDFN(gb.dim_max(), 'flow')
    A_source, b_source = solver_source.matrix_rhs(gb)

    p = sps.linalg.spsolve(A_flux + A_source, b_flux + b_source)
    solver_flux.split(gb, 'pressure', p)

    if if_export:
        save = Exporter(gb, file_export, folder_export)
        save.write_vtk(['pressure'])

    b_box = gb.bounding_box()
    y_range = np.linspace(b_box[0][1] + tol, b_box[1][1] - tol, N_pts)
    pts = np.stack((1.5 * np.ones(N_pts), y_range, 0.5 * np.ones(N_pts)))
    values = example_2_2_data.plot_over_line(gb, pts, 'pressure', tol)

    arc_length = y_range - b_box[0][1]
    np.savetxt(folder_export + "plot_over_line.txt", (arc_length, values))

    # compute the flow rate
    fvutils.compute_discharges(gb, 'flow')
    diam, flow_rate = example_2_2_data.compute_flow_rate(gb, tol)
    np.savetxt(folder_export + "flow_rate.txt", (diam, flow_rate))

#------------------------------------------------------------------------------#


num_simu = 44
for i in np.arange(num_simu):
    main(i + 1, if_export=True)

#------------------------------------------------------------------------------#
