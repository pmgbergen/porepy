import numpy as np
import scipy.sparse as sps

from porepy.viz.exporter import Exporter

from porepy.numerics.fv import tpfa, source

import example_1_create_grid
import example_1_data


def main(id_problem, tol=1e-5, if_export=False):

    folder_export = "example_1_tpfa/"
    file_name_error = folder_export + "tpfa_error.txt"
    gb = example_1_create_grid.create(0.5 / float(id_problem), tol)

    if if_export:
        save = Exporter(gb, "tpfa", folder_export)

    example_1_data.assign_frac_id(gb)

    # Assign parameters
    example_1_data.add_data(gb, tol)

    # Choose and define the solvers and coupler
    solver_flow = tpfa.TpfaDFN(gb.dim_max(), "flow")
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = source.IntegralDFN(gb.dim_max(), "flow")
    A_source, b_source = solver_source.matrix_rhs(gb)

    p = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)
    solver_flow.split(gb, "pressure", p)

    def only_max_dim(g):
        return g.dim == gb.dim_max()

    diam = gb.diameter(only_max_dim)
    error_pressure = example_1_data.error_pressure(gb, "pressure")
    print("h=", diam, "- err(p)=", error_pressure)

    with open(file_name_error, "a") as f:
        info = (
            str(gb.num_cells(only_max_dim))
            + " "
            + str(gb.num_cells(only_max_dim))
            + " "
            + str(error_pressure)
            + "\n"
        )
        f.write(info)

    if if_export:
        save.write_vtk(["pressure", "err"])


# ------------------------------------------------------------------------------#


num_simu = 25
for i in np.arange(num_simu):
    main(i + 1)

# ------------------------------------------------------------------------------#
