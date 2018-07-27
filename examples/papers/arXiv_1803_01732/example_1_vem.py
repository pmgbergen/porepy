import numpy as np
import scipy.sparse as sps

from porepy.viz.exporter import Exporter

from porepy.grids import coarsening as co

from porepy.numerics.vem import vem_dual, vem_source

import example_1_create_grid
import example_1_data

# ------------------------------------------------------------------------------#


def main(id_problem, is_coarse=False, tol=1e-5, if_export=False):

    gb = example_1_create_grid.create(0.5 / float(id_problem), tol)

    if is_coarse:
        co.coarsen(gb, "by_tpfa")
        folder_export = "example_1_vem_coarse/"
    else:
        folder_export = "example_1_vem/"

    file_name_error = folder_export + "vem_error.txt"

    if if_export:
        save = Exporter(gb, "vem", folder_export)

    example_1_data.assign_frac_id(gb)

    # Assign parameters
    example_1_data.add_data(gb, tol)

    # Choose and define the solvers and coupler
    solver_flow = vem_dual.DualVEMDFN(gb.dim_max(), "flow")
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = vem_source.DualSourceDFN(gb.dim_max(), "flow")
    A_source, b_source = solver_source.matrix_rhs(gb)

    up = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)
    solver_flow.split(gb, "up", up)

    gb.add_node_props(["discharge", "pressure", "P0u", "err"])
    solver_flow.extract_u(gb, "up", "discharge")
    solver_flow.extract_p(gb, "up", "pressure")
    solver_flow.project_u(gb, "discharge", "P0u")

    only_max_dim = lambda g: g.dim == gb.dim_max()
    diam = gb.diameter(only_max_dim)
    error_pressure = example_1_data.error_pressure(gb, "p")
    error_discharge = example_1_data.error_discharge(gb, "P0u")
    print("h=", diam, "- err(p)=", error_pressure, "- err(P0u)=", error_discharge)
    error_pressure = example_1_data.error_pressure(gb, "pressure")

    with open(file_name_error, "a") as f:
        info = (
            str(gb.num_cells(only_max_dim))
            + " "
            + str(gb.num_cells(only_max_dim))
            + " "
            + str(error_pressure)
            + " "
            + str(error_discharge)
            + " "
            + str(gb.num_faces(only_max_dim))
            + "\n"
        )
        f.write(info)

    if if_export:
        save.write_vtk(["pressure", "err", "P0u"])


# ------------------------------------------------------------------------------#


num_simu = 25
if_export = False

for i in np.arange(num_simu):
    main(i + 1, is_coarse=True, if_export=if_export)

for i in np.arange(num_simu):
    main(i + 1, is_coarse=False, if_export=if_export)

# ------------------------------------------------------------------------------#
