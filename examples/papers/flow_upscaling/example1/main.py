import numpy as np
import porepy as pp

import data as problem_data
import examples.papers.flow_upscaling.solvers as solvers
import examples.papers.flow_upscaling.import_grid as grid
from examples.papers.flow_upscaling.upscaling import upscaling

def main(file_geo, folder, data):

    mesh_args = {'mesh_size_frac': 10}
    tol = {"geo": 1e-4, "snap": 1e-3}
    data["tol"] = tol["geo"]

    gb, data["domain"] = grid.from_file(file_geo, mesh_args, tol)

    problem_data.add_data(gb, data)
    solvers.solve_rt0(gb, folder)

    advective = solvers.transport(gb, data, folder,
                                  problem_data.AdvectiveDataAssigner)
    np.savetxt(folder+"/outflow.csv", advective._solver.outflow)

if __name__ == "__main__":

    file_geo = "Algeroyna.csv"
    aperture = pp.MILLIMETER
    data = {"aperture": aperture, "kf": aperture**2 / 12}
    folder = "upscaling"
    upscaling(file_geo, data, folder, dfn=True, shrink=0.10)

    sa

    file_geo = "Algeroyna.csv"
    folder = "solution"

    theta_ref = 30. * pp.CELSIUS
    fluid = pp.Water(theta_ref)
    rock = pp.Granite(theta_ref)

    aperture = 2 * pp.MILLIMETER

    # select the permeability depending on the selected test case
    data = {
        "aperture": aperture,
        "kf": aperture**2 / 12,
        "km": 1e7 * rock.PERMEABILITY,
        "porosity_f": 0.85,
        "dt": 1e6 * pp.SECOND,
        "t_max": 1e7 * pp.SECOND,
        "fluid": fluid,
        "rock": rock
    }

    main(file_geo, folder, data)
