import numpy as np
import porepy as pp

import data as problem_data
import examples.papers.flow_upscaling.solvers as solvers

def main(file_geo, folder, data):

    mesh_args = {'mesh_size_frac': 10}
    tol = {"geo": 1e-4, "snap": 1e-3}
    data["tol"] = tol["geo"]

    gb, data["domain"] = problem_data.import_grid(file_geo, mesh_args, tol)

    problem_data.add_data(gb, data)
    solvers.solve_rt0(gb, folder)

    advective = solvers.transport(gb, data, folder,
                                  problem_data.AdvectiveDataAssigner)
    np.savetxt(folder+"/outflow.csv", advective._solver.outflow)

if __name__ == "__main__":
    file_geo = "Algeroyna_3.csv"
    folder = "solution"

    # select the permeability depending on the selected test case
    data = {
        "aperture": 5 * pp.MILLIMETER,
        "km": 1e-6,
        "kf": 1e-1,
        "porosity": 0.2,
        "porosity_f": 0.4,
        "dt": 1e6 * pp.SECOND,
        "t_max": 1e7 * pp.SECOND,
    }

    main(file_geo, folder, data)
