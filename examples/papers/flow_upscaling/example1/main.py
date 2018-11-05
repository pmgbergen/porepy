import numpy as np
import porepy as pp

import data as problem_data
import examples.papers.flow_upscaling.solvers as solvers
from examples.papers.flow_upscaling.import_grid import grid

def main(file_geo, folder, data, mesh_args, tol):

    # import the grid and the domain
    gb, data["domain"] = grid(file_geo, mesh_args, tol)

    # define the data
    problem_data.add_data(gb, data)

    # solve the pressure problem
    solvers.pressure(gb, folder)

    # solve the transport problem
    advective = solvers.transport(gb, data, folder,
                                  problem_data.AdvectiveDataAssigner)
    np.savetxt(folder+"/outflow.csv", advective._solver.outflow)

if __name__ == "__main__":

    if False:
        file_geo = "../networks/Algeroyna_2.csv"
        t_max = 1e5 * pp.SECOND
        mesh_args = {'mesh_size_frac': 0.0390625}
        tol = {"geo": 1e-3, "snap": 0.5*1e-3}

    elif False:
        file_geo = "../networks/Algeroyna_3.csv"
        t_max = 1e7 * pp.SECOND
        mesh_args = {'mesh_size_frac': 0.3125}
        tol = {"geo": 1.25*1e-2, "snap": 0.6125*1e-2}

    elif False:
        file_geo = "../networks/Algeroyna_1to1_Points.csv"
        t_max = 1e7 * pp.SECOND
        mesh_args = {'mesh_size_frac': 0.0390625}
        tol = {"geo": 1e-3, "snap": 0.5*1e-3}

    elif False:
        file_geo = "../networks/Algeroyna_1to1000_Points.csv"
        t_max = 1e7 * pp.SECOND
        mesh_args = {'mesh_size_frac': 10}
        tol = {"geo": 1e-3, "snap": 0.5*1e-3}

    elif True:
        file_geo = "../networks/Algeroyna_1to10000_Points.csv"
        t_max = 1e7 * pp.SECOND
        mesh_args = {'mesh_size_frac': 100}
        tol = {"geo": 1e-3, "snap": 0.5*1e-3}

    elif False:
        file_geo = "../networks/Vikso_1to10_Points.csv"
        t_max = 1e5 * pp.SECOND #
        mesh_args = {'mesh_size_frac': 0.2}
        tol = {"geo": 1e-3, "snap": 0.5*1e-3}

    elif False: # problems
        file_geo = "../networks/Vikso_1to100_Points.csv"
        t_max = 1e5 * pp.SECOND #
        mesh_args = {'mesh_size_frac': 1}
        tol = {"geo": 1e-3, "snap": 0.125*1e-4}

    elif False:
        file_geo = "../networks/Vikso_1to1000_Points.csv"
        t_max = 1e5 * pp.SECOND #
        mesh_args = {'mesh_size_frac': 10}
        tol = {"geo": 0.5*1e-2, "snap": 0.0625*1e-2}

    folder = "solution"

    theta_ref = 80. * pp.CELSIUS
    fluid = pp.Water(theta_ref)
    rock = pp.Granite(theta_ref)

    aperture = 2*pp.MILLIMETER

    # select the permeability depending on the selected test case
    data = {
        "aperture": aperture,
        "kf": aperture**2 / 12 / fluid.dynamic_viscosity(),
        "km": 1e-14 / fluid.dynamic_viscosity(),
        "porosity_f": 0.85,
        "dt": t_max/100,
        "t_max": t_max,
        "fluid": fluid,
        "rock": rock,
        "tol": tol["geo"]
    }

    main(file_geo, folder, data, mesh_args, tol)
