import numpy as np
import porepy as pp

import examples.papers.flow_upscaling.solvers as solvers
from examples.papers.flow_upscaling.import_grid import grid

def main(file_geo, param, mesh_args, tol):

    # import the grid and the domain
    gb, param["domain"] = grid(file_geo, mesh_args, tol)

    # solve the pressure problem
    param["keys_flow"] = solvers.flow(gb, param)

    solvers.advdiff(gb, param)

if __name__ == "__main__":

    theta_ref = 80. * pp.CELSIUS
    fluid = pp.Water(theta_ref)
    rock = pp.Granite(theta_ref)

    aperture = 2*pp.MILLIMETER
    n_steps = 1 #200

    mu = fluid.dynamic_viscosity()

    folder_fractures = "../fracture_networks/"

    if True:
        file_geo = folder_fractures + "Algeroyna_1to1.csv"
        mesh_args = {'mesh_size_frac': 0.0390625/16}
        tol = {"geo": 2.5*1e-3, "snap": 0.75*2.5*1e-3}
        folder = "algeroyna_1to1"

        delta_x = 0.13
        bc_flow = pp.BAR
        km = 1e-14

        t_max = delta_x * delta_x * mu / (bc_flow * km)

    elif False: # quite slow
        file_geo = folder_fractures + "Algeroyna_1to10.csv"
        mesh_args = {'mesh_size_frac': 0.0390625}
        tol = {"geo": 1e-8, "snap": 1e-3}
        folder = "algeroyna_1to10"

        delta_x = 2.8
        bc_flow = pp.BAR
        km = 1e-14

        t_max = delta_x * delta_x * mu / (bc_flow * km)

    elif False:
        file_geo = folder_fractures + "Algeroyna_1to100.csv"
        mesh_args = {'mesh_size_frac': 10+0.3125}
        tol = {"geo": 1e-2, "snap": 1e-2}
        folder = "algeroyna_1to100"

        delta_x = 15.5
        bc_flow = pp.BAR
        km = 1e-14

        t_max = delta_x * delta_x * mu / (bc_flow * km)

    elif False:
        file_geo = folder_fractures + "Algeroyna_1to1000.csv"
        mesh_args = {'mesh_size_frac': 10, "mesh_size_min": 0.01}
        tol = {"geo": 1e-2, "snap": 1e-2}
        folder = "algeroyna_1to1000"

        delta_x = 245
        bc_flow = pp.BAR
        km = 1e-14

        t_max = delta_x * delta_x * mu / (bc_flow * km)

    elif False:
        file_geo = folder_fractures + "Algeroyna_1to10000.csv"
        mesh_args = {'mesh_size_frac': 100}
        tol = {"geo": 1e-3, "snap": 1e-3}
        folder = "algeroyna_1to10000"

        delta_x = 3160
        bc_flow = pp.BAR
        km = 1e-14

        t_max = delta_x * delta_x * mu / (bc_flow * km)

    elif False:
        file_geo = folder_fractures + "Vikso_1to10.csv"
        mesh_args = {'mesh_size_frac': 0.2}
        tol = {"geo": 1e-3, "snap": 0.75*1e-3}
        folder = "vikso_1to10"

        delta_x = 4
        bc_flow = pp.BAR
        km = 1e-14

        t_max = delta_x * delta_x * mu / (bc_flow * km)

    elif False:
        file_geo = folder_fractures + "Vikso_1to100.csv"
        mesh_args = {'mesh_size_frac': 1}
        tol = {"geo": 1e-3, "snap": 0.75*1e-3}
        folder = "vikso_1to100"

        delta_x = 30
        bc_flow = pp.BAR
        km = 1e-14

        t_max = delta_x * delta_x * mu / (bc_flow * km)

    elif False:
        file_geo = folder_fractures + "/Vikso_1to1000.csv"
        mesh_args = {'mesh_size_frac': 10}
        tol = {"geo": 1e-2, "snap": 1e-2}
        folder = "vikso_1to1000"

        delta_x = 295
        bc_flow = pp.BAR
        km = 1e-14

        t_max = delta_x * delta_x * mu / (bc_flow * km)

    # select the permeability depending on the selected test case
    param = {
        "aperture": aperture,
        "kf": aperture**2 / 12,
        "km": km,
        "porosity_f": 0.85,
        "deltaT": t_max/n_steps,
        "n_steps": n_steps,
        "fluid": fluid,
        "rock": rock,
        "tol": tol["geo"],
        "folder": folder,
        "bc_flow": bc_flow,
        "initial_advdiff": theta_ref,
        "bc_advdiff": 20 * pp.CELSIUS
    }

    main(file_geo, param, mesh_args, tol)
