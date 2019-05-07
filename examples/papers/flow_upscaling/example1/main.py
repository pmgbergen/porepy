import numpy as np
import porepy as pp

import examples.papers.flow_upscaling.solvers as solvers
from examples.papers.flow_upscaling.logger import logger
import functions as fct

def main(file_geo, param, mesh_args, tol):

    # import the network
    network = pp.fracture_importer.network_2d_from_csv(file_geo, polyline=True, tol=tol["geo"])
    #network.plot(pts_coord=True)

    # shift the network to use reasonable numbers and snap the fractures
    fct.center_network(network)
    network = network.split_intersections().snap(tol["snap"])

    # Generate a mixed-dimensional mesh
    gb = network.mesh(mesh_args, do_snap=False)

    # Do the computation in both directions: left-to-right and bottom-to-top
    flow_directions = {"left_to_right": 0, "bottom_to_top": 1}
    folder = param["folder"]

    for flow_direction_name, flow_direction in flow_directions.items():

        logger.info("Rescale the problem such that in the direction the lenght is 2, from -1 to 1")
        gb_scaled = fct.rescale_gb(gb)

        domain = gb_scaled.bounding_box(as_dict=True)
        param["domain"] = domain
        domain_length = np.array([domain["xmax"] - domain["xmin"],
                                  domain["ymax"] - domain["ymin"]])
        logger.info("done")

        logger.info("Solve the problem in the direction " + flow_direction_name)
        # solve the pressure problem
        param["flow_direction"] = flow_direction
        param["folder"] = folder + "_" + flow_direction_name
        param["bc_flow"] = param["given_flux"] * domain_length[flow_direction] / param["km"]
        model_flow = solvers.flow(gb_scaled, param)

        # compute the upscaled transmissibility
        t, out_flow = fct.transmissibility(gb_scaled, param, flow_direction)
        # save the transmissibility
        np.savetxt(param["folder"] + "/transmissibility.txt", [t])
        logger.info("done")

        # compute the pore-volume
        pv = fct.pore_volume(gb_scaled, param)

        # compute the actual ending time base on equivalent PVI
        param["time_step"] = 1e-2 / param["given_flux"] / param["n_steps"]

        solvers.advdiff(gb_scaled, param, model_flow)


if __name__ == "__main__":

    theta_ref = 80.0 * pp.CELSIUS
    fluid = pp.Water(theta_ref)
    rock = pp.Granite(theta_ref)

    aperture = 2 * pp.MILLIMETER
    n_steps = 200
    given_flux = 1e-7
    km = 1e-14

    mu = fluid.dynamic_viscosity()

    folder_fractures = "../fracture_networks/"

    id_fracs = np.arange(8)

    for id_frac in id_fracs:

        if id_frac == 0:
            h = 0.25 * 1e-2
            file_geo = folder_fractures + "Algeroyna_1to1.csv"
            tol = {"geo": 2.5 * 1e-3, "snap": 0.75 * 2.5 * 1e-3}
            folder = "algeroyna_1to1"

        elif id_frac == 1:
            h = 0.025
            file_geo = folder_fractures + "Algeroyna_1to10.csv"
            tol = {"geo": 5*1e-3, "snap": 1e-2}
            folder = "algeroyna_1to10"

        elif id_frac == 2:
            h = 0.16
            file_geo = folder_fractures + "Algeroyna_1to100.csv"
            tol = {"geo": 1e-2, "snap": 1e-2}
            folder = "algeroyna_1to100"

        elif id_frac == 3:
            h = 2.5
            file_geo = folder_fractures + "Algeroyna_1to1000.csv"
            tol = {"geo": 1e-2, "snap": 1e-2}
            folder = "algeroyna_1to1000"

        elif id_frac == 4:
            h = 15
            file_geo = folder_fractures + "Algeroyna_1to10000.csv"
            tol = {"geo": 1e-2, "snap": 5}
            folder = "algeroyna_1to10000"

        elif id_frac == 5:
            h = 0.04
            file_geo = folder_fractures + "Vikso_1to10.csv"
            tol = {"geo": 1e-3, "snap": 0.75 * 1e-3}
            folder = "vikso_1to10"

        elif id_frac == 6:
            h = 0.10
            file_geo = folder_fractures + "Vikso_1to100.csv"
            tol = {"geo": 1e-3, "snap": 5 * 1e-3}
            folder = "vikso_1to100"

        elif id_frac == 7:
            h = 1.45
            file_geo = folder_fractures + "/Vikso_1to1000.csv"
            tol = {"geo": 1e-2, "snap": 1e-2}
            folder = "vikso_1to1000"

        mesh_args = {"mesh_size_frac": h, "mesh_size_min": h / 20, "mesh_size_bound": h*2}

        # select the permeability depending on the selected test case
        param = {
            "aperture": aperture,
            "kf": aperture ** 2 / 12,
            "km": km,
            "porosity_f": 0.85,
            "n_steps": n_steps,
            "fluid": fluid,
            "given_flux": given_flux,
            "rock": rock,
            "tol": tol["geo"],
            "folder": folder,
            "initial_advdiff": theta_ref,
            "bc_advdiff": 20 * pp.CELSIUS,
        }

        main(file_geo, param, mesh_args, tol)
