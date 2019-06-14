import numpy as np
import porepy as pp

import examples.papers.dfn_transport.discretization as compute

# from examples.papers.dfn_transport.grid_export import grid_export

# from examples.papers.dfn_transport.flux_trace import jump_flux


def bc_flag(g, domain, out_flow_start, out_flow_end, in_flow_start, in_flow_end, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    # detect all the points aligned with the segment
    dist, _ = pp.distances.points_segments(b_face_centers, out_flow_start, out_flow_end)
    dist = dist.flatten()
    out_flow = np.logical_and(dist < tol, dist >= -tol)

    # detect all the points aligned with the segment
    dist, _ = pp.distances.points_segments(b_face_centers, in_flow_start, in_flow_end)

    dist = dist.flatten()
    in_flow = np.logical_and(dist < tol, dist >= -tol)

    return in_flow, out_flow


def bc_same(g, domain, tol):

    # define outflow type boundary conditions
    out_flow_start = np.array([-319.289, 212.271, 400])
    out_flow_end = np.array([-300.035, 317.811, 128.887])

    # define inflow type boundary conditions
    in_flow_start = np.array([-84.3598, 1500, -6.65313])
    in_flow_end = np.array([-84.3598, 1500, 400])

    return bc_flag(
        g, domain, out_flow_start, out_flow_end, in_flow_start, in_flow_end, tol
    )


def bc_different(g, domain, tol):

    # define outflow type boundary conditions
    out_flow_start = np.array([134.428, 100, 18.9949])
    out_flow_end = np.array([134.429, 100, 400])

    # define inflow type boundary conditions
    in_flow_start = np.array([-84.3598, 1500, -6.65313])
    in_flow_end = np.array([-84.3598, 1500, 400])

    return bc_flag(
        g, domain, out_flow_start, out_flow_end, in_flow_start, in_flow_end, tol
    )


def main():

    input_folder = "../geometries/"
    file_name = input_folder + "example3_connected.fab"

    # define the discretizations for the Darcy part
    discretizations = compute.get_discr()

    # geometric tolerance
    tol = 1e-3

    # initial condition and type of fluid/rock
    theta = 80 * pp.CELSIUS

    # reaction coefficient \gamma * (T - T_rock)
    gamma = 2.44e-9 * 0.125
    theta_rock = theta

    # boundary conditions
    bc_flow = 2500 * pp.METER / 5
    bc_trans = 30 * pp.CELSIUS

    end_time = 3.154e7
    n_steps = 200
    time_step = end_time / n_steps

    bc_types = {"same": bc_same, "different": bc_different}
    for bc_type_key, bc_type in bc_types.items():

        for discr_key, discr in discretizations.items():

            if discr_key == "MVEM":
                mesh_size = 0.17 * 1e2
            else:
                mesh_size = 1e2  # np.power(2., -4)

            mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}

            folder = "solution_" + discr_key + "_" + bc_type_key

            network = pp.fracture_importer.network_3d_from_fab(file_name, tol=tol)
            gb = network.mesh(mesh_kwargs, dfn=True)

            gb.remove_nodes(lambda g: g.dim == 0)
            gb.compute_geometry()
            gb.assign_node_ordering()

            if discr_key == "MVEM":
                pp.coarsening.coarsen(gb, "by_volume")

            domain = gb.bounding_box(as_dict=True)

            param = {
                "domain": domain,
                "tol": tol,
                "k": 1.84e-6,
                "bc_flow": bc_flow,
                "diff": 0.35e-9,
                "mass_weight": 1.95e-3,
                "src": gamma * theta_rock,
                "reaction": gamma,
                "flux_weight": 1.0,
                "bc_trans": bc_trans,
                "init_trans": theta,
                "time_step": time_step,
                "n_steps": n_steps,
                "folder": folder,
            }

            # the flow problem
            compute.flow(gb, discr, param, bc_type)

            # if discr_key == "Tpfa":
            #    grid_export(gb, None, folder + "/grid/")

            # jump_flux(gb, param["mortar_flux"])

            # the advection-diffusion problem
            compute.advdiff(gb, discr, param, bc_type)


if __name__ == "__main__":
    main()
