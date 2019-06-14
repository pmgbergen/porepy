import numpy as np
import porepy as pp

import examples.papers.dfn_transport.discretization as compute

# from examples.papers.dfn_transport.grid_export import grid_export

# from examples.papers.dfn_transport.flux_trace import jump_flux


def bc_flag(g, domain, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    # define inflow type boundary conditions
    out_flow_start = np.array([1.011125, 0.249154, 0.598708])
    out_flow_end = np.array([1.012528, 0.190858, 0.886822])

    # detect all the points aligned with the segment
    dist, _ = pp.distances.points_segments(b_face_centers, out_flow_start, out_flow_end)

    dist = dist.flatten()
    out_flow = np.logical_and(dist < tol, dist >= -tol)

    # define outflow type boundary conditions
    in_flow_start = np.array([0.206507, 0.896131, 0.183632])
    in_flow_end = np.array([0.181980, 0.813947, 0.478618])

    # detect all the points aligned with the segment
    dist, _ = pp.distances.points_segments(b_face_centers, in_flow_start, in_flow_end)
    dist = dist.flatten()
    in_flow = np.logical_and(dist < tol, dist >= -tol)

    return in_flow, out_flow


def main():

    input_folder = "../geometries/"
    file_name = input_folder + "example2.fab"

    # define the discretizations for the Darcy part
    discretizations = compute.get_discr()

    # geometric tolerance
    tol = 1e-5

    # define the mesh sizes
    mesh_sizes = {
        "3k": 0.9 * np.power(2.0, -4),  # for 3k triangles
        "40k": 0.49 * 0.875 * np.power(2.0, -5),  # for 40k triangles
    }

    for mesh_size_key in mesh_sizes.keys():

        for discr_key, discr in discretizations.items():

            if discr_key == "MVEM":
                if mesh_size_key == "3k":
                    mesh_size = 0.9 * np.power(2.0, -4) * 0.675
                elif mesh_size_key == "40k":
                    mesh_size = 0.49 * 0.875 * np.power(2.0, -5) * 0.7
            else:
                mesh_size = mesh_sizes[mesh_size_key]

            mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}

            folder = "solution_" + discr_key + "_" + mesh_size_key

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
                "k": 1,
                "diff": 1e-4,
                "time_step": 0.05,
                "n_steps": 500,
                "folder": folder,
            }

            # the flow problem
            compute.flow(gb, discr, param, bc_flag)

            # if discr_key == "Tpfa":
            #    grid_export(gb, None, "grid_" + mesh_size_key + "/")

            # jump_flux(gb, param["mortar_flux"])

            # the advection-diffusion problem
            compute.advdiff(gb, discr, param, bc_flag)


if __name__ == "__main__":
    main()
