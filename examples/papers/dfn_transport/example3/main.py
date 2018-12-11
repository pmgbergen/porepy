import numpy as np
import porepy as pp

import examples.papers.dfn_transport.discretization as discr
from examples.papers.dfn_transport.flux_trace import jump_flux

def bc_flag(g, domain, out_flow_start, out_flow_end, in_flow_start, in_flow_end, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    # detect all the points aligned with the segment
    dist, _ = pp.cg.dist_points_segments(b_face_centers, out_flow_start, out_flow_end)
    dist = dist.flatten()
    out_flow = np.logical_and(dist < tol, dist >=-tol)

    # detect all the points aligned with the segment
    dist, _ = pp.cg.dist_points_segments(b_face_centers, in_flow_start, in_flow_end)
    dist = dist.flatten()
    in_flow = np.logical_and(dist < tol, dist >=-tol)

    return in_flow, out_flow

def bc_same(g, domain, tol):

    # define outflow type boundary conditions
    out_flow_start = np.array([-319.289, 212.271, 400])
    out_flow_end = np.array([-300.035, 317.811, 128.887])

    # define inflow type boundary conditions
    in_flow_start = np.array([-84.3598, 1500, -6.65313])
    in_flow_end = np.array([-84.3598, 1500, 400])

    return bc_flag(g, domain, out_flow_start, out_flow_end, in_flow_start, in_flow_end, tol)

def bc_different(g, domain, tol):

    # define outflow type boundary conditions
    out_flow_start = np.array([134.428, 100, 18.9949])
    out_flow_end = np.array([134.429, 100, 400])

    # define inflow type boundary conditions
    in_flow_start = np.array([-84.3598, 1500, -6.65313])
    in_flow_end = np.array([-84.3598, 1500, 400])

    return bc_flag(g, domain, out_flow_start, out_flow_end, in_flow_start, in_flow_end, tol)

def main():

    input_folder = "../geometries/"
    file_name = input_folder + "example3.fab"
    file_inters = None #input_folder + "example3.dat"

    mesh_size = 1e2 #np.power(2., -4)
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}
    tol = 1e-2

    bc_types = {"same": bc_same, "different": bc_different}

    for folder, bc_type in bc_types.items():

        gb = pp.importer.dfn_3d_from_fab(file_name, file_inters, tol=tol, **mesh_kwargs)

        gb.remove_nodes(lambda g: g.dim == 0)
        gb.compute_geometry()
        gb.assign_node_ordering()

        # recombute the domanin
        domain = gb.bounding_box(as_dict=True)

        param = {"domain": domain, "tol": tol, "k": 1,
                 "diff": 1e-4, "time_step": 1, "n_steps": 100,
                 "folder": folder}

        # the flow problem
        model_flow = discr.flow(gb, param, bc_type)

        #jump_flux(gb, param["mortar_flux"])

        # the advection-diffusion problem
        discr.advdiff(gb, param, model_flow, bc_type)


if __name__ == "__main__":
    main()
