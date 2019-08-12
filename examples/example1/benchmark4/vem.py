import numpy as np
import scipy.sparse as sps
import porepy as pp

# ------------------------------------------------------------------------------#


def add_data(gb, domain):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(["param"])
    tol = 1e-3
    a = 1e-2

    for g, d in gb:
        # Permeability
        if g.dim == 2:
            perm = pp.SecondOrderTensor(1e-14 * np.ones(g.num_cells))
        else:
            perm = pp.SecondOrderTensor(1e-8 * np.ones(g.num_cells))

        # Source term

        # Assign apertures
        a_dim = np.power(a, gb.dim_max() - g.dim)
        aperture = np.ones(g.num_cells) * a_dim

        specified_parameters = {"aperture": aperture, "second_order_tensor": perm}

        # Boundaries
        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < domain["xmin"] + tol
            right = bound_face_centers[0, :] > domain["xmax"] - tol

            labels = np.array(["neu"] * bound_faces.size)
            labels[np.logical_or(left, right)] = "dir"

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[left]] = 1013250

            bound = pp.BoundaryCondition(g, bound_faces, labels)
            specified_parameters.update({"bc": bound, "bc_values": bc_val})
        else:
            bound = pp.BoundaryCondition(g, np.empty(0), np.empty(0))
            specified_parameters.update({"bc": bound})
        pp.initialize_default_data(g, d, "flow", specified_parameters)

    # Assign coupling permeability
    gb.add_edge_props("kn")
    for e, d in gb.edges():
        gn = gb.nodes_of_edge(e)
        mg = d["mortar_grid"]
        kn = 2 * 1e-8 * np.ones(gn[0].num_cells) / a
        d[pp.PARAMETERS] = pp.Parameters(mg, ["flow"], [{"normal_diffusivity": kn}])
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}


# ------------------------------------------------------------------------------#


mesh_kwargs = {"mesh_size_frac": 10, "mesh_size_bound": 40, "mesh_size_min": 1e-1}

domain = {"xmin": 0, "xmax": 700, "ymin": 0, "ymax": 600}
gb = pp.importer.dfm_2d_from_csv("network.csv", mesh_kwargs, domain)
gb.compute_geometry()
pp.coarsening.coarsen(gb, "by_volume")
gb.assign_node_ordering()

# Assign parameters
add_data(gb, domain)

# Choose and define the solvers and coupler
solver_flow = pp.DualVEMMixedDim("flow")
A_flow, b_flow = solver_flow.matrix_rhs(gb)

solver_source = pp.DualSourceMixedDim("flow")
A_source, b_source = solver_source.matrix_rhs(gb)

up = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)
solver_flow.split(gb, "up", up)

gb.add_node_props(["darcy_flux", "pressure", "P0u"])
solver_flow.extract_u(gb, "up", "darcy_flux")
solver_flow.extract_p(gb, "up", "pressure")
solver_flow.project_u(gb, "darcy_flux", "P0u")

save = pp.Exporter(gb, "vem", folder="vem")
save.write_vtk(["pressure", "P0u"])
