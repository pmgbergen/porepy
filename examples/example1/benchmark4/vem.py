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
        param = pp.Parameters(g)

        # Permeability
        if g.dim == 2:
            perm = pp.SecondOrderTensor(g.dim, 1e-14 * np.ones(g.num_cells))
        else:
            perm = pp.SecondOrderTensor(g.dim, 1e-8 * np.ones(g.num_cells))
        param.set_tensor("flow", perm)

        # Source term
        param.set_source("flow", np.zeros(g.num_cells))

        # Assign apertures
        aperture = np.power(a, gb.dim_max() - g.dim)
        param.set_aperture(np.ones(g.num_cells) * aperture)

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

            param.set_bc("flow", pp.BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", pp.BoundaryCondition(g, np.empty(0), np.empty(0)))

        d["param"] = param

    # Assign coupling permeability
    gb.add_edge_props("kn")
    for e, d in gb.edges():
        gn = gb.nodes_of_edge(e)
        d["kn"] = 1e-8 * np.ones(gn[0].num_cells)


# ------------------------------------------------------------------------------#


mesh_kwargs = {}
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

gb.add_node_props(["discharge", "pressure", "P0u"])
solver_flow.extract_u(gb, "up", "discharge")
solver_flow.extract_p(gb, "up", "pressure")
solver_flow.project_u(gb, "discharge", "P0u")

save = pp.Exporter(gb, "vem", folder="vem")
save.write_vtk(["pressure", "P0u"])
