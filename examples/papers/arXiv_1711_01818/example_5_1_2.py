import numpy as np
import scipy.sparse as sps

from porepy.viz import exporter
from porepy.fracs import importer

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.grids import coarsening as co

from porepy.numerics.vem import dual

from porepy.utils import comp_geom as cg
from porepy.utils import sort_points

# ------------------------------------------------------------------------------#


def add_data(gb, domain):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(["param", "is_tangent"])
    tol = 1e-3
    a = 1e-2

    for g, d in gb:
        param = Parameters(g)

        # Permeability
        d["is_tangential"] = True
        if g.dim == 2:
            kxx = 1e-14 * np.ones(g.num_cells)
        else:
            kxx = 1e-8 * np.ones(g.num_cells)

        perm = tensor.SecondOrderTensor(g.dim, kxx)
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

            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", BoundaryCondition(g, np.empty(0), np.empty(0)))

        d["param"] = param

    # Assign coupling permeability
    gb.add_edge_prop("kn")
    for e, d in gb.edges_props():
        gn = gb.sorted_nodes_of_edge(e)
        aperture = np.power(a, gb.dim_max() - gn[0].dim)
        d["kn"] = 1e-10 * np.ones(gn[0].num_cells) / aperture


# ------------------------------------------------------------------------------#


def plot_over_line(gb, pts, name, tol):

    values = np.zeros(pts.shape[1])
    is_found = np.zeros(pts.shape[1], dtype=np.bool)

    for g, d in gb:
        if g.dim < gb.dim_max():
            continue

        if not cg.is_planar(np.hstack((g.nodes, pts)), tol=1e-4):
            continue

        faces_cells, _, _ = sps.find(g.cell_faces)
        nodes_faces, _, _ = sps.find(g.face_nodes)

        normal = cg.compute_normal(g.nodes)
        for c in np.arange(g.num_cells):
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
            pts_id_c = np.array(
                [
                    nodes_faces[g.face_nodes.indptr[f] : g.face_nodes.indptr[f + 1]]
                    for f in faces_cells[loc]
                ]
            ).T
            pts_id_c = sort_points.sort_point_pairs(pts_id_c)[0, :]
            pts_c = g.nodes[:, pts_id_c]

            mask = np.where(np.logical_not(is_found))[0]
            if mask.size == 0:
                break
            check = np.zeros(mask.size, dtype=np.bool)
            last = False
            for i, pt in enumerate(pts[:, mask].T):
                check[i] = cg.is_point_in_cell(pts_c, pt)
                if last and not check[i]:
                    break
            is_found[mask] = check
            values[mask[check]] = d[name][c]

    return values


##------------------------------------------------------------------------------#


tol = 1e-4

mesh_kwargs = {"mesh_size_frac": 500, "mesh_size_min": 20}
domain = {"xmin": 0, "xmax": 700, "ymin": 0, "ymax": 600}
gb = importer.from_csv("network.csv", mesh_kwargs, domain)
gb.compute_geometry()
co.coarsen(gb, "by_volume")
gb.assign_node_ordering()

# Assign parameters
add_data(gb, domain)

# Choose and define the solvers and coupler
solver = dual.DualVEMMixDim("flow")
A, b = solver.matrix_rhs(gb)

up = sps.linalg.spsolve(A, b)
solver.split(gb, "up", up)

gb.add_node_props(["discharge", "pressure", "P0u"])
solver.extract_u(gb, "up", "discharge")
solver.extract_p(gb, "up", "pressure")
solver.project_u(gb, "discharge", "P0u")

exporter.export_vtk(gb, "vem", ["pressure", "P0u"], folder="example_5_1_2")

# This part is very slow and not optimized, it's just to obtain the plots once.
b_box = gb.bounding_box()
N_pts = 1000
y_range = np.linspace(b_box[0][1] + tol, b_box[1][1] - tol, N_pts)
pts = np.stack((625 * np.ones(N_pts), y_range, np.zeros(N_pts)))
values = plot_over_line(gb, pts, "pressure", tol)

arc_length = y_range - b_box[0][1]
np.savetxt("example_5_1_2/vem_x_625.csv", (arc_length, values))

x_range = np.linspace(b_box[0][0] + tol, b_box[1][0] - tol, N_pts)
pts = np.stack((x_range, 500 * np.ones(N_pts), np.zeros(N_pts)))
values = plot_over_line(gb, pts, "pressure", tol)

arc_length = x_range - b_box[0][0]
np.savetxt("example_5_1_2/vem_y_500.csv", (arc_length, values))


print("diam", gb.diameter(lambda g: g.dim == gb.dim_max()))
print("num_cells 2d", gb.num_cells(lambda g: g.dim == 2))
print("num_cells 1d", gb.num_cells(lambda g: g.dim == 1))
