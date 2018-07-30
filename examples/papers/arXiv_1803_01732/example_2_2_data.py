import numpy as np
import scipy.sparse as sps

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.utils import comp_geom as cg
from porepy.utils import sort_points

# ------------------------------------------------------------------------------#


def add_data(gb, tol):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(["param", "Aavatsmark_transmissibilities"])

    for g, d in gb:
        param = Parameters(g)
        d["Aavatsmark_transmissibilities"] = True

        if g.dim == 2:

            # Permeability
            kxx = np.ones(g.num_cells)
            param.set_tensor("flow", tensor.SecondOrderTensor(g.dim, kxx))

            # Source term
            param.set_source("flow", np.zeros(g.num_cells))

            # Boundaries
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = g.face_centers[:, bound_faces]

                bottom = bound_face_centers[1, :] < tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[bottom] = "dir"

                bc_val = np.zeros(g.num_faces)
                mask = bound_face_centers[0, :] < tol
                bc_val[bound_faces[np.logical_and(bottom, mask)]] = 1

                param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
                param.set_bc_val("flow", bc_val)
            else:
                param.set_bc("flow", BoundaryCondition(g, np.empty(0), np.empty(0)))

        d["param"] = param

    # Assign coupling discharge
    gb.add_edge_props(["param", "Aavatsmark_transmissibilities"])
    for e, d in gb.edges_props():
        g_h = gb.sorted_nodes_of_edge(e)[1]
        d["param"] = Parameters(g_h)
        d["Aavatsmark_transmissibilities"] = True


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


# ------------------------------------------------------------------------------#


def compute_flow_rate(gb, tol):

    total_flow_rate = 0
    for g, d in gb:
        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]
            mask = np.logical_and(
                bound_face_centers[1, :] < tol, bound_face_centers[0, :] > 1.5 - tol
            )
            flow_rate = d["param"].get_discharge()[bound_faces[mask]]
            total_flow_rate += np.sum(flow_rate)

    diam = gb.diameter(lambda g: g.dim == gb.dim_max())
    return diam, total_flow_rate


# ------------------------------------------------------------------------------#


def compute_flow_rate_vem(gb, tol):

    total_flow_rate = 0
    for g, d in gb:
        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]
            mask = np.logical_and(
                bound_face_centers[1, :] < tol, bound_face_centers[0, :] > 1.5 - tol
            )
            flow_rate = d["discharge"][bound_faces[mask]]
            total_flow_rate += np.sum(flow_rate)

    diam = gb.diameter(lambda g: g.dim == gb.dim_max())
    return diam, total_flow_rate


# ------------------------------------------------------------------------------#
