import numpy as np

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

# ------------------------------------------------------------------------------#


def add_data(gb, domain, direction, tol):
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
            param.set_tensor("flow", tensor.SecondOrder(3, kxx))

            # Source term
            param.set_source("flow", np.zeros(g.num_cells))

            # Boundaries
            bound_faces = g.get_boundary_faces()
            if bound_faces.size != 0:
                bound_face_centers = g.face_centers[:, bound_faces]

                if direction == "left_right":
                    left = bound_face_centers[0, :] < domain["xmin"] + tol
                    right = bound_face_centers[0, :] > domain["xmax"] - tol
                    bc_dir = np.logical_or(left, right)
                    bc_one = right
                elif direction == "bottom_top":
                    bottom = bound_face_centers[2, :] < domain["zmin"] + tol
                    top = bound_face_centers[2, :] > domain["zmax"] - tol
                    bc_dir = np.logical_or(top, bottom)
                    bc_one = top
                elif direction == "back_front":
                    back = bound_face_centers[1, :] < domain["ymin"] + tol
                    front = bound_face_centers[1, :] > domain["ymax"] - tol
                    bc_dir = np.logical_or(back, front)
                    bc_one = front

                labels = np.array(["neu"] * bound_faces.size)
                labels[bc_dir] = "dir"

                bc_val = np.zeros(g.num_faces)
                bc_val[bound_faces[bc_one]] = 1

                param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
                param.set_bc_val("flow", bc_val)
            else:
                param.set_bc("flow", BoundaryCondition(g, np.empty(0), np.empty(0)))

        d["param"] = param

    gb.add_edge_props(["param", "Aavatsmark_transmissibilities"])
    for e, d in gb.edges_props():
        g_h = gb.sorted_nodes_of_edge(e)[1]
        d["param"] = Parameters(g_h)
        d["Aavatsmark_transmissibilities"] = True


# ------------------------------------------------------------------------------#


def compute_flow_rate(gb, direction, domain, tol):

    total_flow_rate = 0
    for g, d in gb:
        bound_faces = g.get_domain_boundary_faces()

        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            if direction == "left_right":
                right = bound_face_centers[0, :] > domain["xmax"] - tol
                mask = right
            elif direction == "bottom_top":
                top = bound_face_centers[2, :] > domain["zmax"] - tol
                mask = top
            elif direction == "back_front":
                front = bound_face_centers[1, :] > domain["ymax"] - tol
                mask = front

            flow_rate = d["param"].get_discharge()[bound_faces[mask]]
            total_flow_rate += np.sum(flow_rate)

    diam = gb.diameter(lambda g: g.dim == gb.dim_max())
    return diam, total_flow_rate


# ------------------------------------------------------------------------------#


def compute_flow_rate_vem(gb, direction, domain, tol):

    total_flow_rate = 0
    for g, d in gb:
        bound_faces = g.get_domain_boundary_faces()

        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            if direction == "left_right":
                right = bound_face_centers[0, :] > domain["xmax"] - tol
                mask = right
            elif direction == "bottom_top":
                top = bound_face_centers[2, :] > domain["zmax"] - tol
                mask = top
            elif direction == "back_front":
                front = bound_face_centers[1, :] > domain["ymax"] - tol
                mask = front

            flow_rate = d["discharge"][bound_faces[mask]]
            total_flow_rate += np.sum(flow_rate)

    diam = gb.diameter(lambda g: g.dim == gb.dim_max())
    return diam, total_flow_rate


# ------------------------------------------------------------------------------#
