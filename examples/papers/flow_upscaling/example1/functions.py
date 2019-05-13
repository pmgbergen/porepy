import numpy as np
import scipy.sparse as sps

# ------------------------------------------------------------------------------#

def center_network(network):
    # shift the network to use reasonable numbers
    shift = 0.5 * (np.amax(network.pts, axis=1) + np.amin(network.pts, axis=1))
    network.pts -= np.tile(shift, (network.pts.shape[1], 1)).T
    network.domain["xmin"] -= shift[0]
    network.domain["xmax"] -= shift[0]
    network.domain["ymin"] -= shift[1]
    network.domain["ymax"] -= shift[1]

# ------------------------------------------------------------------------------#

def rescale_gb(gb):

    domain = gb.bounding_box(as_dict=True)

    dx = domain["xmax"] - domain["xmin"]
    dy = domain["ymax"] - domain["ymin"]
    ratio = dx/dy

    dx_new = 2e2
    dy_new = dx_new / ratio

    S = np.diag([dx_new/dx, dy_new/dy, 1])

    for g, _ in gb:
        g.nodes = S.dot(g.nodes)

    gb.compute_geometry()
    return gb

# ------------------------------------------------------------------------------#


def pore_volume(gb, param):

    pv = 0
    for g, _ in gb:
        # get the porosity
        if g.dim == 2:
            phi = param["rock"].POROSITY
        else:
            phi = param["porosity_f"] * param["aperture"]
        pv += phi * np.sum(g.cell_volumes)
    return pv

# ------------------------------------------------------------------------------#

def transmissibility(gb, param, flow_direction):

    # compute the thickness of the domain
    dmin = "ymin" if flow_direction else "xmin"
    dmax = "ymax" if flow_direction else "xmax"
    delta = (param["domain"][dmax] - param["domain"][dmin])

    t = 0
    total_out_flow = 0
    for g, d in gb:
        b_faces = g.tags["domain_boundary_faces"]
        if np.any(b_faces):
            faces, cells, sign = sps.find(g.cell_faces)
            index = np.argsort(cells)
            faces, cells, sign = faces[index], cells[index], sign[index]

            u = d["darcy_flux"].copy()
            u[np.logical_not(b_faces)] = 0
            u[faces] *= sign

            # it's ok since we consider only the boundary
            aperture = np.power(param["aperture"], 2 - g.dim) * np.ones(g.num_cells)

            out_flow = g.face_centers[flow_direction, :] > param["domain"][dmax] - param["tol"]
            out_flow_u = np.sum(u[out_flow])

            t += out_flow_u * delta / param["bc_flow"]
            total_out_flow += out_flow_u

    return t, total_out_flow

# ------------------------------------------------------------------------------#
