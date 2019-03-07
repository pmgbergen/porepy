import numpy as np
import porepy as pp

import examples.papers.arXiv_1903_01117.discretization as compute
from examples.papers.arXiv_1903_01117.multilayer_grid import multilayer_grid_bucket

from examples.papers.arXiv_1903_01117.example2.import_grid import import_grid

np.set_printoptions(linewidth=2000)


def bc_flag(g, data, tol):

    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    b_top = np.logical_and(b_face_centers[0] < 0 + tol, b_face_centers[2] > 90 - tol)

    b_bottom = np.logical_and(b_face_centers[1] < 0 + tol, b_face_centers[2] < 10 + tol)

    labels = np.array(["neu"] * b_faces.size)
    labels[b_top] = "dir"
    labels[b_bottom] = "dir"

    bc_val = np.zeros(g.num_faces)
    bc_val[b_faces[b_top]] = 4 * pp.METER
    bc_val[b_faces[b_bottom]] = 1 * pp.METER

    return labels, bc_val


def low_zones(g):
    return g.cell_centers[2, :] > 10


def main():

    tol = 1e-6
    gb, domain = import_grid("mesh10k.geo", tol)

    # construct the multi-layer grid bucket, we give also a name to the fault and layer grids
    gb_ml = multilayer_grid_bucket(gb)
    case = "case1"

    # the flow problem
    param = {
        "domain": domain,
        "tol": tol,
        "k": None,
        "layer": {"aperture": 1e-1, "kf_t": 1e-1, "kf_n": 1e-1},
        "fault": {"aperture": 1e-3, "kf_t": 1e-7, "kf_n": 1e-7},
        "folder": case,
    }

    # define the non-constant tangential permeability
    for g in gb_ml.grids_of_dimension(2):

        kf = np.ones(g.num_cells)

        if "layer" in g.name:

            # we assume that the first half of the cells belongs to a layer, which is normally true.
            # set the permeability for the first layer
            half_cells = int(g.num_cells / 2)
            layer1 = np.hstack((np.ones(half_cells), np.zeros(half_cells))).astype(bool)
            kf[layer1] = 1e-2

            # set the permeability for the second layer
            layer2 = np.logical_not(layer1)
            kf[layer2] = 1e-1

            param["layer"]["kf_t"] = kf

    # define the non-constant normal permeability
    for e, d in gb_ml.edges():
        mg = d["mortar_grid"]
        g_l, g_h = gb_ml.nodes_of_edge(e)

        kf = np.ones(mg.num_cells)

        # we assume that the first half of the cells belongs to a layer, which is normally true.
        # set the permeability for the first layer
        half_cells = int(mg.num_cells / 2)
        layer1 = np.hstack((np.ones(half_cells), np.zeros(half_cells))).astype(bool)
        kf[layer1] = 1e-2

        # set the permeability for the second layer
        layer2 = np.logical_not(layer1)
        kf[layer2] = 1e-1

        if not ("fault" in g_l.name or "fault" in g_h.name):
            param["layer"]["kf_n"] = kf

    # set the low zone permeability for the rock matrix
    for g in gb_ml.grids_of_dimension(3):
        param["k"] = 1e-5 * np.ones(g.num_cells)
        param["k"][low_zones(g)] = 1e-6

    # solve the Darcy problem
    compute.flow(gb_ml, param, bc_flag)

    # for g, d in gb_ml:
    #    if g.dim == 2:
    #        pressure = param["pressure"]
    #        np.savetxt("pressure.txt", d[pressure])


if __name__ == "__main__":
    main()
