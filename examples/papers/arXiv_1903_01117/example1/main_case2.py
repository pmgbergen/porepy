import numpy as np
import porepy as pp

# from shutil import copyfile

import examples.papers.arXiv_1903_01117.discretization as compute
from examples.papers.arXiv_1903_01117.multilayer_grid import multilayer_grid_bucket

np.set_printoptions(linewidth=2000)


def bc_flag(g, data, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    # define outflow type boundary conditions
    out_flow = b_face_centers[1] > 2 - tol

    # define inflow type boundary conditions
    in_flow = b_face_centers[1] < 0 + tol

    # define the labels and values for the boundary faces
    labels = np.array(["neu"] * b_faces.size)
    bc_val = np.zeros(g.num_faces)

    if g.dim == 2:
        labels[in_flow + out_flow] = "dir"
        bc_val[b_faces[in_flow]] = 0
        bc_val[b_faces[out_flow]] = 1

    return labels, bc_val


def main():

    h = 0.025  # 0.0125
    tol = 1e-6
    mesh_args = {"mesh_size_frac": h}
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 2}

    # Point coordinates, as a 2xn array
    p = np.array([[0, 1], [1, 1]])

    # Point connections as a 2 x num_frac arary
    e = np.array([[0], [1]])

    # Define a fracture network in 2d
    network_2d = pp.FractureNetwork2d(p, e, domain)

    # Generate a mixed-dimensional mesh
    gb = network_2d.mesh(mesh_args)

    # construct the multi-layer grid bucket, we give also a name to the fault and layer grids
    gb_ml = multilayer_grid_bucket(gb)
    # pp.plot_grid(gb_ml, alpha=0, info="cf")

    # folder_export = "/home/elle/Dropbox/Work/PresentazioniArticoli/2019/Articles/multilayer/results/"
    case = "case2"
    aperture = 2.5 * 1e-3  # 10*e-3, 5*1e-3, 2.5*1e-3

    # the flow problem
    param = {
        "domain": gb_ml.bounding_box(as_dict=True),
        "tol": tol,
        "k": 1,
        "bc_inflow": 0,
        "bc_outflow": 1,
        "layer": {"aperture": aperture, "kf_t": None, "kf_n": None},
        "fault": {"aperture": aperture, "kf_t": None, "kf_n": None},
        "folder": case,
    }

    # define the non-constant tangential permeability
    for g in gb_ml.grids_of_dimension(1):
        kf = np.ones(g.num_cells)
        mask = np.logical_and(g.cell_centers[0] < 0.75, g.cell_centers[0] > 0.25)
        kf[mask] = 2e-3

        if "layer" in g.name:
            param["layer"]["kf_t"] = kf
        elif "fault" in g.name:
            param["fault"]["kf_t"] = kf

    # define the non-constant normal permeability
    for e, d in gb_ml.edges():
        mg = d["mortar_grid"]
        kf = np.ones(mg.num_cells)
        mask = np.logical_and(mg.cell_centers[0] < 0.75, g.cell_centers[0] > 0.25)
        kf[mask] = 2e-3

        g_l, g_h = gb_ml.nodes_of_edge(e)
        if "fault" in g_l.name or "fault" in g_h.name:
            param["fault"]["kf_n"] = kf
        else:
            param["layer"]["kf_n"] = kf

    # solve the Darcy problem
    compute.flow(gb_ml, param, bc_flag)

    # for g, d in gb_ml:
    #    if g.dim == 2:
    #        pressure = param["pressure"]
    #        name_root = folder_export + case + "_" + str(aperture)
    #        np.savetxt(name_root + "_pressure.txt", d[pressure])
    #        copyfile("gmsh_frac_file.msh", name_root + "_grid.msh")


if __name__ == "__main__":
    main()
