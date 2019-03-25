import numpy as np

# from shutil import copyfile

import porepy as pp

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
    else:
        labels[:] = "dir"
        bc_val[b_faces] = (b_face_centers[0, :] < 0.5).astype(np.float)

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
    case = "case1"
    aperture = 10 * 1e-3  # 10*e-3, 5*1e-3, 2.5*1e-3

    # the flow problem
    param = {
        "domain": gb_ml.bounding_box(as_dict=True),
        "tol": tol,
        "k": 1,
        "layer": {"aperture": aperture, "kf_t": 1e2, "kf_n": 1e2},
        "fault": {"aperture": aperture, "kf_t": 1e2, "kf_n": 1e2},
        "folder": case,
    }

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
