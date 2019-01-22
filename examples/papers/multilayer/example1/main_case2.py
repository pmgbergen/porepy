import numpy as np
import porepy as pp

import examples.papers.multilayer.discretization as compute
from examples.papers.multilayer.multilayer_grid import multilayer_grid_bucket

np.set_printoptions(linewidth=2000)

def bc_flag(g, data, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    # define outflow type boundary conditions
    out_flow_y = b_face_centers[1] > 2 - tol
    #out_flow_x = b_face_centers[0] < tol

    #out_flow = np.logical_and(out_flow_x, out_flow_z)
    out_flow = out_flow_y

    # define inflow type boundary conditions
    in_flow = b_face_centers[1] < 0 + tol

    # define the labels for the boundary faces
    labels = np.array(["neu"] * b_faces.size)
    if g.dim == 2:
        labels[in_flow + out_flow] = "dir"
    else:
        labels[:] = "dir"

    # define the values for the boundary faces
    bc_val = np.zeros(g.num_faces)
    if g.dim == 2:
        bc_val[b_faces[in_flow]] = data["bc_inflow"]
        bc_val[b_faces[out_flow]] = data["bc_outflow"]
    else:
        bc_val[b_faces] = (b_face_centers[0, :] < 0.5).astype(np.float)

    return labels, bc_val

def main():

    h = 0.025
    tol = 1e-6
    mesh_args = {'mesh_size_frac': h}
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 2}

    f = np.array([[0, 1], [1, 1]])

    gb = pp.meshing.simplex_grid([f], domain, **mesh_args)

    # construct the multi-layer grid bucket, we give also a name to the fault and layer grids
    gb_ml = multilayer_grid_bucket(gb)
    #pp.plot_grid(gb_ml, alpha=0, info="cf")

    # nella discretizzazione tra master e nuovo strato devo mettere una legge di tipo
    # robin come nel caso usuale

    # mentre nel caso di discretizzazioen tra nuovo strato e slave devo mettere una
    # legge tipo robin in cui devo scrivere meglio dato che non ho tracce della pressione
    # ma variabile di cella e devo capire come si comporta il caso in  cui ho la variabile di
    # flusso che interagisce tra i due.

    # the flow problem
    param = {"domain": gb_ml.bounding_box(as_dict=True), "tol": tol,
             "k": 1, "bc_inflow": 0, "bc_outflow": 1,
             "layer": {"aperture": 1e-2, "kf_t": 1e2, "kf_n": 1e2},
             "fault": {"aperture": 1e-2, "kf_t": 1e2, "kf_n": 1e2},
             "folder": "case2"}
    compute.flow(gb_ml, param, bc_flag)

    for g, d in gb_ml:
        if g.dim == 2:
            pressure = param["pressure"]
            np.savetxt("pressure.txt", d[pressure])

if __name__ == "__main__":
    main()
