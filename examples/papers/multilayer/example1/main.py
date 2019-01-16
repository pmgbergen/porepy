import numpy as np
import scipy.sparse as sps
import copy

import porepy as pp

import examples.papers.multilayer.discretization as compute

np.set_printoptions(linewidth=2000)

def bc_flag(g, domain, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    # define outflow type boundary conditions
    out_flow_y = b_face_centers[1] > 1 - tol
    #out_flow_x = b_face_centers[0] < tol

    #out_flow = np.logical_and(out_flow_x, out_flow_z)
    out_flow = out_flow_y

    # define inflow type boundary conditions
    in_flow_y = b_face_centers[1] < 0 + tol
    in_flow = in_flow_y

    return in_flow, out_flow


def unite_grids(gs):
    # it's simple index transformation
    gs = np.atleast_1d(gs)

    # collect the nodes
    nodes = np.hstack((g.nodes for g in gs))

    # collect the face_nodes map
    face_nodes = sps.block_diag([g.face_nodes for g in gs], "csc")

    # collect the cell_faces map
    cell_faces = sps.block_diag([g.cell_faces for g in gs], "csc")

    # dimension and name
    dim = gs[0].dim
    name = "United grid"

    # create the grid and give it back
    g = pp.Grid(dim, nodes, face_nodes, cell_faces, name)
    g.compute_geometry()
    return g

def multilayer_grid_bucket(gb):
    # we assume conforming grids and no 0d grids

    gb_new = pp.GridBucket()
    gb_new.add_nodes([g for g, _ in gb])

    for e, d in gb.edges():
        mg = d["mortar_grid"]
        gs, gm = gb.nodes_of_edge(e)

        # we construct the grid for the new layer
        g_new = unite_grids([g for g in mg.side_grids.values()])

        # update the names
        gs.name += ["fault"]
        g_new.name += ["layer"]

        # add the new grid to the grid bucket
        gb_new.add_nodes(g_new)

        # we store the master-grid with the new layer grid with the
        # mortar mapping from the original edge
        edge = [gm, g_new]
        gb_new.add_edge(edge, None)

        mg1 = copy.deepcopy(mg)
        # the slave to mortar needs to be changed
        mg1.slave_to_mortar_int = sps.identity(g_new.num_cells, format="csc")
        gb_new.set_edge_prop(edge, "mortar_grid", mg1)

        # we store the slave-grid with the new layer grid with the
        # mortar mapping from the original edge
        edge = [g_new, gs]
        gb_new.add_edge(edge, None)

        mg2 = copy.deepcopy(mg)
        # the master to mortar needs to be changed
        mg2.master_to_mortar_int = sps.identity(g_new.num_cells, format="csc")
        gb_new.set_edge_prop(edge, "mortar_grid", mg2)

    gb_new.assign_node_ordering()

    return gb_new

def main():

    h = 0.01
    tol = 1e-6
    mesh_args = {'mesh_size_frac': h}
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

    f = np.array([[0, 1], [0.5, 0.5]])

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
             "layer": {"aperture": 1e-3, "kf_t": 1, "kf_n": 1e2},
             "fault": {"aperture": 1e-3, "kf_t": 1, "kf_n": 1e-2},
             "folder": "solution"}
    compute.flow(gb_ml, param, bc_flag)

if __name__ == "__main__":
    main()
