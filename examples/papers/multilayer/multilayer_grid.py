import copy
import numpy as np
import scipy.sparse as sps
import porepy as pp


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
