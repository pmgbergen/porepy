import numpy as np
import scipy.sparse as sps
import porepy as pp


class FvSubGrid(pp.Grid):
    def __init__(self, g, eta):
        subcell_topology = pp.numerics.fv.fvutils.SubcellTopology(g)

        # we collect all nodes in the subgrid. This will be all nodes of the
        # grid + the face-centers + the cell_centers
        # First we find the mapping from sub_faces to the nodes,
        # which is just a 1-1 mapping
        nodes = g.nodes.copy()

        nno_unique = subcell_topology.nno_unique
        sf_n_indptr = np.arange(nno_unique.size+1)
        data = np.ones(nno_unique.size, dtype=np.bool)
        subface_to_nodes  = sps.csc_matrix((data, nno_unique, sf_n_indptr))
        # We now add the mapping from sub_cells to nodes
        sc_n_indices = g.cell_nodes().indices
        sc_n_indptr = np.arange(sc_n_indices.size+1)
        data = np.ones(sc_n_indices.size, dtype=np.bool)
        subcell_to_nodes = sps.csc_matrix((data, sc_n_indices, sc_n_indptr))
        # And from this obtain the subcell to subface mapping
        subcell_to_subfaces = (subface_to_nodes.T * subcell_to_nodes).tocsc()

        # As the face centers we use the continuity point
        f_c = g.face_centers[:, subcell_topology.fno_unique]
        sub_nodes = g.nodes[:, subcell_topology.nno_unique]
        self.face_centers = f_c + eta* (sub_nodes - f_c)
        # Finally we call the parrent constructor
        name = g.name.copy()
        name.append('FvSubGrid')
        pp.Grid.__init__(self, g.dim, nodes, subface_to_nodes, subcell_to_subfaces, name)
