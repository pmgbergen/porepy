"""
This is a module for creating the sub-grids used in the multi-point finite
volume discretizations (MPFA MPSA). Note that this is experimental and should
not be excpected to work with standard porepy functions (e.g., g.compute_geometry()).

Intended use is for now to set sub-face boundary conditions.
"""

import numpy as np
import scipy.sparse as sps

import porepy as pp


class FvSubGrid(pp.Grid):
    """
    Define a subgrid for the finite volume discretizations. Each cell is divided
    up into subcells according to the nodes. A 2D Cartesian cell is divided in
    four sub-cells while a 2D simplex cell is divided into three cells. The
    face-centers of the sub-grid will be the continuity point as given by eta.
    This should correspond the the eta given to the finite volume discretization.

    """

    def __init__(self, g, eta):
        """
        Initialization of grid.

        Parameters:
        -----------
        g (pp.Grid): A grid that will be divided into subcells.
        eta (np.ndarray): Defines the face centers of the sub-grid (continuity point
            of the finite volume methods). eta=0 retrives the face centers of g, while
            eta=1 sets the face centers of the subgrid equal the nodes of g.

        Returns:
        --------
        pp.Grid: The computed subgrid.
        """
        subcell_topology = pp.numerics.fv.fvutils.SubcellTopology(g)
        # we collect all nodes in the subgrid. This will be all nodes of the
        # grid + the face-centers + the cell_centers
        # First we find the mapping from sub_faces to the nodes,
        # which is just a 1-1 mapping
        nodes = g.nodes.copy()

        nno_unique = subcell_topology.nno_unique
        sf_n_indptr = np.arange(nno_unique.size + 1)
        data = np.ones(nno_unique.size, dtype=np.bool)
        subface_to_nodes = sps.csc_matrix((data, nno_unique, sf_n_indptr))
        # We now add the mapping from sub_cells to nodes
        sc_n_indices = g.cell_nodes().indices
        sc_n_indptr = np.arange(sc_n_indices.size + 1)
        data = np.ones(sc_n_indices.size, dtype=np.bool)
        subcell_to_nodes = sps.csc_matrix((data, sc_n_indices, sc_n_indptr))
        # And from this obtain the subcell to subface mapping
        subcell_to_subfaces = (subface_to_nodes.T * subcell_to_nodes).tocsc()
        # flip one sign of duplicate subfaces
        _, ix = np.unique(subcell_to_subfaces.indices, return_index=True)
        subcell_to_subfaces.data = subcell_to_subfaces.data.astype(np.int)
        subcell_to_subfaces.data[ix] *= -1

        # As the face centers we use the continuity point
        f_c = g.face_centers[:, subcell_topology.fno_unique]
        sub_nodes = g.nodes[:, subcell_topology.nno_unique]
        self.face_centers = f_c + eta * (sub_nodes - f_c)
        # Finally we call the parrent constructor
        name = g.name.copy()
        name.append("FvSubGrid")
        pp.Grid.__init__(
            self, g.dim, nodes, subface_to_nodes, subcell_to_subfaces, name
        )

        num_nodes_per_face = np.diff(g.face_nodes.indptr)
        self.face_areas = g.face_areas / num_nodes_per_face
        self.face_areas = self.face_areas[subcell_topology.fno_unique]
        if hasattr(g, "frac_pairs"):
            is_primary = np.zeros(g.num_faces, dtype=np.bool)
            is_secondary = np.zeros(g.num_faces, dtype=np.bool)
            is_primary[g.frac_pairs[0]] = True
            is_secondary[g.frac_pairs[1]] = True
            is_primary_hf = is_primary[subcell_topology.fno_unique]
            is_secondary_hf = is_secondary[subcell_topology.fno_unique]
            self.frac_pairs = np.vstack(
                (np.where(is_primary_hf)[0], np.where(is_secondary_hf)[0])
            )
