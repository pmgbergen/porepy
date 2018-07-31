import numpy as np
import scipy.sparse as sps
import porepy as pp

class Multiscale(object):

    def __init__(self, gb):
        self.gb = gb

        # Number of dofs for the higher dimensional grid
        self.g_h = None
        self.dof_h = None
        self.A_h = None
        self.b_h = None
        self.C_h = None
        self.x_h = None
        self.bases = None

        # Co-dimensional grid positions
        self.pos_ln = None
        # Positions of the mortars blocks
        self.pos_le = None
        self.C_l = None

    def extract_blocks_h(self, A, b):
        # Select the grid of higher dimension and the position in the matrix A
        self.g_h = self.gb.grids_of_dimension(gb.dim_max())[0]
        pos_hn = [self.gb.node_props(self.g_h, "node_number")]
        # Determine the number of dofs [u, p] for the higher dimensional grid
        self.dof_h = A[pos_hn[0], pos_hn[0]].shape[0]

        # Select the co-dimensional grid positions
        self.pos_ln = [d["node_number"] for g, d in self.gb if g.dim < self.g_h.dim]

        # select the positions of the mortars blocks, for both the high dimensional
        # and lower dimensional grids
        num_nodes = self.gb.num_graph_nodes()
        pos_he = []
        self.pos_le = []
        for e, d in self.gb.edges():
            gl_h = self.gb.nodes_of_edge(e)[1]
            if gl_h.dim == self.g_h.dim:
                pos_he.append(d["edge_number"] + num_nodes)
            else:
                self.pos_le.append(d["edge_number"] + num_nodes)

        # extract the blocks for the higher dimension
        self.A_h = sps.bmat(A[np.ix_(pos_hn + pos_he, pos_hn + pos_he)])
        self.b_h = np.concatenate(tuple(b[pos_hn + pos_he]))

        # matrix that couple the 1 co-dimensional pressure to the mortar variables
        self.C_h = sps.bmat(A[np.ix_(pos_he, pos_ln)])

        # Select the grid that realise the jump operator given the mortar variables
        self.C_l = sps.bmat(A[np.ix_(pos_ln, pos_he)])

#------------------------------------------------------------------------------#

def compute_bases(is_mixed):
    """
    is_mixed: if the numerical scheme is in mixed form, useful to impose the
            boundary conditions
    """

    # construct the rhs with all the active pressure dof in the 1 co-dimension
    if_p = np.zeros(self.C_h.shape[1], dtype=np.bool)
    pos = 0
    for g, d in self.gb:
        if g.dim != self.g_h.dim:
            if is_mixed: # shift the dofs in case of mixed method
                pos += g.num_faces
            # only the 1 co-dimensional grids are interesting
            if g.dim == self.g_h.dim - 1:
                if_p[pos : pos + g.num_cells] = True
            pos += g.num_cells

    # compute the bases
    num_bases = np.sum(if_p)
    dof_bases = self.C_h.shape[0]
    self.bases = np.zeros((self.C_h.shape[1], self.C_h.shape[1]))

    # we solve many times the same problem, better to factorize the matrix
    LU = sps.linalg.factorized(self.A_h.tocsc())

    # solve to compute the ms bases functions for homogeneous boundary
    # conditions
    for dof_basis in np.where(if_p)[0]:
        rhs = np.zeros(if_p.size)
        rhs[dof_basis] = 1.
        # project from the co-dimensional pressure to the Robin boundary
        # condition
        rhs = np.concatenate((np.zeros(self.dof_h), -self.C_h * rhs))
        x = LU(rhs)
        # compute the jump of the mortars
        self.bases[:, dof_basis] = self.C_l * x[-dof_bases:]

        # save = pp.Exporter(g_h, "vem"+str(dof_basis), folder="vem_ms")
        # save.write_vtk({'pressure': x[g_h.num_faces:dof_h]})

    # solve for non-zero boundary conditions
    self.x_h = LU(b_h)

