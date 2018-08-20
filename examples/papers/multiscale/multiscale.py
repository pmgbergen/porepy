import numpy as np
import scipy.sparse as sps
import porepy as pp

class Multiscale(object):

    def __init__(self, gb):
        # The grid bucket
        self.gb = gb
        # Higher dimensional grid, assumed to be 1
        self.g_h = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        # Number of dofs for the higher dimensional grid
        self.dof_h = None
        # Diffusion matrix of the higher dimensional domain
        self.A_h = None
        # Right-hand side of the higher dimensional domain
        self.b_h = None
        # Couple the 1 co-dimensional pressure to the mortar variables
        self.C_h = None
        # Non-homogeneous solution
        self.x_h = None
        # Bases
        self.bases = None
        # LU factorization of the higher dimensional matrix
        self.LU = None
        # Position of the mortar 1 co-dimensional mortar variables
        self.if_p = None

        # Co-dimensional grid positions
        self.pos_ln = None
        # Positions of the mortars blocks for the co-dimensional grids
        self.pos_le = None
        # Realise the jump operator given the mortar variables
        self.C_l = None
        # Number of dofs for the co-dimensional grids
        self.dof_l = None

#------------------------------------------------------------------------------#

    def extract_blocks_h(self, A, b):
        # Select the position in the matrix A of the higher dimensional domain
        pos_hn = [self.gb.node_props(self.g_h, "node_number")]
        # Determine the number of dofs for the higher dimensional grid
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
        self.b_h = np.r_[tuple(b[pos_hn + pos_he])]

        # Couple the 1 co-dimensional pressure to the mortar variables
        self.C_h = sps.bmat(A[np.ix_(pos_he, self.pos_ln)])

        # Realise the jump operator given the mortar variables
        self.C_l = sps.bmat(A[np.ix_(self.pos_ln, pos_he)])

#------------------------------------------------------------------------------#

    def compute_bases(self):

        # construct the rhs with all the active pressure dof in the 1 co-dimension
        self.if_p = np.zeros(self.C_h.shape[1], dtype=np.bool)
        pos = 0
        for g, d in self.gb:
            if g.dim != self.g_h.dim:
                pos += g.num_faces
                # only the 1 co-dimensional grids are interesting
                if g.dim == self.g_h.dim - 1:
                    self.if_p[pos : pos + g.num_cells] = True
                pos += g.num_cells

        # compute the bases
        num_bases = np.sum(self.if_p)
        dof_bases = self.C_h.shape[0]
        self.bases = np.zeros((self.C_h.shape[1], self.C_h.shape[1]))

        # we solve many times the same problem, better to factorize the matrix
        self.LU = sps.linalg.factorized(self.A_h.tocsc())

        # solve to compute the ms bases functions for homogeneous boundary
        # conditions
        for dof_basis in np.where(self.if_p)[0]:
            rhs = np.zeros(self.if_p.size)
            rhs[dof_basis] = 1.
            # project from the co-dimensional pressure to the Robin boundary
            # condition
            rhs = np.r_[[0]*self.dof_h, -self.C_h * rhs]
            # compute the jump of the mortars
            self.bases[:, dof_basis] = self.C_l * self.LU(rhs)[-dof_bases:]

            # save = pp.Exporter(g_h, "vem"+str(dof_basis), folder="vem_ms")
            # save.write_vtk({'pressure': x[g_h.num_faces:dof_h]})

        # save the bases in a csr format
        self.bases = sps.csr_matrix(self.bases)
        self.dof_l = self.bases.shape[0]

        # solve for non-zero boundary conditions
        self.x_h = - self.C_l * self.LU(self.b_h)[-dof_bases:]

        return {"solve_h": num_bases+1}

#------------------------------------------------------------------------------#

    def solve_l(self, A, b):
        # construct the problem in the fracture network
        A_l = np.empty((2, 2), dtype=np.object)
        b_l = np.empty(2, dtype=np.object)

        # the multiscale bases are thus inserted in the right block of the lower
        # dimensional problem
        A_l[0, 0] = sps.bmat(A[np.ix_(self.pos_ln, self.pos_ln)])
        A_l[0, 0] += self.bases
        # add to the righ-hand side the non-homogenous solution from the higher
        # dimensional problem
        b_l[0] = np.r_[tuple(b[self.pos_ln])] + self.x_h
        # in the case of > 1 co-dimensional problems
        if len(self.pos_le) > 0:
            A_l[0, 1] = sps.bmat(A[np.ix_(self.pos_ln, self.pos_le)])
            A_l[1, 0] = sps.bmat(A[np.ix_(self.pos_le, self.pos_ln)])
            A_l[1, 1] = sps.bmat(A[np.ix_(self.pos_le, self.pos_le)])
            b_l[1] = np.r_[tuple(b[self.pos_le])]
        else:
            b_l[1] = np.empty(0)

        # assemble and return
        A_l = sps.bmat(A_l, "csr")
        b_l = np.r_[tuple(b_l)]

        return sps.linalg.spsolve(A_l, b_l)

#------------------------------------------------------------------------------#

    def solve_h(self, x_l):
        # compute the higher dimensional solution
        b = np.r_[[0]*self.dof_h, -self.C_h * x_l[:self.C_h.shape[1]]]
        return self.LU(self.b_h + b)

#------------------------------------------------------------------------------#

    def concatenate(self, x_h, x_l):
        # save and export using standard algorithm
        x = np.zeros(x_h.size + x_l.size)
        x[:self.dof_h] = x_h[:self.dof_h]
        x[self.dof_h : (self.dof_h + self.dof_l)] = x_l[:self.dof_l]
        return x

#------------------------------------------------------------------------------#
