import numpy as np
import scipy.sparse as sps
import porepy as pp

iteration_number = 0

class DomainDecomposition(object):

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

        # Realise the jump operator given the mortar variables
        self.C_l = None
        # Number of dofs for the co-dimensional grids
        self.dof_l = None

#------------------------------------------------------------------------------#

    def ndof(self):
        return self.b_h.size + self.b_l.size

#------------------------------------------------------------------------------#

    def extract_blocks(self, A, b):

        # Select the position in the matrix A of the higher dimensional domain
        pos_hn = [self.gb.node_props(self.g_h, "node_number")]
        # Determine the number of dofs for the higher dimensional grid
        self.dof_h = A[pos_hn[0], pos_hn[0]].shape[0]

        # Select the co-dimensional grid positions
        pos_ln = [d["node_number"] for g, d in self.gb if g.dim < self.g_h.dim]

        # select the positions of the mortars blocks, for both the high dimensional
        # and lower dimensional grids
        num_nodes = self.gb.num_graph_nodes()
        pos_he = []
        pos_le = []
        for e, d in self.gb.edges():
            gl_h = self.gb.nodes_of_edge(e)[1]
            if gl_h.dim == self.g_h.dim:
                pos_he.append(d["edge_number"] + num_nodes)
            else:
                pos_le.append(d["edge_number"] + num_nodes)

        # extract the blocks for the higher dimension
        self.A_h = sps.bmat(A[np.ix_(pos_hn, pos_hn)])
        self.b_h = np.concatenate(tuple(b[pos_hn]))

        self.C_mh = sps.bmat(A[np.ix_(pos_he, pos_hn)])
        self.C_hm = sps.bmat(A[np.ix_(pos_hn, pos_he)])

        # extract the block for the 1-codimensional mortar variable
        self.A_m = sps.bmat(A[np.ix_(pos_he, pos_he)])
        self.b_m = np.concatenate(tuple(b[pos_he]))

        # construct the problem in the fracture network
        A_l = np.empty((2, 2), dtype=np.object)
        b_l = np.empty(2, dtype=np.object)

        # coupling matrices
        C_ml = np.empty((2, 2), dtype=np.object)
        C_lm = np.empty((2, 2), dtype=np.object)

        A_l[0, 0] = sps.bmat(A[np.ix_(pos_ln, pos_ln)])
        self.dof_l = A_l[0, 0].shape[0]

        C_ml[0, 0] = sps.bmat(A[np.ix_(pos_he, pos_ln)])
        C_lm[0, 0] = sps.bmat(A[np.ix_(pos_ln, pos_he)])

        # add to the righ-hand side the non-homogenous solution from the higher
        # dimensional problem
        b_l[0] = np.concatenate(tuple(b[pos_ln]))

        # in the case of > 1 co-dimensional problems
        if len(pos_le) > 0:
            A_l[0, 1] = sps.bmat(A[np.ix_(pos_ln, pos_le)])
            A_l[1, 0] = sps.bmat(A[np.ix_(pos_le, pos_ln)])
            A_l[1, 1] = sps.bmat(A[np.ix_(pos_le, pos_le)])

            C_ml[1, 1] = sps.bmat(A[np.ix_(pos_he, pos_he)])

            C_ml[1, 1] = sps.bmat(A[np.ix_(pos_le, pos_le)])

            b_l[1] = np.concatenate(tuple(b[pos_le]))
        else:
            b_l[1] = np.empty(0)

        print(pos_le)
        print(pos_ln)
        print(b_l.shape)

        # assemble and return
        self.A_l = sps.bmat(A_l, "csr")
        self.b_l = np.concatenate(tuple(b_l))

#------------------------------------------------------------------------------#

    def solve(self, tol, maxiter):

        # we solve many times the same problem, better to factorize the matrix
        LU_h = sps.linalg.factorized(self.A_h.tocsc())
        LU_l = sps.linalg.factorized(self.A_l.tocsc())

        # compute the right-hand side
        b_h = LU_h(-self.b_h)
        b_l = LU_l(-self.b_l)
        print(b_l.size)
        print(self.C_ml.shape)
        b = self.b_m + self.C_mh * b_h + self.C_ml * b_l

        def schur_complement(x_m):

            # given a mortar solution compute the high dimensional
            x_h = LU_h(-self.C_hm * x_m)
            # given a mortar solution compute the co-dimensional
            x_l = LU_l(-self.C_lm * x_m)
            # construct the resulting M*x_m
            return self.A_m * x_m + self.C_mh * x_h + self.C_ml * x_l

        def callback(x):
            global iteration_number
            iteration_number += 1

        M = sps.linalg.LinearOperator(self.A_m.shape, schur_complement)
        x_m, info = sps.linalg.gmres(M, b, tol=tol, maxiter=maxiter,
                                     callback=callback)
        print(info, iteration_number)

        # given a mortar solution compute the high dimensional
        x_h = LU_h(self.b_h - self.C_hm * x_m)
        # given a mortar solution compute the co-dimensional
        x_l = LU_l(self.b_l - self.C_lm * x_m)

        x = self.concatenate(x_h, x_l)

        return x

#------------------------------------------------------------------------------#

    def error(self, x, x_old):
        norm = np.linalg.norm(x_old)
        return np.linalg.norm(x - x_old)/(norm if norm else 1)

#------------------------------------------------------------------------------#

    def split(self, x):
        x_h = np.zeros(self.A_h.shape[0])
        x_l = np.zeros(self.A_l.shape[0])

        x_h[:self.dof_h] = x[:self.dof_h]
        x_l[:self.dof_l] = x[self.dof_h : (self.dof_h + self.dof_l)]

        return x_h, x_l

#------------------------------------------------------------------------------#

    def concatenate(self, x_h, x_l):
        # save and export using standard algorithm, the mortar variable is not
        # yet exported
        x = np.zeros(x_h.size + x_l.size)
        x[:self.dof_h] = x_h[:self.dof_h]
        x[self.dof_h : (self.dof_h + self.dof_l)] = x_l[:self.dof_l]
        return x

#------------------------------------------------------------------------------#
