import numpy as np
import scipy.sparse as sps

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
        # LU factorization of the higher dimensional problem
        self.LU = None

        # Realise the jump operator given the mortar variables
        self.C_l = None
        # Number of dofs for the co-dimensional grids
        self.dof_l = None

        # Position in the matrix A of the higher dimensional domain
        self.pos_hn = None
        # Position in the matrix A of the lower dimensional domain
        self.pos_ln = None
        # Position in the matrix A of the mortar higher dimensional domain
        self.pos_he = None
        # Position in the matrix A of the mortar lower dimensional domain
        self.pos_le = None

    # ------------------------------------------------------------------------------#

    def ndof(self):
        return self.b_h.size + self.b_l.size

    # ------------------------------------------------------------------------------#

    def extract_blocks(self, A, b):

        # Select the position in the matrix A of the higher dimensional domain
        pos_hn = [self.gb.node_props(self.g_h, "node_number")]
        self.pos_hn = pos_hn
        # Determine the number of dofs for the higher dimensional grid
        self.dof_h = A[pos_hn[0], pos_hn[0]].shape[0]

        # Select the co-dimensional grid positions
        pos_ln = [d["node_number"] for g, d in self.gb if g.dim < self.g_h.dim]
        self.pos_ln = pos_ln

        # select the positions of the mortars blocks, for both the high
        # dimensional and lower dimensional grids
        num_nodes = self.gb.num_graph_nodes()
        pos_he = []
        pos_le = []
        for e, d in self.gb.edges():
            gl_h = self.gb.nodes_of_edge(e)[1]
            if gl_h.dim == self.g_h.dim:
                pos_he.append(d["edge_number"] + num_nodes)
            else:
                pos_le.append(d["edge_number"] + num_nodes)

        self.pos_he = pos_he
        self.pos_le = pos_le

        # extract the blocks for the higher dimension
        self.A_h = sps.bmat(A[np.ix_(pos_hn + pos_he, pos_hn + pos_he)])
        self.b_h = np.r_[tuple(b[pos_hn + pos_he])]

        # Couple the 1 co-dimensional pressure to the mortar variables
        self.C_h = sps.bmat(A[np.ix_(pos_he, pos_ln)])

        # Realise the jump operator given the mortar variables
        self.C_l = sps.bmat(A[np.ix_(pos_ln, pos_he)])

        # construct the problem in the fracture network
        A_l = np.empty((2, 2), dtype=np.object)
        b_l = np.empty(2, dtype=np.object)

        A_l[0, 0] = sps.bmat(A[np.ix_(pos_ln, pos_ln)])
        self.dof_l = A_l[0, 0].shape[0]

        # add to the right-hand side the non-homogeneous solution from the
        # higher dimensional problem
        b_l[0] = np.r_[tuple(b[pos_ln])]

        # in the case of > 1 co-dimensional problems
        if len(pos_le) > 0:
            A_l[0, 1] = sps.bmat(A[np.ix_(pos_ln, pos_le)])
            A_l[1, 0] = sps.bmat(A[np.ix_(pos_le, pos_ln)])
            A_l[1, 1] = sps.bmat(A[np.ix_(pos_le, pos_le)])
            b_l[1] = np.r_[tuple(b[pos_le])]
        else:
            b_l[1] = np.empty(0)

        # assemble and return
        self.A_l = sps.bmat(A_l, "csr")
        self.b_l = np.r_[tuple(b_l)]

    # ------------------------------------------------------------------------------#

    def update_lower_blocks(self, A):

        # construct the problem in the fracture network
        A_l = np.empty((2, 2), dtype=np.object)
        A_l[0, 0] = sps.bmat(A[np.ix_(self.pos_ln, self.pos_ln)])

        # in the case of > 1 co-dimensional problems
        if len(self.pos_le) > 0:
            A_l[0, 1] = sps.bmat(A[np.ix_(self.pos_ln, self.pos_le)])
            A_l[1, 0] = sps.bmat(A[np.ix_(self.pos_le, self.pos_ln)])
            A_l[1, 1] = sps.bmat(A[np.ix_(self.pos_le, self.pos_le)])

        # assemble and factorize
        self.A_l = sps.bmat(A_l, "csr")

    # ------------------------------------------------------------------------------#

    def solve(self, tol, maxiter, info=False, return_lower=False):

        global iteration_number
        iteration_number = 0

        def callback(x):
            global iteration_number
            iteration_number += 1

        # construct the matrix and preconditioner
        A = sps.linalg.LinearOperator(self.A_l.shape, self.schur_complement)
        P = sps.linalg.splu(self.A_l.tocsc())
        M = sps.linalg.LinearOperator(self.A_l.shape, lambda x: P.solve(x))

        # construct the right-hand side
        x_h = self.LU(self.b_h)
        b_h = np.zeros(self.b_l.size)
        b_h[: self.dof_l] = -self.C_l * x_h[self.dof_h :]
        b = self.b_l + b_h

        # solve with an iterative scheme the problem
        x_l, _ = sps.linalg.gmres(
            A, b, tol=tol, maxiter=maxiter, M=M, callback=callback
        )

        # reconstruct the higher dimensional solution
        if return_lower:
            if info:
                return x_l, {"solve_h": iteration_number + 1}
            else:
                return x_l
        else:
            x_h += self.solve_h(x_l)
            x = self.concatenate(x_h, x_l)
            if info:
                return x, {"solve_h": iteration_number + 2}
            else:
                return x

    # ------------------------------------------------------------------------------#

    def solve_jacobian(self, b, tol, maxiter, info=False):

        global iteration_number
        iteration_number = 0

        def callback(x):
            global iteration_number
            iteration_number += 1

        # construct the matrix and preconditioner
        J = sps.linalg.LinearOperator(self.A_l.shape, self.schur_complement)
        P = sps.linalg.splu(self.A_l.tocsc())
        M = sps.linalg.LinearOperator(self.A_l.shape, lambda x: P.solve(x))

        # solve with an iterative scheme the problem
        x_l, _ = sps.linalg.gmres(
            J, b, tol=tol, maxiter=maxiter, M=M, callback=callback
        )

        # reconstruct the higher dimensional solution:
        if info:
            return x_l, {"solve_h": iteration_number}
        else:
            return x_l

    # -----------------------------------------------------------------------------#

    def factorize(self):
        self.LU = sps.linalg.factorized(self.A_h.tocsc())

    # -----------------------------------------------------------------------------#

    def schur_complement(self, x_l):
        x_h = np.zeros(x_l.size)
        x_h[: self.dof_l] = self.C_l * self.solve_h(x_l)[self.dof_h :]
        return self.A_l * x_l + x_h

    # ------------------------------------------------------------------------------#

    def solve_h(self, x_l):
        # compute the higher dimensional solution
        b = np.r_[[0] * self.dof_h, -self.C_h * x_l[: self.dof_l]]
        return self.LU(b)

    # ------------------------------------------------------------------------------#

    def concatenate(self, x_h, x_l):
        # save and export using standard algorithm
        x = np.zeros(self.b_h.size + x_l.size)
        if x_h is not None:
            x[: self.dof_h] = x_h[: self.dof_h]
        if x_l is not None:
            x[self.dof_h : (self.dof_h + self.dof_l)] = x_l[: self.dof_l]
        return x

    # ------------------------------------------------------------------------------#

    def steklov_poincare(self, x_l):
        Sf_l = np.zeros(x_l.size)

        # evaluate SP-operator on right-hand side
        b = np.r_[[0] * self.dof_h, -self.C_h * x_l[: self.dof_l]]
        Sf_l[: self.dof_l] = self.C_l * self.LU(b + self.b_h)[self.dof_h :]

        return Sf_l

    # ------------------------------------------------------------------------------#

    def residual_l(self, x_l):
        # evaluate lower-dimensional part
        f_l = self.b_l - self.A_l * x_l

        # evaluate Steklov-Poincare operator part
        Sf_l = self.steklov_poincare(x_l)

        return f_l - Sf_l


# -----------------------------------------------------------------------------#
