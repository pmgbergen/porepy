import numpy as np
import scipy.sparse as sps

import porepy as pp

from porepy.numerics.mixed_dim.solver import Solver, SolverMixedDim
from porepy.numerics.mixed_dim.coupler import Coupler

# ------------------------------------------------------------------------------#


class P1MassMatrixMixedDim(SolverMixedDim):
    def __init__(self, physics="flow"):
        self.physics = physics

        self.discr = P1MassMatrix(self.physics)

        self.solver = Coupler(self.discr)


# ------------------------------------------------------------------------------#


class P1MassMatrix(Solver):

    # ------------------------------------------------------------------------------#

    def __init__(self, physics="flow"):
        self.physics = physics

    # ------------------------------------------------------------------------------#

    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of cells.

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_nodes

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self, g, data):
        """
        Return the matrix and righ-hand side for a discretization of a mass
        bilinear form using P1 method on simplices.

        phi: porosity (optional)
        aperture: aperture (optional)
        deltaT: time step (optional)
        bc : boundary conditions (optional)

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: where the data are stored.

        Return
        ------
        matrix: sparse csr (g.num_nodes, g.num_nodes)
            Matrix obtained from the discretization.
        rhs: array (g.num_nodes)
            Right-hand side which contains the boundary conditions and the scalar
            source term.

        """
        return self.matrix(g, data), self.rhs(g, data, bc_weight)

    # ------------------------------------------------------------------------------#

    def matrix(self, g, data):
        """
        Return the matrix and righ-hand side (null) for a discretization of a
        L2-mass bilinear form with P1 test and trial functions.

        The name of data in the input dictionary (data) are:
        phi: array (self.g.num_cells)
            Scalar values which represent the porosity.
            If not given assumed unitary.
        deltaT: Time step for a possible temporal discretization scheme.
            If not given assumed unitary.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

        Return
        ------
        matrix: sparse dia (g.num_cells, g_num_cells)
            Mass matrix obtained from the discretization.
        rhs: array (g_num_cells)
            Null right-hand side.

        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        param = data["param"]
        phi = param.get_porosity()
        aperture = param.get_aperture()
        coeff = phi / data.get("deltaT", 1.) * aperture

        # TODO the coeff is not used since the default values are cell based.
        bc = param.get_bc(self)
        assert isinstance(bc, pp.BoundaryConditionNode)

        # If a 0-d grid is given then we return an identity matrix
        if g.dim == 0:
            return coeff * sps.csr_matrix((self.ndof(g), self.ndof(g)))

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        c_centers, f_normals, f_centers, R, dim, node_coords = pp.cg.map_grid(g)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(g.dim + 1, 2) * g.num_cells
        I = np.empty(size, dtype=np.int)
        J = np.empty(size, dtype=np.int)
        dataIJ = np.empty(size)
        idx = 0

        cell_nodes = g.cell_nodes()
        nodes, cells, _ = sps.find(cell_nodes)

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
            nodes_loc = nodes[loc]

            # Compute the mass-H1 local matrix
            # A = coeff[nodes_loc]*self.massH1(g.cell_volumes[c], g.dim)
            A = self.massH1(g.cell_volumes[c], g.dim)

            # Save values for stiff-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            I[loc_idx] = cols.T.ravel()
            J[loc_idx] = cols.ravel()
            dataIJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        M = sps.csr_matrix((dataIJ, (I, J)))

        # assign the Dirichlet boundary conditions
        if bc and np.any(bc.is_dir):
            dir_nodes = np.where(bc.is_dir)[0]
            # set in an efficient way the essential boundary conditions, by
            # clear the rows and put norm in the diagonal
            for row in dir_nodes:
                M.data[M.indptr[row] : M.indptr[row + 1]] = 0.

        return M

    # ------------------------------------------------------------------------------#

    def rhs(self, g, data):
        """
        Return the righ-hand side for a discretization of a mass bilinear form
        using P1 method. See self.matrix_rhs for a detaild description.

        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        return np.zeros(self.ndof())

    # ------------------------------------------------------------------------------#

    def massH1(self, c_volume, dim):
        """ Compute the local mass H1 matrix using the P1 Lagrangean approach.

        Parameters
        ----------
        c_volume : scalar
            Cell volume.

        Return
        ------
        out: ndarray (num_faces_of_cell, num_faces_of_cell)
            Local mass Hdiv matrix.
        """
        # Allow short variable names in this function
        # pylint: disable=invalid-name

        M = np.ones((dim + 1, dim + 1)) + np.identity(dim + 1)
        return c_volume * M / ((dim + 1) * (dim + 2))


# ------------------------------------------------------------------------------#
