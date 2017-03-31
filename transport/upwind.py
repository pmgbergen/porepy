import numpy as np
import scipy.sparse as sps

from core.solver.solver import *
from core.grids import grid

class Upwind(Solver):
    """
    Discretize a hyperbolic transport equation using a single point upstream
    weighting scheme.


    """

#------------------------------------------------------------------------------#

    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of cells (concentration dof).

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_cells

#------------------------------------------------------------------------------#

    def matrix_rhs(self, g, data):
        """
        Return the matrix and righ-hand side for a discretization of a scalar
        linear transport problem using the upwind scheme.
        Note: the vector field is assumed to be given as the normal velocity,
        weighted with the face area, at each face.

        The name of data in the input dictionary (data) are:
        beta_n : array (g.num_faces)
            Normal velocity at each face, weighted by the face area.
        bc : boundary conditions (optional)
        bc_val : dictionary (optional)
            Values of the boundary conditions. The dictionary has at most the
            following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
            conditions, respectively.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

        Return
        ------
        matrix: sparse csr (g.num_cells, g_num_cells)
            Upwind matrix obtained from the discretization.
        rhs: array (g_num_cells)
            Right-hand side which contains the boundary conditions.

        Examples
        --------
        data = {'beta_n': u, 'bc': bnd, 'bc_val': bnd_val}
        advect = upwind.Upwind()
        U, rhs = advect.matrix_rhs(g, data)

        data = {'deltaT': advect.cfl(g, data)}
        M, _ = mass.Mass().matrix_rhs(g, data)

        M_minus_U = M - U
        invM = mass.Mass().inv(M)

        # Loop over the time
        for i in np.arange( N ):
            conc = invM.dot((M_minus_U).dot(conc) + rhs)

        """
        beta_n, bc, bc_val = data['beta_n'], data.get('bc'), data.get('bc_val')
        indices = g.cell_faces.indices

        flow_faces = g.cell_faces.copy()
        flow_faces.data *= beta_n[indices]

        if_inflow_faces = flow_faces.copy()
        if_inflow_faces.data = np.sign(if_inflow_faces.data.clip(max=0))

        flow_cells = if_inflow_faces.transpose() * flow_faces
        flow_cells.tocsr()

        if bc is None or bc_val is None:
            return flow_cells, np.zeros(g.num_cells)

        # Dirichlet boundary condition
        flow_faces.data = np.multiply(flow_faces.data,
                                      bc.is_dir[indices]).clip(max=0)
        flow_faces.eliminate_zeros()

        bc_val_dir = np.zeros(g.num_faces)
        bc_val_dir[bc.is_dir] = bc_val['dir']

        flow_faces.data = -flow_faces.data * bc_val_dir[flow_faces.indices]

        return flow_cells, np.squeeze(np.asarray(flow_faces.sum(axis=0)))

#------------------------------------------------------------------------------#

    def cfl(self, g, data):
        """
        Return the time step according to the CFL (Courant–Friedrichs–Lewy)
        condition.
        Note: the vector field is assumed to be given as the normal velocity,
        weighted with the face area, at each face.

        The name of data in the input dictionary (data) are:
        beta_n : array (g.num_faces)
            Normal velocity at each face, weighted by the face area.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

        Return
        ------
        deltaT: time step according to CFL condition.

        """
        beta_n = data['beta_n']
        faces, cell, _ = sps.find(g.cell_faces)
        not_zero = ~np.isclose(np.zeros(faces.size), beta_n[faces], atol=0)
        if not np.any(not_zero): return np.inf

        beta_n = np.abs(beta_n[faces[not_zero]])
        volumes = g.cell_volumes[cell[not_zero]]

        return np.amin(np.divide(volumes, beta_n))/g.dim

#------------------------------------------------------------------------------#

    def beta_n(self, g, beta):
        """
        Return the normal component of the velocity, for each face, weighted by
        the face area.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        beta: (3x1) array which represents the constant velocity.

        Return
        ------
        beta_n : array (g.num_faces)
            Normal velocity at each face, weighted by the face area.

        """
        beta = np.asarray(beta)
        assert beta.size == 3
        return np.array([np.dot(n, beta) for n in g.face_normals.T])

#------------------------------------------------------------------------------#
