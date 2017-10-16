from __future__ import division
import numpy as np
import scipy.sparse as sps

from porepy.numerics.mixed_dim.solver import Solver


class Upwind(Solver):
    """
    Discretize a hyperbolic transport equation using a single point upstream
    weighting scheme.


    """
#------------------------------------------------------------------------------#

    def __init__(self, physics='transport'):
        self.physics = physics

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
        Note: if not specified the inflow boundary conditions are no-flow, while
        the outflow boundary conditions are open.

        The name of data in the input dictionary (data) are:
        discharge : array (g.num_faces)
            Normal velocity at each face, weighted by the face area.
        bc : boundary conditions (optional)
        bc_val : dictionary (optional)
            Values of the boundary conditions. The dictionary has at most the
            following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
            conditions, respectively.
        source : array (g.num_cells) of source (positive) or sink (negative) terms.
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
        data = {'discharge': u, 'bc': bnd, 'bc_val': bnd_val}
        advect = upwind.Upwind()
        U, rhs = advect.matrix_rhs(g, data)

        data = {'deltaT': advect.cfl(g, data)}
        M, _ = mass.MassMatrix().matrix_rhs(g, data)

        M_minus_U = M - U
        invM = mass.MassMatrix().inv(M)

        # Loop over the time
        for i in np.arange( N ):
            conc = invM.dot((M_minus_U).dot(conc) + rhs)

        """
        if g.dim == 0:
            return sps.csr_matrix([0]), [0]

        param = data['param']
        discharge = param.get_discharge()
        bc = param.get_bc(self)
        bc_val = param.get_bc_val(self)

        has_bc = not(bc is None or bc_val is None)

        # Compute the face flux respect to the real direction of the normals
        indices = g.cell_faces.indices
        flow_faces = g.cell_faces.copy()
        flow_faces.data *= discharge[indices]

        # Retrieve the faces boundary and their numeration in the flow_faces
        # We need to impose no-flow for the inflow faces without boundary
        # condition
        mask = np.unique(indices, return_index=True)[1]
        bc_neu = g.get_boundary_faces()

        if has_bc:
            # If boundary conditions are imposed remove the faces from this
            # procedure.
            bc_dir = np.where(bc.is_dir)[0]
            bc_neu = np.setdiff1d(bc_neu, bc_dir, assume_unique=True)
            bc_dir = mask[bc_dir]

            # Remove Dirichlet inflow
            inflow = flow_faces.copy()

            inflow.data[bc_dir] = inflow.data[bc_dir].clip(max=0)
            flow_faces.data[bc_dir] = flow_faces.data[bc_dir].clip(min=0)

        # Remove all Neumann
        bc_neu = mask[bc_neu]
        flow_faces.data[bc_neu] = 0

        # Determine the outflow faces
        if_faces = flow_faces.copy()
        if_faces.data = np.sign(if_faces.data)

        # Compute the inflow/outflow related to the cells of the problem
        flow_faces.data = flow_faces.data.clip(min=0)
        flow_cells = if_faces.transpose() * flow_faces
        flow_cells.tocsr()

        if not has_bc:
            return flow_cells, np.zeros(g.num_cells)

        # Impose the boundary conditions
        bc_val_dir = np.zeros(g.num_faces)
        if np.any(bc.is_dir):
            is_dir = np.where(bc.is_dir)[0]
            bc_val_dir[is_dir] = bc_val[is_dir]

        # We assume that for Neumann boundary condition a positive 'bc_val'
        # represents an outflow for the domain. A negative 'bc_val' represents
        # an inflow for the domain.
        bc_val_neu = np.zeros(g.num_faces)
        if np.any(bc.is_neu):
            is_neu = np.where(bc.is_neu)[0]
            bc_val_neu[is_neu] = bc_val[is_neu]

        return flow_cells, - inflow.transpose() * bc_val_dir \
            - np.abs(g.cell_faces.transpose()) * bc_val_neu

#------------------------------------------------------------------------------#

    def cfl(self, g, data):
        """
        Return the time step according to the CFL condition.
        Note: the vector field is assumed to be given as the normal velocity,
        weighted with the face area, at each face.

        The name of data in the input dictionary (data) are:
        discharge : array (g.num_faces)
            Normal velocity at each face, weighted by the face area.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

        Return
        ------
        deltaT: time step according to CFL condition.

        """
        if g.dim == 0:
            return np.inf
        # Retrieve the data, only "discharge" is mandatory
        param = data['param']
        discharge = param.get_discharge()
        aperture = param.get_aperture()
        phi = param.get_porosity()

        faces, cells, _ = sps.find(g.cell_faces)

        # Detect and remove the faces which have zero in discharge
        not_zero = ~np.isclose(np.zeros(faces.size), discharge[faces], atol=0)
        if not np.any(not_zero):
            return np.inf

        cells = cells[not_zero]
        faces = faces[not_zero]

        # Compute discrete distance cell to face centers
        dist_vector = g.face_centers[:, faces] - g.cell_centers[:, cells]
        # Element-wise scalar products between the distance vectors and the
        # normals
        dist = np.einsum('ij,ij->j', dist_vector, g.face_normals[:, faces])
        # Since discharge is multiplied by the aperture, we get rid of it!!!!
        # Additionally we consider the phi (porosity) and the cell-mapping
        coeff = (aperture * phi)[cells]
        # deltaT is deltaX/discharge with coefficient
        return np.amin(np.abs(np.divide(dist, discharge[faces])) * coeff)

#------------------------------------------------------------------------------#

    def discharge(self, g, beta, cell_apertures=None):
        """
        Return the normal component of the velocity, for each face, weighted by
        the face area and aperture.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        beta: (3x1) array which represents the constant velocity.
        cell_apertures: (g.num_faces) array of apertures

        Return
        ------
        discharge : array (g.num_faces)
            Normal velocity at each face, weighted by the face area.

        """
        if cell_apertures is None:
            face_apertures = np.ones(g.num_faces)
        else:
            face_apertures = abs(g.cell_faces) * cell_apertures
            r, _, _ = sps.find(g.cell_faces)
            face_apertures = face_apertures / np.bincount(r)

        beta = np.asarray(beta)
        assert beta.size == 3

        if g.dim == 0:
            return np.atleast_1d(np.dot(g.face_normals.ravel('F'), face_apertures * beta))

        return np.array([np.dot(n, a * beta)
                         for n, a in zip(g.face_normals.T, face_apertures)])

#------------------------------------------------------------------------------#
