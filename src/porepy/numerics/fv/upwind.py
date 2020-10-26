import numpy as np
import scipy.sparse as sps

import porepy as pp


class Upwind(pp.numerics.discretization.Discretization):
    """
    Discretize a hyperbolic transport equation using a single point upstream
    weighting scheme.


    """

    def __init__(self, keyword="transport"):
        self.keyword = keyword

        # Keywords used to store matrix and right hand side in the matrix_dictionary
        self.matrix_keyword = "transport"
        self.rhs_keyword = "rhs"

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

    def assemble_matrix_rhs(self, g, data):
        """Return the matrix for an upwind discretization of a linear transport
        problem.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization.
                Size: g.num_cells x g.num_cells.
            np.ndarray: Right hand side vector with representation of boundary
                conditions. The size of the vector will depend on the discretization.

        """
        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

    def assemble_matrix(self, g, data):
        """Return the matrix for an upwind discretization of a linear transport
        problem.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization.
                Size: g.num_cells x g.num_cells.

        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        return matrix_dictionary[self.matrix_keyword]

    # ------------------------------------------------------------------------------#

    def assemble_rhs(self, g, data):
        """Return the right-hand side for an upwind discretization of a linear
        transport problem.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            np.ndarray: Right hand side vector with representation of boundary
                conditions. The size of the vector will depend on the discretization.

        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        return matrix_dictionary[self.rhs_keyword]

    def discretize(self, g, data, d_name="darcy_flux"):
        """
        Return the matrix and righ-hand side for a discretization of a scalar
        linear transport problem using the upwind scheme.
        Note: the vector field is assumed to be given as the normal velocity,
        weighted with the face area, at each face.
        Note: if not specified the inflow boundary conditions are no-flow, while
        the outflow boundary conditions are open.

        The name of data in the input dictionary (data) are:
        darcy_flux : array (g.num_faces)
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
        d_name: (string) keyword for data field in data containing the dischages

        Return
        ------
        matrix: sparse csr (g.num_cells, g_num_cells)
            Upwind matrix obtained from the discretization.
        rhs: array (g_num_cells)
            Right-hand side which contains the boundary conditions.

        Examples
        --------
        data = {'darcy_flux': u, 'bc': bnd, 'bc_val': bnd_val}
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

        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        # Shortcut for point grids
        if g.dim == 0:
            matrix_dictionary[self.matrix_keyword] = sps.csr_matrix([0.0])
            matrix_dictionary[self.rhs_keyword] = np.array([0.0])
            return

        darcy_flux = parameter_dictionary[d_name]
        bc = parameter_dictionary["bc"]
        bc_val = parameter_dictionary["bc_values"]

        has_bc = not (bc is None or bc_val is None)

        # Compute the face flux respect to the real direction of the normals
        indices = g.cell_faces.indices
        flow_faces = g.cell_faces.copy()
        flow_faces.data *= darcy_flux[indices]

        # Retrieve the faces boundary and their numeration in the flow_faces
        # We need to impose no-flow for the inflow faces without boundary
        # condition
        mask = np.unique(indices, return_index=True)[1]
        bc_neu = g.get_all_boundary_faces()

        if has_bc:
            # If boundary conditions are imposed remove the faces from this
            # procedure.
            # For primal-like discretizations, internal boundaries
            # are handled by assigning Neumann conditions.
            is_dir = np.logical_and(bc.is_dir, np.logical_not(bc.is_internal))
            bc_dir = np.where(is_dir)[0]
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
        flow_cells = flow_cells.astype(np.float)

        # Store disrcetization matrix
        matrix_dictionary[self.matrix_keyword] = flow_cells

        if not has_bc:
            # Short cut if there are no trivial boundary conditions
            matrix_dictionary[self.rhs_keyword] = np.zeros(g.num_cells)
        else:
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

            matrix_dictionary[self.rhs_keyword] = (
                -inflow.transpose() * bc_val_dir
                - np.abs(g.cell_faces.transpose()) * bc_val_neu
            )

    def cfl(self, g, data, d_name="darcy_flux"):
        """
        Return the time step according to the CFL condition.
        Note: the vector field is assumed to be given as the normal velocity,
        weighted with the face area, at each face.

        The name of data in the input dictionary (data) are:
        darcy_flux : array (g.num_faces)
            Normal velocity at each face, weighted by the face area.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.
        d_name: (string) keyword for dischagre file in data dictionary

        Return
        ------
        deltaT: time step according to CFL condition.

        """
        if g.dim == 0:
            return np.inf
        # Retrieve the data
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        darcy_flux = parameter_dictionary[d_name]
        phi = parameter_dictionary["mass_weight"]

        faces, cells, _ = sps.find(g.cell_faces)

        # Detect and remove the faces which have zero in darcy_flux
        not_zero = ~np.isclose(np.zeros(faces.size), darcy_flux[faces], atol=0)
        if not np.any(not_zero):
            return np.inf

        cells = cells[not_zero]
        faces = faces[not_zero]

        # Compute discrete distance cell to face centers
        dist_vector = g.face_centers[:, faces] - g.cell_centers[:, cells]
        # Element-wise scalar products between the distance vectors and the
        # normals
        dist = np.einsum("ij,ij->j", dist_vector, g.face_normals[:, faces])
        # Additionally we consider the phi (porosity) and the cell-mapping
        coeff = phi[cells]
        # deltaT is deltaX/darcy_flux with coefficient
        return np.amin(np.abs(np.divide(dist, darcy_flux[faces])) * coeff)

    def darcy_flux(self, g, beta, cell_apertures=None):
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
        darcy_flux : array (g.num_faces)
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
            if g.num_faces == 0:
                dot_prod = np.zeros(0)
            else:
                dot_prod = np.dot(g.face_normals.ravel("F"), face_apertures * beta)
            return np.atleast_1d(dot_prod)

        return np.array(
            [np.dot(n, a * beta) for n, a in zip(g.face_normals.T, face_apertures)]
        )

    def outflow(self, g, data, d_name="darcy_flux"):
        if g.dim == 0:
            return sps.csr_matrix([0])

        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        darcy_flux = parameter_dictionary[d_name]
        bc = parameter_dictionary["bc"]
        bc_val = parameter_dictionary["bc_values"]

        has_bc = not (bc is None or bc_val is None)

        # Compute the face flux respect to the real direction of the normals
        indices = g.cell_faces.indices
        flow_faces = g.cell_faces.copy()
        flow_faces.data *= darcy_flux[indices]

        # Retrieve the faces boundary and their numeration in the flow_faces
        # We need to impose no-flow for the inflow faces without boundary
        # condition
        mask = np.unique(indices, return_index=True)[1]
        bc_neu = g.tags["domain_boundary_faces"].nonzero()[0]

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

        outflow_faces = if_faces.indices[if_faces.data > 0]
        domain_boundary_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        outflow_faces = np.intersect1d(
            outflow_faces, domain_boundary_faces, assume_unique=True
        )

        # va tutto bene se ho neumann omogeneo
        # gli outflow sono positivi

        if_outflow_faces = if_faces.copy()
        if_outflow_faces.data[:] = 0
        if_outflow_faces.data[np.in1d(if_faces.indices, outflow_faces)] = 1

        if_outflow_cells = if_outflow_faces.transpose() * flow_faces
        if_outflow_cells.tocsr()

        return if_outflow_cells
