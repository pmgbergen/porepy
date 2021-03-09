from typing import Any, Dict, Tuple

import numpy as np
import scipy.sparse as sps

import porepy as pp

module_sections = ["numerics", "disrcetization"]


class Upwind(pp.numerics.discretization.Discretization):
    """
    Discretize a hyperbolic transport equation using a single point upstream
    weighting scheme.


    """

    def __init__(self, keyword: str = "transport") -> None:
        self.keyword = keyword

        # Keywords used to store matrix and right hand side in the matrix_dictionary
        self.upwind_matrix_key = "transport"
        self.rhs_matrix_key = "rhs"

    def ndof(self, g: pp.Grid) -> int:
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

    @pp.time_logger(sections=module_sections)
    def assemble_matrix_rhs(
        self, g: pp.Grid, data: Dict
    ) -> Tuple[sps.spmatrix, np.ndarray]:
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

    @pp.time_logger(sections=module_sections)
    def assemble_matrix(self, g: pp.Grid, data: Dict) -> sps.spmatrix:
        """Return the matrix for an upwind discretization of a linear transport
        problem.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization.
                Size: g.num_cells x g.num_cells.

        """
        matrix_dictionary: Dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
            self.keyword
        ]
        upwind = matrix_dictionary[self.upwind_matrix_key]
        div: sps.spmatrix = pp.fvutils.scalar_divergence(g)
        return div * upwind

    # ------------------------------------------------------------------------------#

    @pp.time_logger(sections=module_sections)
    def assemble_rhs(self, g: pp.Grid, data: Dict) -> np.ndarray:
        """Return the right-hand side for an upwind discretization of a linear
        transport problem.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            np.ndarray: Right hand side vector with representation of boundary
                conditions. The size of the vector will depend on the discretization.

        """
        parameter_dictionary: Dict[str, Any] = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary: Dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
            self.keyword
        ]

        bc_values: np.ndarray = parameter_dictionary["bc_values"]
        bc_discr = matrix_dictionary[self.rhs_matrix_key]

        div = sps.spmatrix = pp.fvutils.scalar_divergence(g)

        return div * bc_discr * bc_values

    @pp.time_logger(sections=module_sections)
    def discretize(self, g: pp.Grid, data: Dict, d_name: str = "darcy_flux") -> None:
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

        parameter_dictionary: Dict[str, Any] = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary: Dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
            self.keyword
        ]

        # Shortcut for point grids
        if g.dim == 0:
            matrix_dictionary[self.upwind_matrix_key] = sps.csr_matrix((0, 1))
            matrix_dictionary[self.rhs_matrix_key] = sps.csr_matrix((0, 0))
            return

        darcy_flux: np.ndarray = parameter_dictionary[d_name]
        bc: pp.BoundaryCondition = parameter_dictionary["bc"]

        # Discretization is made of two parts: One is a mapping from the upstream cell
        # to the face. Second is a scaling of that upstream function with the Darcy
        # flux. Then boundary conditions of course complicate everything.

        # booleans of flux direction
        pos_flux = darcy_flux >= 0
        neg_flux = np.logical_not(pos_flux)

        # Array to store index of the cell in the usptream direction
        upstream_cell_ind = np.zeros(g.num_faces, dtype=int)
        # Fill the array based on the cell-face relation. By construction, the normal
        # vector of a face points from the first to the second row in this array
        cf_dense = g.cell_face_as_dense()
        # Positive fluxes point in the same direction as the normal vector, find the
        # upstream cell
        upstream_cell_ind[pos_flux] = cf_dense[0, pos_flux]
        upstream_cell_ind[neg_flux] = cf_dense[1, neg_flux]

        # Make row and data arrays, preparing to make an coo-matrix for the upstream
        # cell-to-face map.
        row = np.arange(g.num_faces)
        values = np.ones(g.num_faces, dtype=int)

        # We need to eliminate faces on the boundary; these will be discretized
        # separately below. On faces with Neumann conditions, boundary conditions apply
        # for both inflow and outflow. For Dirichlet, only inflow conditions are given;
        # for outflow, we use upstream weighting (thus no need to modify the matrix
        # we are about to build).

        # faces with Neumann conditions
        neumann_ind = np.where(bc.is_neu)[0]

        # Faces with Dirichlet conditions and inflow. The latter is identified by
        # considering the direction of the flux, and the upstream element in cf_dense
        # (note that the exterior of the domain is represented by -1 in cf_dense).
        inflow_ind = np.where(
            np.logical_and(
                bc.is_dir,
                np.logical_or(
                    np.logical_and(pos_flux, cf_dense[0] < 0),
                    np.logical_and(neg_flux, cf_dense[1] < 0),
                ),
            )
        )[0]

        # Delete indices that should be treated by boundary conditions
        delete_ind = np.sort(np.r_[neumann_ind, inflow_ind])
        row = np.delete(row, delete_ind)
        values = np.delete(values, delete_ind)
        col = np.delete(upstream_cell_ind, delete_ind)

        # Finally we can construct the upstream weighting matrix.
        upstream_mat = sps.coo_matrix(
            (
                values,
                (row, col),
            ),
            shape=(g.num_faces, g.num_cells),
        ).tocsr()

        # Scaling with Darcy fluxes is a diagonal matrix
        flux_mat = sps.dia_matrix((darcy_flux, 0), shape=(g.num_faces, g.num_faces))

        # Form and store disrcetization matrix
        matrix_dictionary[self.upwind_matrix_key] = flux_mat * upstream_mat

        ## Boundary conditions
        # On Neumann boundaries the precribed boundary value should effectively
        # be added to the adjacent cell, with the convention that influx (so
        # negative boundary value) should correspond to accumulation
        # On Dirichlet boundaries, we consider only inflow boundaries.

        # For Neumann faces we need to assign the sign of the divergence, to
        # counteract multiplication with the same sign when the divergence is
        # applied (e.g. in self.assemble_matrix).
        sgn_div = pp.fvutils.scalar_divergence(g).sum(axis=0).A.squeeze()

        row = np.hstack([neumann_ind, inflow_ind])
        col = row
        # Need minus signs on both Neumann and Dirichlet data to ensure that accumulation
        # follows from negative fluxes.
        values_bc = np.hstack([-sgn_div[neumann_ind], -darcy_flux[inflow_ind]])
        bc_discr = sps.coo_matrix(
            (values_bc, (row, col)), shape=(g.num_faces, g.num_faces)
        ).tocsr()

        matrix_dictionary[self.rhs_matrix_key] = bc_discr

    @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
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
