"""
Module contains superclass for mpfa and tpfa.
"""
import porepy as pp


class FVElliptic(pp.numerics.interface_laws.elliptic_discretization.EllipticDiscretization):
    """ Superclass for finite volume discretizations of the elliptic equation.

    Should not be used by itself, instead use a subclass that implements an
    actual discretization method. Known subclasses are Tpfa and Mpfa.

    """

    def __init__(self, keyword):

        # Identify which parameters to use:
        self.keyword = keyword

    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of cells (pressure dof).

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_cells

    def extract_pressure(self, g, solution_array, d):
        """ Extract the pressure part of a solution.
        The method is trivial for finite volume methods, with the pressure
        being the only primary variable.

        Parameters:
            g (grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid obtained from
                either a mono-dimensional or a mixed-dimensional problem.
            d (dictionary): Data dictionary associated with the grid. Not used,
                but included for consistency reasons.
        Returns:
            np.array (g.num_cells): Pressure solution vector. Will be identical
                to solution_array.
        """
        return solution_array

    def extract_flux(self, g, solution_array, d):
        """ Extract the flux related to a solution.

        The flux is computed from the discretization and the given pressure solution.

        @ALL: We should incrude the boundary condition as well?

        Parameters:
            g (grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid obtained from
                either a mono-dimensional or a mixed-dimensional problem. Will
                correspond to the pressure solution.
            d (dictionary): Data dictionary associated with the grid.

        Returns:
            np.array (g.num_faces): Flux vector.

        """
        flux_discretization = d[self._key() + "flux"]
        return flux_discretization * solution_array

    # ------------------------------------------------------------------------------#

    def assemble_matrix_rhs(self, g, data):
        """ Return the matrix and right-hand side for a discretization of a second
        order elliptic equation.

        Also discretize the necessary operators if the data dictionary does not
        contain a transmissibility matrix. In that case, we assume the following two
        sub-dictionaries to be present in the data dictionary:
            parameter_dictionary, storing all parameters.
                Stored in data[pp.PARAMETERS][self.keyword].
            matrix_dictionary, for storage of discretization matrices.
                Stored in data[pp.DISCRETIZATION_MATRICES][self.keyword]

        parameter_dictionary contains the entries:
            second_order_tensor: (pp.SecondOrderTensor) Permeability defined cell-wise.
            bc: (pp.BoundaryCondition) boundary conditions.
            bc_values: array (self.num_faces) The boundary condition values.
            aperture: (np.ndarray) apertures of the cells for scaling of the face
                normals.
            Optional parameters: See the discretize methods.

        After discretization, matrix_dictionary will be updated with the following
        entries:
            flux: sps.csc_matrix (g.num_faces, g.num_cells)
                Flux discretization, cell center contribution.
            bound_flux: sps.csc_matrix (g.num_faces, g.num_faces)
                Flux discretization, face contribution.
            bound_pressure_cell: sps.csc_matrix (g.num_faces, g.num_cells)
                Operator for reconstructing the pressure trace, cell center
                contribution.
            bound_pressure_face: sps.csc_matrix (g.num_faces, g.num_faces)
                Operator for reconstructing the pressure trace, face contribution.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization. The size of
                the matrix will depend on the specific discretization.
            np.ndarray: Right hand side vector with representation of boundary
                conditions. The size of the vector will depend on the discretization.
        """

        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

    def assemble_matrix(self, g, data):
        """ Return the matrix for a discretization of a second order elliptic equation
        using a FV method.

        Also discretize the necessary operators if the data dictionary does not contain
        a discretization of the boundary equation. For the required fields of the data
        dictionary, see the assemble_matrix_rhs and discretize methods.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization. The
                size of the matrix will depend on the specific discretization.
        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        if not "flux" in matrix_dictionary:
            self.discretize(g, data)

        div = pp.fvutils.scalar_divergence(g)
        flux = matrix_dictionary["flux"]
        M = div * flux

        return M

    # ------------------------------------------------------------------------------#

    def assemble_rhs(self, g, data):
        """ Return the right-hand side for a discretization of a second order elliptic
        equation using a finite volume method.

        Also discretize the necessary operators if the data dictionary does not contain
        a discretization of the boundary equation. For the required fields of the data
        dictionary, see the assemble_matrix_rhs and discretize methods.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            np.ndarray: Right hand side vector with representation of boundary
                conditions. The size of the vector will depend on the discretization.
        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        if not "bound_flux" in matrix_dictionary:
            self.discretize(g, data)

        bound_flux = matrix_dictionary["bound_flux"]

        parameter_dictionary = data[pp.PARAMETERS][self.keyword]

        bc_val = parameter_dictionary["bc_values"]

        div = g.cell_faces.T

        return -div * bound_flux * bc_val

    def assemble_int_bound_flux(
        self, g, data, data_edge, grid_swap, cc, matrix, self_ind
    ):
        """Assemble the contribution from an internal boundary, manifested as a
        flux boundary condition.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a higher-dimensional domain.

        Implementations of this method will use an interplay between the grid
        on the node and the mortar grid on the relevant edge.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_edge (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            grid_swap (boolean): If True, the grid g is identified with the @
                slave side of the mortar grid in data_adge.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                master and slave side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.

        """
        div = g.cell_faces.T

        bound_flux = data[pp.DISCRETIZATION_MATRICES][self.keyword]["bound_flux"]
        # Projection operators to grid
        mg = data_edge["mortar_grid"]

        if grid_swap:
            proj = mg.slave_to_mortar_avg()
        else:
            proj = mg.master_to_mortar_avg()

        cc[self_ind, 2] += div * bound_flux * proj.T

    def assemble_int_bound_source(
        self, g, data, data_edge, grid_swap, cc, matrix, self_ind
    ):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a source term.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a lower-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_edge (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            grid_swap (boolean): If True, the grid g is identified with the @
                slave side of the mortar grid in data_adge.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                master and slave side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.

        """
        mg = data_edge["mortar_grid"]

        if grid_swap:
            proj = mg.master_to_mortar_avg()
        else:
            proj = mg.slave_to_mortar_avg()

        cc[self_ind, 2] -= proj.T

    def assemble_int_bound_pressure_trace(
        self, g, data, data_edge, grid_swap, cc, matrix, self_ind
    ):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a condition on the boundary pressure.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a higher-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_edge (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            grid_swap (boolean): If True, the grid g is identified with the @
                slave side of the mortar grid in data_adge.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                master and slave side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.

        """
        mg = data_edge["mortar_grid"]

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        # TODO: this should become first or second or something
        if grid_swap:
            proj = mg.slave_to_mortar_avg()
        else:
            proj = mg.master_to_mortar_avg()

        cc[2, self_ind] += proj * matrix_dictionary["bound_pressure_cell"]
        cc[2, 2] += proj * matrix_dictionary["bound_pressure_face"] * proj.T

    def assemble_int_bound_pressure_cell(
        self, g, data, data_edge, grid_swap, cc, matrix, self_ind
    ):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a condition on the cell pressure.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a lower-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_edge (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            grid_swap (boolean): If True, the grid g is identified with the @
                slave side of the mortar grid in data_adge.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                master and slave side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.
        """
        mg = data_edge["mortar_grid"]

        if grid_swap:
            proj = mg.master_to_mortar_avg()
        else:
            proj = mg.slave_to_mortar_avg()

        cc[2, self_ind] -= proj

    def enforce_neumann_int_bound(
        self, g_master, data_edge, matrix, swap_grid, self_ind
    ):
        """ Enforce Neumann boundary conditions on a given system matrix.

        The method is void for finite volume approaches, but is implemented
        to be compatible with the general framework.

        Parameters:
            g (Grid): On which the equation is discretized
            data (dictionary): Of data related to the discretization.
            matrix (scipy.sparse.matrix): Discretization matrix to be modified.
        """
        # Operation is void for finite volume methods
        pass
