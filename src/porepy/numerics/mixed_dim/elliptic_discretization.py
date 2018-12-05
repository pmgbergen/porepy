"""
The module contains the base class for all elliptic discretization methods,
to be used in both mono and mixed-dimensional problems.

"""
from porepy.numerics.mixed_dim.solver import Solver


class EllipticDiscretization(Solver):
    """ This is the parent class of all discretizations for second order elliptic
    problems. The class cannot be used itself, but should rather be seen as a
    declaration of which methods are assumed implemented for all specific
    discretization schemes.

    Subclasses are intended used both on single grids, and as components in a
    mixed-dimensional, or more generally multiple grid problem. In the latter case, this class will provide the
    discretization on individual grids; a full discretization will also need
    discretizations of the edge problems (coupling between grids) and a way to
    assemble the grids.

    The class methods should together take care of both the discretization
    (defined as construction of operators used when writing down the continuous
    form of the equations, e.g. divergence, transmissibility matrix etc), and
    assembly of (whole or part of) the system matrix. For problems on multiple
    grids, an assembly will only be a partial (considering a single grid)- the
    full system matrix for multiple grids is handled somewhere else.

    Known realizations of the elliptic discretization are:
        Tpfa: Finite volume method using a wwo-point flux approximation.
        Mpfa: Finite volume method using a multi-point flux approximation.
        RT0: Mixed finite element method, using the lowest order Raviart-Thomas
            elements.
        MVEM: Mixed formulation of the lowest order virtual element method.

    Attributes:
        keyword (str): This is used to identify operators and parameters that
            the discretization will work on.

    """

    def __init__(self, keyword):
        """ Set the discretization, with the keyword used for storing various
        information associated with the discretization.

        Paramemeters:
            keyword (str): Identifier of all information used for this
                discretization.
        """
        self.keyword = keyword

    def _key(self):
        """ Get the keyword of this object, on a format friendly to access relevant
        fields in the data dictionary

        Returns:
            String, on the form self.keyword + '_'.

        """
        return self.keyword + "_"

    def ndof(self, g):
        """ Abstract method. Return the number of degrees of freedom associated to the method.

        Parameters
            g (grid): Computational grid

        Returns:
            int: the number of degrees of freedom.

        """
        raise NotImplementedError("Method not implemented")

    def extract_pressure(self, g, solution_array, d):
        """ Abstract method. Extract the pressure part of a solution.

        The implementation will depend what are the primary variables of the specific implementation.

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
        raise NotImplementedError("Method not implemented")

    def extract_flux(self, g, solution_array, d):
        """ Abstract method. Extract the pressure part of a solution.

        The implementation will depend what are the primary variables of the specific implementation.

        TODO: We should incrude the boundary condition as well?

        Parameters:
            g (grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid obtained from
                either a mono-dimensional or a mixed-dimensional problem. Will
                correspond to the pressure solution.
            d (dictionary): Data dictionary associated with the grid.

        Returns:
            np.array (g.num_faces): Flux vector.

        """
        raise NotImplementedError("Method not implemented")

    def assemble_matrix_rhs(self, g, data):
        """ Return the matrix and right-hand side for a discretization of a second
        order elliptic equation.

        Also discretize the necessary operators if the data dictionary does not
        contain the necessary fields. The decision on whether to discretize
        will depend on the specific numerical method.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization. The
                size of the matrix will depend on the specific discretization.
            np.ndarray: Right hand side vector with representation of boundary
                conditions. The size of the vector will depend on the
                discretization.

        """
        raise NotImplementedError("Method not implemented")

    def assemble_matrix(self, g, data):
        """ Return the matrix for a discretization of a second order elliptic equation.

        Also discretize the necessary operators if the data dictionary does not
        contain the necessary fields. The decision on whether to discretize
        will depend on the specific numerical method.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization. The
                size of the matrix will depend on the specific discretization.

        """
        raise NotImplementedError("Method not implemented")

    # ------------------------------------------------------------------------------#

    def assemble_rhs(self, g, data):
        """ Return the right-hand side for a discretization of a second
        order elliptic equation.

        Also discretize the necessary operators if the data dictionary does not
        contain the necessary fields. The decision on whether to discretize
        will depend on the specific numerical method.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            np.ndarray: Right hand side vector with representation of boundary
                conditions. The size of the vector will depend on the
                discretization.

        """
        raise NotImplementedError("Method not implemented")

    def assemble_int_bound_flux(
        self, g, data, data_edge, grid_swap, cc, matrix, self_ind
    ):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a flux boundary condition.

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
        raise NotImplementedError("Method not implemented")

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
        raise NotImplementedError("Method not implemented")

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
        raise NotImplementedError("Method not implemented")

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
        raise NotImplementedError("Method not implemented")

    def enforce_neumann_int_bound(
        self, g_master, data_edge, matrix, swap_grid, self_ind
    ):
        """ Enforce Neumann boundary conditions on a given system matrix.

        Methods based on a mixed variational form will need this function to
        implement essential boundary conditions.

        The discretization matrix should be modified in place.

        Parameters:
            g (Grid): On which the equation is discretized
            data (dictionary): Of data related to the discretization.
            matrix (scipy.sparse.matrix): Discretization matrix to be modified.

        """
        raise NotImplementedError("Method not implemented")


class VectorEllipticDiscretization(Solver):
    """ Superclass for finite volume discretizations of the vector elliptic equation.

    Should not be used by itself, instead use a subclass that implements an
    actual discretization method. Known subclass is Mpsa.

    """

    def __init__(self, keyword):
        self.keyword = keyword

    def _key(self):
        """ Get the keyword of this object, on a format friendly to access relevant
        fields in the data dictionary

        Returns:
            String, on the form self.keyword + '_'.

        """
        return self.keyword + "_"

    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of cells times dimension (stress dof).

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        raise NotImplementedError("Method not implemented")

    def extract_displacement(self, g, solution_array, d):
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
        raise NotImplementedError("Method not implemented")

    def extract_traction(self, g, solution_array, d):
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
        raise NotImplementedError()

    # ------------------------------------------------------------------------------#

    def assemble_matrix_rhs(self, g, data):

        """
        Return the matrix and right-hand side for a discretization of a second
        order elliptic equation using a FV method with a multi-point stress
        approximation.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data. For details on necessary keywords,
            see method discretize()
        discretize (boolean, optional): default True. Whether to discetize
            prior to matrix assembly. If False, data should already contain
            discretization.

        Return
        ------
        matrix: sparse csr (g.dim * g_num_cells, g.dim * g_num_cells)
            Discretization matrix.
        rhs: array (g.dim * g_num_cells)
            Right-hand side which contains the boundary conditions and the vector
            source term.
        """
        raise NotImplementedError("Method not implemented")

    def assemble_matrix(self, g, data):
        """
        Return the matrix for a discretization of a second order elliptic equation
        using a FV method.

        The name of data in the input dictionary (data) are:
        k : FourthOrderTensor
            stiffness tensor defined cell-wise.
        bc : boundary conditions (optional)
        bc_val : dictionary (optional)
            Values of the boundary conditions. The dictionary has at most the
            following keys: 'dir', 'neu' 'rob', for Dirichlet, Neumann and Robin
            boundary conditions, respectively.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization. The
                size of the matrix will depend on the specific discretization.
        """
        raise NotImplementedError("Method not implemented")

    # ------------------------------------------------------------------------------#

    def assemble_rhs(self, g, data):
        """ Return the right-hand side for a discretization of a second
        order elliptic equation using a finite volume method.

        Also discretize the necessary operators if the data dictionary does not
        contain a discretization of the boundary equation.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            np.ndarray: Right hand side vector with representation of boundary
                conditions. The size of the vector will depend on the
                discretization.
        """
        raise NotImplementedError("Method not implemented")

    def assemble_int_bound_stress(
        self, g, data, data_edge, grid_swap, cc, matrix, self_ind
    ):
        """Assemble the contribution from an internal boundary, manifested as a
        stress boundary condition.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose stress continuity on a higher-dimensional domain.

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
        raise NotImplementedError("Method not implemented")

    def assemble_int_bound_source(
        self, g, data, data_edge, grid_swap, cc, matrix, self_ind
    ):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a body force term.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose stress continuity on a lower-dimensional domain.

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
        raise NotImplementedError("Method not implemented")

    def assemble_int_bound_displacement_trace(
        self, g, data, data_edge, grid_swap, cc, matrix, self_ind
    ):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a condition on the boundary displacement.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose stress continuity on a higher-dimensional domain.

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
        raise NotImplementedError("Method not implemented")

    def assemble_int_bound_displacement_cell(
        self, g, data, data_edge, grid_swap, cc, matrix, self_ind
    ):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a condition on the cell displacement.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose stress continuity on a lower-dimensional domain.

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
        raise NotImplementedError("Method not implemented")

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
        raise NotImplementedError("Method not implemented")
