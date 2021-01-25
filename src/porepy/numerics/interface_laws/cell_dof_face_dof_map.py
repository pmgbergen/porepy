"""
This module is primarily thought for the elliptic part in DFN simulations,
where the one codimensional domains are used as Lagrange
multipliers for the interface law. Use this module for the co-dimensional objects.
In this way no equations are explicitly associated
and some of the interface operators are provided.
"""

import numpy as np
import scipy.sparse as sps

import porepy as pp

module_sections = ["numerics"]


class CellDofFaceDofMap(object):
    """
    Only the methods int_bound_source and int_bound_pressure_cell are implemented to allow
    appropriate interface laws. No discretization is associated to the grid.
    A possible usage is for the co-dimensional objects in a DFN discretization.

    This implementation has been tested in connection with the elliptic discretizations for the
    one higher dimensional grids:
        Tpfa: Finite volume method using a two-point flux approximation.
        RT0: Mixed finite element method, using the lowest order Raviart-Thomas
            elements.
        MVEM: Mixed formulation of the lowest order virtual element method.
    And the the interface law between the higher and current grid:
        FluxPressureContinuity

    Attributes:
        keyword (str): This is used to identify operators and parameters that
            the discretization will work on.

    """

    @pp.time_logger(sections=module_sections)
    def __init__(self, keyword):
        """Set the discretization, with the keyword used for storing various
        information associated with the discretization.

        Paramemeters:
            keyword (str): Identifier of all information used for this
                discretization.
        """
        self.keyword = keyword

    @pp.time_logger(sections=module_sections)
    def _key(self):
        """Get the keyword of this object, on a format friendly to access relevant
        fields in the data dictionary

        Returns:
            String, on the form self.keyword + '_'.

        """
        return self.keyword + "_"

    @pp.time_logger(sections=module_sections)
    def ndof(self, g):
        """Return the number of degrees of freedom, in this case the number of cells

        Parameters
            g (grid): Computational grid

        Returns:
            int: the number of degrees of freedom (number of cells).

        """
        return g.num_cells

    @pp.time_logger(sections=module_sections)
    def extract_pressure(self, g, solution_array, data=None):
        """Return exactly the solution_array itself.

        Parameters:
            g (grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid obtained from
                either a mono-dimensional or a mixed-dimensional problem.
            data (dictionary): Data dictionary associated with the grid. Not used.

        Returns:
            np.array (g.num_cells): Pressure solution vector. Will be identical
                to solution_array.

        """
        return solution_array

    @pp.time_logger(sections=module_sections)
    def extract_flux(self, g, solution_array, data=None):
        """We return an empty vector for consistency with the previous method.

        Parameters:
            g (grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid obtained from
                either a mono-dimensional or a mixed-dimensional problem.
            data (dictionary): Data dictionary associated with the grid. Not used.

        Returns:
            np.array (0): Empty vector.

        """
        return np.empty(0)

    @pp.time_logger(sections=module_sections)
    def discretize(self, g, data):
        """Construct discretization matrices. Operation is void for this discretization.

        Parameters:
            g (pp.Grid): Grid to be discretized.
            data (dictionary): With discretization parameters.

        """
        pass

    @pp.time_logger(sections=module_sections)
    def assemble_matrix_rhs(self, g, data=None):
        """Return the matrix and right-hand side, no PDE are associate with
        this discretization so empty matrix and zero rhs is returned.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization.
            np.ndarray: zero right hand side vector.

        """
        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

    @pp.time_logger(sections=module_sections)
    def assemble_matrix(self, g, data=None):
        """An empty matrix with size num_cells x num_cells.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored. Not used.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization.

        """
        return sps.csr_matrix((self.ndof(g), self.ndof(g)))

    @pp.time_logger(sections=module_sections)
    def assemble_rhs(self, g, data=None):
        """Zero right-hand side vector.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored. Not used.

        Returns:
            np.ndarray: Zero right hand side vector.

        """
        return np.zeros(self.ndof(g))

    @pp.time_logger(sections=module_sections)
    def assemble_int_bound_flux(
        self, g, data, data_edge, cc, matrix, rhs, self_ind, use_secondary_proj
    ):
        """Abstract method. Assemble the contribution from an internal
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
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                primary and secondary side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Right hand side contribution for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.
            use_secondary_proj (boolean): If True, the secondary side projection operator is
                used. Needed for periodic boundary conditions.

        """
        raise NotImplementedError("Method not implemented")

    @pp.time_logger(sections=module_sections)
    def assemble_int_bound_source(self, g, data, data_edge, cc, matrix, rhs, self_ind):
        """Assemble the contribution from an internal boundary,
        manifested as a source term.

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
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                primary and secondary side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Right hand side contribution for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 0 or 1.

        """
        mg = data_edge["mortar_grid"]

        proj = mg.secondary_to_mortar_avg()

        cc[self_ind, 2] -= proj.T

    @pp.time_logger(sections=module_sections)
    def assemble_int_bound_pressure_trace(
        self, g, data, data_edge, cc, matrix, rhs, self_ind, use_secondary_proj
    ):
        """Abstract method. Assemble the contribution from an internal
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
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                primary and secondary side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Right hand side contribution for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.
            use_secondary_proj (boolean): If True, the secondary side projection operator is
                used. Needed for periodic boundary conditions.

        """
        raise NotImplementedError("Method not implemented")

    @pp.time_logger(sections=module_sections)
    def assemble_int_bound_pressure_cell(
        self, g, data, data_edge, cc, matrix, rhs, self_ind
    ):
        """Assemble the contribution from an internal
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
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                primary and secondary side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Right hand side contribution for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 0 or 1.

        """
        mg = data_edge["mortar_grid"]

        proj = mg.secondary_to_mortar_avg()

        cc[2, self_ind] -= proj

    @pp.time_logger(sections=module_sections)
    def enforce_neumann_int_bound(self, g_primary, data_edge, matrix, self_ind):
        """Enforce Neumann boundary conditions on a given system matrix.

        Methods based on a mixed variational form will need this function to
        implement essential boundary conditions.

        The discretization matrix should be modified in place.

        Parameters:
            g (Grid): On which the equation is discretized
            data (dictionary): Of data related to the discretization.
            matrix (scipy.sparse.matrix): Discretization matrix to be modified.

        """
        raise NotImplementedError("Method not implemented")
