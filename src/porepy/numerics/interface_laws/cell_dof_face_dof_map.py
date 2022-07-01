"""
This module is primarily thought for the elliptic part in DFN simulations,
where the one codimensional domains are used as Lagrange
multipliers for the interface law. Use this module for the co-dimensional objects.
In this way no equations are explicitly associated
and some of the interface operators are provided.
"""
from __future__ import annotations

from typing import Optional
from warnings import warn

import numpy as np
import scipy.sparse as sps

import porepy as pp


class CellDofFaceDofMap:
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

    def __init__(self, keyword: str) -> None:
        """Set the discretization, with the keyword used for storing various
        information associated with the discretization.

        Paramemeters:
            keyword (str): Identifier of all information used for this
                discretization.
        """
        self.keyword = keyword

    def _key(self) -> str:
        """Get the keyword of this object, on a format friendly to access relevant
        fields in the data dictionary

        Returns:
            String, on the form self.keyword + '_'.

        """
        return self.keyword + "_"

    def ndof(self, sd: pp.Grid) -> int:
        """Return the number of degrees of freedom, in this case the number of cells

        Parameters
            sd (pp.Grid): Computational grid

        Returns:
            int: the number of degrees of freedom (number of cells).

        """
        return sd.num_cells

    def extract_pressure(self, sd: pp.Grid, solution_array, data=None) -> np.ndarray:
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

    def extract_flux(
        self, sd: pp.Grid, solution_array: np.ndarray, data=None
    ) -> np.ndarray:
        """We return an empty vector for consistency with the previous method.

        Parameters:
            sd (grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid obtained from
                either a mono-dimensional or a mixed-dimensional problem.
            data (dictionary): Data dictionary associated with the grid. Not used.

        Returns:
            np.array (0): Empty vector.

        """
        return np.empty(0)

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        """Construct discretization matrices. Operation is void for this discretization.

        Parameters:
            sd (pp.Grid): Grid to be discretized.
            data (dictionary): With discretization parameters.

        """
        pass

    def assemble_matrix_rhs(
        self, sd: pp.Grid, data: Optional[dict] = None
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Return the matrix and right-hand side, no PDE are associate with
        this discretization so empty matrix and zero rhs is returned.

        Parameters:
            sd (pp.Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization.
            np.ndarray: zero right hand side vector.

        """
        return self.assemble_matrix(sd, data), self.assemble_rhs(sd, data)

    def assemble_matrix(self, sd: pp.Grid, data: Optional[dict] = None) -> sps.spmatrix:
        """An empty matrix with size num_cells x num_cells.

        Parameters:
            sd (pp.Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored. Not used.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization.

        """
        return sps.csr_matrix((self.ndof(sd), self.ndof(sd)))

    def assemble_rhs(self, sd: pp.Grid, data: Optional[dict] = None) -> np.ndarray:
        """Zero right-hand side vector.

        Parameters:
            sd (pp.Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored. Not used.

        Returns:
            np.ndarray: Zero right hand side vector.

        """
        msg = """This function is deprecated and will be removed, most likely in the
        second half of 2022.

        To assemble mixed-dimensional problems, the recommended solution is
        either to use the models, or to use the automatic differentiation framework
        directly.
        """
        warn(msg, DeprecationWarning, stacklevel=2)

        return np.zeros(self.ndof(sd))

    def assemble_int_bound_flux(
        self, sd, data, data_intf, cc, matrix, rhs, self_ind, use_secondary_proj
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
            sd (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_intf (dictionary): Data dictionary for the edge in the
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
        msg = """This function is deprecated and will be removed, most likely in the
        second half of 2022.

        To assemble mixed-dimensional problems, the recommended solution is
        either to use the models, or to use the automatic differentiation framework
        directly.
        """
        warn(msg, DeprecationWarning, stacklevel=2)

        raise NotImplementedError("Method not implemented")

    def assemble_int_bound_source(
        self, sd, data, intf, data_intf, cc, matrix, rhs, self_ind
    ):
        """Assemble the contribution from an internal boundary,
        manifested as a source term.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a lower-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Parameters:
            sd (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_intf (dictionary): Data dictionary for the edge in the
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
        msg = """This function is deprecated and will be removed, most likely in the
        second half of 2022.

        To assemble mixed-dimensional problems, the recommended solution is
        either to use the models, or to use the automatic differentiation framework
        directly.
        """
        warn(msg, DeprecationWarning, stacklevel=2)

        proj = intf.secondary_to_mortar_avg()

        cc[self_ind, 2] -= proj.T

    def assemble_int_bound_pressure_trace(
        self, sd, data, data_intf, cc, matrix, rhs, self_ind, use_secondary_proj
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
            data_intf (dictionary): Data dictionary for the edge in the
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

    def assemble_int_bound_pressure_cell(
        self, sd, data, intf, data_intf, cc, matrix, rhs, self_ind
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
            data_intf (dictionary): Data dictionary for the edge in the
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
        msg = """This function is deprecated and will be removed, most likely in the
        second half of 2022.

        To assemble mixed-dimensional problems, the recommended solution is
        either to use the models, or to use the automatic differentiation framework
        directly.
        """
        warn(msg, DeprecationWarning, stacklevel=2)

        proj = intf.secondary_to_mortar_avg()

        cc[2, self_ind] -= proj

    def enforce_neumann_int_bound(self, sd_primary, data_intf, matrix, self_ind):
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
