"""
Module contains superclass for mpfa and tpfa.
"""
from warnings import warn
import numpy as np
import scipy.sparse as sps

import porepy as pp


class FVElliptic(pp.EllipticDiscretization):
    """Superclass for finite volume discretizations of the elliptic equation.

    Should not be used by itself, instead use a subclass that implements an
    actual discretization method. Known subclasses are Tpfa and Mpfa.

    """

    def __init__(self, keyword):

        # Identify which parameters to use:
        self.keyword = keyword

        # Keywords used to identify individual terms in the discretization matrix dictionary
        # The flux discretization (transmissibility matrix)
        self.flux_matrix_key = "flux"
        # Discretization of boundary conditions.
        self.bound_flux_matrix_key = "bound_flux"
        # Contribution of cell center values in reconstruction of boundary pressures
        self.bound_pressure_cell_matrix_key = "bound_pressure_cell"
        # Contribution of boundary values (Neumann or Dirichlet, depending on the
        # condition set on faces) in reconstruction of boundary pressures
        self.bound_pressure_face_matrix_key = "bound_pressure_face"
        # Discretization of vector source terms (gravity)
        self.vector_source_matrix_key = "vector_source"
        self.bound_pressure_vector_source_matrix_key = "bound_pressure_vector_source"

    def ndof(self, sd: pp.Grid) -> int:
        """Return the number of degrees of freedom associated to the method.

        Args:
            sd (pp.Grid): A grid.

        Returns:
            int: The number of degrees of freedom.

        """
        return sd.num_cells

    def extract_pressure(
        self, sd: pp.Grid, solution_array: np.ndarray, data: dict
    ) -> np.ndarray:
        """Extract the pressure part of a solution.
        The method is trivial for finite volume methods, with the pressure
        being the only primary variable.

        Args:
            sd (pp.Grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid obtained from
                either a mono-dimensional or a mixed-dimensional problem.
            data (dictionary): Data dictionary associated with the grid. Not used,
                but included for consistency reasons.
        Returns:
            np.array (g.num_cells): Pressure solution vector. Will be identical
                to solution_array.

        """
        return solution_array

    def extract_flux(
        self, sd: pp.Grid, solution_array: np.ndarray, data: dict
    ) -> np.ndarray:
        """Extract the flux related to a solution.

        The flux is computed from the discretization and the given pressure solution.

        Args:
            sd (pp.Grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid obtained from
                either a mono-dimensional or a mixed-dimensional problem. Will
                correspond to the pressure solution.
            data (dictionary): Data dictionary associated with the grid.

        Returns:
            np.array (g.num_faces): Flux vector.

        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]

        flux = matrix_dictionary[self.flux_matrix_key].tocsr()
        bound_flux = matrix_dictionary[self.bound_flux_matrix_key].tocsr()

        bc_val = parameter_dictionary["bc_values"]

        return flux * solution_array + bound_flux * bc_val

    def assemble_matrix_rhs(
        self, sd: pp.Grid, data: dict
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Return the matrix and right-hand side for a discretization of a second
        order elliptic equation.

        Args:
            sd (pp.Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization. The size of
                the matrix will depend on the specific discretization.
            np.ndarray: Right-hand side vector with representation of boundary
                conditions. The size of the vector will depend on the discretization.

        """

        return self.assemble_matrix(sd, data), self.assemble_rhs(sd, data)

    def assemble_matrix(self, sd: pp.Grid, data: dict) -> sps.spmatrix:
        """Return the matrix for a discretization of a second order elliptic equation
        using a FV method.

        Args:
            sd (pp.Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization. The
                size of the matrix will depend on the specific discretization.

        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        div = pp.fvutils.scalar_divergence(sd)
        flux = matrix_dictionary[self.flux_matrix_key]
        if flux.shape[0] != sd.num_faces:
            hf2f = pp.fvutils.map_hf_2_f(nd=1, sd=sd)
            flux = hf2f * flux

        M = div * flux

        return M

    def assemble_rhs(self, sd: pp.Grid, data: dict) -> np.ndarray:
        """Return the right-hand side for a discretization of a second order elliptic
        equation using a finite volume method.

        Args:
            sd (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            np.ndarray: Right hand side vector with representation of boundary
                conditions. The size of the vector will depend on the discretization.

        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        bound_flux = matrix_dictionary[self.bound_flux_matrix_key]
        if sd.dim > 0 and bound_flux.shape[0] != sd.num_faces:
            hf2f = pp.fvutils.map_hf_2_f(nd=1, sd=sd)
            bound_flux = hf2f * bound_flux

        parameter_dictionary = data[pp.PARAMETERS][self.keyword]

        bc_val = parameter_dictionary["bc_values"]

        div = sd.cell_faces.T

        val = -div * bound_flux * bc_val

        # Also assemble vector sources.
        # Discretization of the vector source term if specified

        if "vector_source" in parameter_dictionary:
            vector_source_discr = matrix_dictionary[self.vector_source_matrix_key]
            vector_source = parameter_dictionary.get("vector_source")
            val -= div * vector_source_discr * vector_source

        return val

    def assemble_int_bound_flux(
        self,
        sd,
        data,
        intf,
        data_edge,
        cc,
        matrix,
        rhs,
        self_ind,
        use_secondary_proj=False,
    ):
        """Assemble the contribution from an internal boundary, manifested as a
        flux boundary condition.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a higher-dimensional domain.

        Implementations of this method will use an interplay between the grid
        on the node and the mortar grid on the relevant edge.

        Args:
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
        msg = """This function is deprecated and will be removed, most likely in the
        second half of 2022.
        
        To assemble mixed-dimensional elliptic problems, the recommended solution is
        either to use the models, or to use the automatic differentiation framework
        directly.
        """
        warn(msg, DeprecationWarning, stacklevel=2)

        div = sd.cell_faces.T

        bound_flux = data[pp.DISCRETIZATION_MATRICES][self.keyword][
            self.bound_flux_matrix_key
        ]

        if use_secondary_proj:
            proj = intf.mortar_to_secondary_int()
        else:
            proj = intf.mortar_to_primary_int()

        if sd.dim > 0 and bound_flux.shape[0] != sd.num_faces:
            # If bound flux is gven as sub-faces we have to map it from sub-faces
            # to faces
            hf2f = pp.fvutils.map_hf_2_f(nd=1, g=sd)
            bound_flux = hf2f * bound_flux
        if sd.dim > 0 and bound_flux.shape[1] != proj.shape[0]:
            raise ValueError(
                """Inconsistent shapes. Did you define a
            sub-face boundary condition but only a face-wise mortar?"""
            )

        cc[self_ind, 2] += div * bound_flux * proj

    def assemble_int_bound_source(
        self, sd, data, intf, data_edge, cc, matrix, rhs, self_ind
    ):
        """Abstract method. Assemble the contribution from an internal
        boundary, manifested as a source term.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a lower-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Args:
            sd (Grid): Grid which the condition should be imposed on.
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

        """
        msg = """This function is deprecated and will be removed, most likely in the
        second half of 2022.
        
        To assemble mixed-dimensional elliptic problems, the recommended solution is
        either to use the models, or to use the automatic differentiation framework
        directly.
        """
        warn(msg, DeprecationWarning, stacklevel=2)

        proj = intf.mortar_to_secondary_int()

        cc[self_ind, 2] -= proj

    def assemble_int_bound_pressure_trace(
        self,
        sd,
        data,
        intf,
        data_edge,
        cc,
        matrix,
        rhs,
        self_ind,
        use_secondary_proj=False,
        assemble_matrix=True,
        assemble_rhs=True,
    ):
        """Assemble the contribution from an internal
        boundary, manifested as a condition on the boundary pressure.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a higher-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Args:
            sd (Grid): Grid which the condition should be imposed on.
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
        msg = """This function is deprecated and will be removed, most likely in the
        second half of 2022.
        
        To assemble mixed-dimensional elliptic problems, the recommended solution is
        either to use the models, or to use the automatic differentiation framework
        directly.
        """
        warn(msg, DeprecationWarning, stacklevel=2)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]

        if use_secondary_proj:
            proj = intf.secondary_to_mortar_avg()
            proj_int = intf.mortar_to_secondary_int()
        else:
            proj = intf.primary_to_mortar_avg()
            if assemble_matrix:
                proj_int = intf.mortar_to_primary_int()

        if assemble_matrix:
            assert cc is not None
            cc[2, self_ind] += (
                proj * matrix_dictionary[self.bound_pressure_cell_matrix_key]
            )
            cc[2, 2] += (
                proj * matrix_dictionary[self.bound_pressure_face_matrix_key] * proj_int
            )
        # Add contribution from boundary conditions to the pressure at the fracture
        # faces. For TPFA this will be zero, but for MPFA we will get a contribution
        # on the fractures extending to the boundary due to the interaction region
        # around a node.
        if assemble_rhs:
            bc_val = parameter_dictionary["bc_values"]
            rhs[2] -= (
                proj * matrix_dictionary[self.bound_pressure_face_matrix_key] * bc_val
            )

            # Add gravity contribution if relevant
            if "vector_source" in parameter_dictionary:
                vector_source_discr = matrix_dictionary[
                    self.bound_pressure_vector_source_matrix_key
                ]
                # The vector source, defaults to zero if not specified.
                vector_source = parameter_dictionary.get("vector_source")
                rhs[2] -= proj * vector_source_discr * vector_source

    def assemble_int_bound_pressure_trace_rhs(
        self, sd, data, intf, data_edge, cc, rhs, self_ind, use_secondary_proj=False
    ):
        """Assemble the rhs contribution from an internal
        boundary, manifested as a condition on the boundary pressure.

        For details, see self.assemble_int_bound_pressure_trace()

        Args:
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
        msg = """This function is deprecated and will be removed, most likely in the
        second half of 2022.
        
        To assemble mixed-dimensional elliptic problems, the recommended solution is
        either to use the models, or to use the automatic differentiation framework
        directly.
        """
        warn(msg, DeprecationWarning, stacklevel=2)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]

        if use_secondary_proj:
            proj = intf.secondary_to_mortar_avg()
        else:
            proj = intf.primary_to_mortar_avg()

        # Add contribution from boundary conditions to the pressure at the fracture
        # faces. For TPFA this will be zero, but for MPFA we will get a contribution
        # on the fractures extending to the boundary due to the interaction region
        # around a node.
        bc_val = parameter_dictionary["bc_values"]
        rhs[2] -= proj * matrix_dictionary[self.bound_pressure_face_matrix_key] * bc_val

        # Add gravity contribution if relevant
        if "vector_source" in parameter_dictionary:
            vector_source_discr = matrix_dictionary[
                self.bound_pressure_vector_source_matrix_key
            ]

            # The vector source, defaults to zero if not specified.
            vector_source = parameter_dictionary.get("vector_source")
            rhs[2] -= proj * vector_source_discr * vector_source

    def assemble_int_bound_pressure_trace_between_interfaces(
        self, sd, data_grid, proj_primary, proj_secondary, cc, matrix, rhs
    ):
        """Assemble the contribution from an internal
        boundary, manifested as a condition on the boundary pressure.

        Args:
            g (Grid): Grid which the condition should be imposed on.
            data_grid (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            proj_primary (sparse matrix): Pressure projection from the higher-dim
                grid to the primary mortar grid.
            proj_secondary (sparse matrix): Flux projection from the secondary mortar
                grid to the main grid.
            cc (block matrix, 3x3): Block matrix of size 3 x 3, whwere each block represents
                coupling between variables on this interface. Index 0, 1 and 2
                represent the primary grid, the primary and secondary interface,
                respectively.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Block matrix of size 3 x 1, representing the right hand
                side of this coupling. Index 0, 1 and 2 represent the primary grid,
                the primary and secondary interface, respectively.

        """
        msg = """This function is deprecated and will be removed, most likely in the
        second half of 2022.
        
        To assemble mixed-dimensional elliptic problems, the recommended solution is
        either to use the models, or to use the automatic differentiation framework
        directly.
        """
        warn(msg, DeprecationWarning, stacklevel=2)

        matrix_dictionary = data_grid[pp.DISCRETIZATION_MATRICES][self.keyword]

        cc[1, 2] += (
            proj_primary
            * matrix_dictionary[self.bound_pressure_face_matrix_key]
            * proj_secondary
        )

    def assemble_int_bound_pressure_cell(
        self, sd, data, intf, data_edge, cc, matrix, rhs, self_ind
    ):
        """Abstract method. Assemble the contribution from an internal
        boundary, manifested as a condition on the cell pressure.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a lower-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Args:
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
        """
        msg = """This function is deprecated and will be removed, most likely in the
        second half of 2022.
        
        To assemble mixed-dimensional elliptic problems, the recommended solution is
        either to use the models, or to use the automatic differentiation framework
        directly.
        """
        warn(msg, DeprecationWarning, stacklevel=2)

        proj = intf.secondary_to_mortar_avg()

        cc[2, self_ind] -= proj

    def enforce_neumann_int_bound(
        self, sd, intf: pp.MortarGrid, data_edge, matrix, self_ind
    ):
        """Enforce Neumann boundary conditions on a given system matrix.

        The method is void for finite volume approaches, but is implemented
        to be compatible with the general framework.

        Args:
            g (Grid): On which the equation is discretized
            data (dictionary): Of data related to the discretization.
            matrix (scipy.sparse.matrix): Discretization matrix to be modified.
        """
        # Operation is void for finite volume methods
        pass


class EllipticDiscretizationZeroPermeability(FVElliptic):
    """Specialized discretization for domains with zero tangential permeability.

    Intended usage is to impose full continuity conditions between domains of higher
    dimensions separated by a lower-dimensional domain (think two intersecting
    fractures), in cases where one does not want to eliminate the lower-dimensional
    domain from the MixedDimensionalGrid. The class is designed to interact with the class
    FluxPressureContinuity. Wider usage is possible, but be cautious.

    The subclassing from FVElliptic was convenient, but other options could also have
    worked.

    NOTICE: There seems  no point in assigning this method as the higher-dimensional
    discretization. Accordingly, the methods for assembly of interface contributions
    from the primary side of a mortar grid are delibierately designed to fail.

    """

    def discretize(self, sd, data):
        """
        Formal discretization method - nothing to do here.

        Args
        ----------
        g (pp.Grid): grid, or a subclass.
        data (dict).

        """
        pass

    def assemble_matrix(self, sd, data):
        """Assemble system matrix. Will be zero matrix of appropriate size.

        Args:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: Zero matrix.

        """
        return sps.csc_matrix((self.ndof(sd), self.ndof(sd)))

    def assemble_rhs(self, sd, data):
        """Assemble right hand side vector. Will be zero vector of appropriate size.

        Args:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            np.array: Zero vector.

        """

        return np.zeros(self.ndof(sd))

    def assemble_int_bound_flux(
        self, sd, data, data_edge, cc, matrix, rhs, self_ind, use_secondary_proj=False
    ):
        """Assemble the contribution from an internal boundary, manifested as a
        flux boundary condition.

        This method should not be used for the zero-permeability case; it would
        require a flux in the higher-dimensional grid. Therefore raise an error if
        this method is invoked.

        Args:
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
        raise NotImplementedError(
            """This class should not be used as a
                                  higher-dimensional discretization"""
        )

    def assemble_int_bound_pressure_trace(
        self, sd, data, data_edge, cc, matrix, rhs, self_ind, use_secondary_proj=False
    ):
        """Assemble the contribution from an internal
        boundary, manifested as a condition on the boundary pressure.

        This method should not be used for the zero-permeability case; it would
        require a flux in the higher-dimensional grid. Therefore raise an error if
        this method is invoked.

        Args:
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
        raise NotImplementedError(
            """This class should not be used as a
                                  higher-dimensional discretization"""
        )
