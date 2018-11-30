"""
Module contains superclass for mpfa and tpfa.
"""
import scipy.sparse as sps
import numpy as np

import porepy as pp


class FVElliptic(pp.numerics.mixed_dim.EllipticDiscretization):
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
        contain a transmissibility matrix.

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

        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

    def assemble_matrix(self, g, data):
        """
        Return the matrix for a discretization of a second order elliptic equation
        using a FV method.

        The name of data in the input dictionary (data) are:
        k : second_order_tensor
            Permeability defined cell-wise.
        bc : boundary conditions (optional)
        bc_val : dictionary (optional)
            Values of the boundary conditions. The dictionary has at most the
            following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
            conditions, respectively.

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



class FVVectorElliptic(pp.numerics.mixed_dim.solver.Solver):
    """ Superclass for finite volume discretizations of the elliptic equation.

    Should not be used by itself, instead use a subclass that implements an
    actual discretization method. Known subclasses are Tpfa and Mpfa.

    """

    def __init__(self, keyword, physics=None):
        self.keyword = keyword

        # @ALL: We kee the physics keyword for now, or else we completely
        # break the parameter assignment workflow. The physics keyword will go
        # to be replaced by a more generalized approach, but one step at a time
        if physics is None:
            self.physics = keyword
        else:
            self.physics = physics

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
        return g.dim * g.num_cells

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
        return solution_array

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

    def assemble_matrix_rhs(self, g, data, discretize=True, **kwargs):

        """
        Return the matrix and right-hand side for a discretization of a fourth
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
        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

    def assemble_matrix(self, g, data):
        """
        Return the matrix for a discretization of a fourth order elliptic equation
        using a FV method.

        The name of data in the input dictionary (data) are:
        k : FourtOrderTensor
            stress tensor defined cell-wise.
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
        if not self._key() + "stress" in data.keys():
            self.discretize(g, data)

        div = pp.fvutils.vector_divergence(g)
        stress = data[self._key() + "stress"]
        if stress.shape[0] != g.dim * g.num_faces:
            hf2f = pp.fvutils.map_hf_2_f(g=g)
            stress = hf2f * stress
        M = div * stress

        return M

    # ------------------------------------------------------------------------------#

    def assemble_rhs(self, g, data):
        """ Return the right-hand side for a discretization of a fourth
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
        if not self._key() + "bound_stress" in data.keys():
            self.discretize(g, data)

        bound_stress = data[self._key() + "bound_stress"]
        if bound_stress.shape[0] != g.dim * g.num_faces:
            hf2f = pp.fvutils.map_hf_2_f(g=g)
            bound_stress = hf2f * bound_stress

        param = data["param"]

        bc_val = param.get_bc_val(self)

        div = pp.fvutils.vector_divergence(g)

        return -div * bound_stress * bc_val + param.get_source(self)

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
        div = pp.fvutils.vector_divergence(g)

        # Projection operators to grid
        mg = data_edge["mortar_grid"]

        if grid_swap:
            proj = mg.slave_to_mortar_avg()
        else:
            proj = mg.master_to_mortar_avg()
        # Expand indices as Fortran.
        proj_int = sps.kron(proj, sps.eye(g.dim)).tocsr()

        bound_stress = data[self._key() + "bound_stress"]

        if bound_stress.shape[0] != g.dim * g.num_faces:
            # If bound stress is gven as sub-faces we have to map it from sub-faces
            # to faces
            hf2f = pp.fvutils.map_hf_2_f(g=g)
            bound_stress = hf2f * bound_stress
        if bound_stress.shape[1] != proj_int.shape[1]:
            raise ValueError('''Inconsistent shapes. Did you define a
            sub-face boundary condition but only a face-wise mortar?''')
        cc[self_ind, 2] += div * bound_stress * proj_int.T

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
        mg = data_edge["mortar_grid"]

        if grid_swap:
            proj = mg.master_to_mortar_avg()
        else:
            proj = mg.slave_to_mortar_avg()
        proj = sps.kron(proj, sps.eye(g.dim)).tocsr()

        cc[self_ind, 2] -= proj.T

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
        mg = data_edge["mortar_grid"]

        # TODO: this should become first or second or something
        if grid_swap:
            proj = mg.slave_to_mortar_avg()
            proj_int = mg.slave_to_mortar_int
            proj_swap = mg.master_to_mortar_avg()
            proj_int_swap = mg.master_to_mortar_int
        else:
            proj = mg.master_to_mortar_avg()
            proj_int = mg.master_to_mortar_int
            proj_swap = mg.slave_to_mortar_avg()
            proj_int_swap = mg.slave_to_mortar_int

        # Expand indices as Fortran indexes
        proj_avg = sps.kron(proj, sps.eye(g.dim)).tocsr()
        proj_int = sps.kron(proj_int, sps.eye(g.dim)).tocsr()
        proj_avg_swap = sps.kron(proj_swap, sps.eye(g.dim)).tocsr()
        proj_int_swap = sps.kron(proj_int_swap, sps.eye(g.dim)).tocsr()

        bp = data[self._key() + "bound_displacement_cell"]
        if proj_avg.shape[1] == g.dim * g.num_faces:
            # In this case we the projection is from faces to cells
            # We therefore need to map the boundary displacements which is given as
            # half-faces to faces.
            hf2f = pp.fvutils.map_hf_2_f(g=g)
            num_nodes = np.diff(g.face_nodes.indptr)
            weight = sps.kron(sps.eye(g.dim), sps.diags(1 / num_nodes))
            # hf2f adds all subface values to one face values. For the displacement we want
            # to take the average, therefore we divide each face by the number of subfaces.
            cc[2, self_ind] += proj_avg * weight * hf2f * bp
            cc[2, 2] += proj_avg * weight * hf2f * data[self._key() + "bound_displacement_face"] *proj_int.T
        else:
            cc[2, self_ind] += proj_avg * bp
            cc[2, 2] += proj_avg * data[self._key() + "bound_displacement_face"] * proj_int.T
            # Add the contibution to the displacement for the other mortar. This can 
            # typically happen if you simulate the contact between the two sides of a
            # fracture. The interaction region around the nodes on the edge will then
            # get a contribution from both sides. We need a negative sign because the
            # tractions T_s = -T_m has different sign.
            cc[2, 2] -= proj_avg * data[self._key() + "bound_displacement_face"] * proj_int_swap.T
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
        mg = data_edge["mortar_grid"]

        if grid_swap:
            proj = mg.master_to_mortar_avg()
        else:
            proj = mg.slave_to_mortar_avg()
        proj = sps.kron(sps.eye(g.dim), proj).tocsr()

        cc[2, self_ind] -= proj

    def enforce_neumann_int_bound(self, g_master, data_edge, matrix, swap_grid, self_ind):
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

