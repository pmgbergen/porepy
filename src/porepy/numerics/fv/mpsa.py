"""
Implementation of the multi-point stress appoximation method, and also terms
related to poro-elastic coupling.

The methods are very similar to those of the MPFA method, although vector
equations tend to become slightly more complex thus, it may be useful to confer
that module as well.

"""
import warnings
import numpy as np
import scipy.sparse as sps
import logging
import porepy as pp

# Module-wide logger
logger = logging.getLogger(__name__)


class Mpsa(pp.numerics.mixed_dim.EllipticDiscretization):
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

    def discretize(self, g, data):
        """
        Discretize the second order vector elliptic equation using multi-point
        stress approximation.

        The method computes traction over faces in terms of cell center displacements.

        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            parameter_dictionary, storing all parameters.
                Stored in data[pp.PARAMETERS][self.keyword].
            matrix_dictionary, for storage of discretization matrices.
                Stored in data[pp.DISCRETIZATION_MATRICES][self.keyword]

        parameter_dictionary contains the entries:
            fourth_order_tensor : (pp.FourthOrderTensor) Stiffness tensor defined cell-wise.
            bc : (BoundaryConditionVectorial) boundary conditions
            mpsa_eta: (float/np.ndarray) Optional. Range [0, 1). Location of
                displacement continuity point: eta. eta = 0 gives cont. pt. at face midpoint,
                eta = 1 at the vertex. If not given, porepy tries to set an optimal
                value. If a float is given this value is set to all subfaces, except the
                boundary (where, 0 is used). If eta is a np.ndarray its size should
                equal SubcellTopology(g).num_subfno.

        matrix_dictionary will be updated with the following entries:
            stress: sps.csc_matrix (g.dim * g.num_faces, g.dim * g.num_cells)
                stress discretization, cell center contribution 
            bound_flux: sps.csc_matrix (g.dim * g.num_faces, g.dim * g.num_faces)
                stress discretization, face contribution 
            bound_displacement_cell: sps.csc_matrix (g.dim * g.num_faces, g.dim * g.num_cells)
                Operator for reconstructing the displacement trace. Cell center contribution
            bound_displacement_face: sps.csc_matrix (g.dim * g.num_faces, g.dim * g.num_faces)
                Operator for reconstructing the displacement trace. Face contribution
            

        Hidden option (intended as "advanced" option that one should normally not
        care about):
            Half transmissibility calculation according to Ivar Aavatsmark, see
            folk.uib.no/fciia/elliptisk.pdf. Activated by adding the entry
            Aavatsmark_transmissibilities: True   to the data dictionary.

        Parameters
        ----------
        g (pp.Grid): grid, or a subclass, with geometry fields computed.
        data (dict): For entries, see above.
        faces (np.ndarray): optional. Defines active faces.
        """
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        c = parameter_dictionary["fourth_order_tensor"]
        bnd = parameter_dictionary["bc"]

        eta = parameter_dictionary.get("mpsa_eta", None)

        partial = parameter_dictionary.get("partial_update", False)
        inverter = parameter_dictionary.get("inverter", None)

        if not partial:
            stress, bound_stress, bound_displacement_cell, bound_displacement_face = mpsa(
                g, c, bnd, eta=eta, inverter=inverter
            )
            matrix_dictionary["stress"] = stress
            matrix_dictionary["bound_stress"] = bound_stress
            matrix_dictionary["bound_displacement_cell"] = bound_displacement_cell
            matrix_dictionary["bound_displacement_face"] = bound_displacement_face
        else:
            raise NotImplementedError(
                """Partial discretiation for the Mpsa class is not
            implemented. See mpsa.mpsa_partial(...)"""
            )

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
        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

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
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        if not "stress" in matrix_dictionary:
            self.discretize(g, data)

        div = pp.fvutils.vector_divergence(g)
        stress = matrix_dictionary["stress"]
        if stress.shape[0] != g.dim * g.num_faces:
            hf2f = pp.fvutils.map_hf_2_f(g=g)
            stress = hf2f * stress
        M = div * stress

        return M

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
        if not "bound_stress" in matrix_dictionary:
            self.discretize(g, data)

        bound_stress = matrix_dictionary["bound_stress"]
        if bound_stress.shape[0] != g.dim * g.num_faces:
            hf2f = pp.fvutils.map_hf_2_f(g=g)
            bound_stress = hf2f * bound_stress

        parameter_dictionary = data[pp.PARAMETERS][self.keyword]

        bc_val = parameter_dictionary["bc_values"]

        div = pp.fvutils.vector_divergence(g)

        return -div * bound_stress * bc_val + parameter_dictionary["source"]

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

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        bound_stress = matrix_dictionary["bound_stress"]

        if bound_stress.shape[0] != g.dim * g.num_faces:
            # If bound stress is gven as sub-faces we have to map it from sub-faces
            # to faces
            hf2f = pp.fvutils.map_hf_2_f(g=g)
            bound_stress = hf2f * bound_stress
        if bound_stress.shape[1] != proj_int.shape[1]:
            raise ValueError(
                """Inconsistent shapes. Did you define a
            sub-face boundary condition but only a face-wise mortar?"""
            )
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
        proj_int_swap = sps.kron(proj_int_swap, sps.eye(g.dim)).tocsr()

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        bp = matrix_dictionary["bound_displacement_cell"]
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
            cc[2, 2] += (
                proj_avg
                * weight
                * hf2f
                * matrix_dictionary["bound_displacement_face"]
                * proj_int.T
            )
        else:
            cc[2, self_ind] += proj_avg * bp
            cc[2, 2] += (
                proj_avg * matrix_dictionary["bound_displacement_face"] * proj_int.T
            )
            # Add the contibution to the displacement for the other mortar. This can
            # typically happen if you simulate the contact between the two sides of a
            # fracture. The interaction region around the nodes on the edge will then
            # get a contribution from both sides. We need a negative sign because the
            # tractions T_s = -T_m has different sign.
            cc[2, 2] -= (
                proj_avg
                * matrix_dictionary["bound_displacement_face"]
                * proj_int_swap.T
            )

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


class FracturedMpsa(Mpsa):
    """
    Subclass of MPSA for discretizing a fractured domain. Adds DOFs on each
    fracture face which describe the fracture deformation.
    """

    def __init__(self, keyword, given_traction=False, **kwargs):
        Mpsa.__init__(self, keyword, **kwargs)
        if not hasattr(self, "keyword"):
            raise AttributeError("Mpsa must assign keyword")
        self.given_traction_flag = given_traction

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
        num_fracs = np.sum(g.tags["fracture_faces"])
        return g.dim * (g.num_cells + num_fracs)

    def assemble_matrix_rhs(self, g, data, discretize=True, **kwargs):
        """
        Return the matrix and right-hand side for a discretization of a second
        order elliptic equation using a FV method with a multi-point stress
        approximation with dofs added on the fracture interfaces.

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
        matrix: sparse csr (g.dim * g_num_cells + 2 * {#of fracture faces},
                            2 * {#of fracture faces})
            Discretization matrix.
        rhs: array (g.dim * g_num_cells  + g.dim * num_frac_faces)
            Right-hand side which contains the boundary conditions and the scalar
            source term.
        """
        if discretize:
            self.discretize_fractures(g, data, **kwargs)

        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        stress = matrix_dictionary["stress"]
        bound_stress = matrix_dictionary["bound_stress"]
        b_e = matrix_dictionary["b_e"]
        A_e = matrix_dictionary["A_e"]

        if self.given_traction_flag:
            L, b_l = self.given_traction(g, stress, bound_stress)
        else:
            L, b_l = self.given_slip_distance(g, stress, bound_stress)

        bc_val = parameter_dictionary["bc_values"]

        frac_faces = np.matlib.repmat(g.tags["fracture_faces"], g.dim, 1)
        if parameter_dictionary["bc"].bc_type == "scalar":
            frac_faces = frac_faces.ravel("F")
        elif parameter_dictionary["bc"].bc_type == "vectorial":
            bc_val = bc_val.ravel("F")
        else:
            raise ValueError("Unknown boundary type")

        slip_distance = parameter_dictionary["slip_distance"]

        A = sps.vstack((A_e, L), format="csr")
        rhs = np.hstack((b_e * bc_val, b_l * (slip_distance + bc_val)))

        return A, rhs

    def rhs(self, g, data):
        """
        Return the matrix and right-hand side for a discretization of a second
        order elliptic equation using a FV method with a multi-point stress
        approximation with dofs added on the fracture interfaces.

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
        matrix: sparse csr (g.dim * g_num_cells + 2 * {#of fracture faces},
                            2 * {#of fracture faces})
            Discretization matrix.
        rhs: array (g.dim * g_num_cells  + g.dim * num_frac_faces)
            Right-hand side which contains the boundary conditions and the scalar
            source term.
        """

        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        stress = matrix_dictionary["stress"]
        bound_stress = matrix_dictionary["bound_stress"]
        b_e = matrix_dictionary["b_e"]

        if self.given_traction_flag:
            _, b_l = self.given_traction(g, stress, bound_stress)
        else:
            _, b_l = self.given_slip_distance(g, stress, bound_stress)

        bc_val = parameter_dictionary["bc_values"]

        frac_faces = np.matlib.repmat(g.tags["fracture_faces"], 3, 1)
        if parameter_dictionary["bc"].bc_type == "scalar":
            frac_faces = frac_faces.ravel("F")

        elif parameter_dictionary["bc"].bc_type == "vectorial":
            bc_val = bc_val.ravel("F")
        else:
            raise ValueError("Unknown boundary type")

        slip_distance = parameter_dictionary["slip_distance"]

        rhs = np.hstack((b_e * bc_val, b_l * (slip_distance + bc_val)))

        return rhs

    def traction(self, g, data, sol):
        """
        Extract the traction on the faces from fractured fv solution.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        sol : array (g.dim * (g.num_cells + {#of fracture faces}))
            Solution, stored as [cell_disp, fracture_disp]

        Return
        ------
        T : array (g.dim * g.num_faces)
            traction on each face

        """
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        bc_val = parameter_dictionary["bc_values"].copy()
        frac_disp = self.extract_frac_u(g, sol)
        cell_disp = self.extract_u(g, sol)

        frac_faces = (g.frac_pairs).ravel("C")

        if parameter_dictionary["bc"].bc_type == "vectorial":
            bc_val = bc_val.ravel("F")

        frac_ind = pp.utils.mcolon.mcolon(
            g.dim * frac_faces, g.dim * frac_faces + g.dim
        )
        bc_val[frac_ind] = frac_disp

        T = (
            matrix_dictionary["stress"] * cell_disp
            + matrix_dictionary["bound_stress"] * bc_val
        )

        return T

    def extract_u(self, g, sol):
        """  Extract the cell displacement from fractured fv solution.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        sol : array (g.dim * (g.num_cells + {#of fracture faces}))
            Solution, stored as [cell_disp, fracture_disp]

        Return
        ------
        u : array (g.dim * g.num_cells)
            displacement at each cell

        """
        # pylint: disable=invalid-name
        return sol[: g.dim * g.num_cells]

    def extract_frac_u(self, g, sol):
        """  Extract the fracture displacement from fractured fv solution.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        sol : array (g.dim * (g.num_cells + {#of fracture faces}))
            Solution, stored as [cell_disp, fracture_disp]

        Return
        ------
        u : array (g.dim *{#of fracture faces})
            displacement at each fracture face

        """
        # pylint: disable=invalid-name
        return sol[g.dim * g.num_cells :]

    def discretize_fractures(self, g, data, faces=None, **kwargs):
        """
        Discretize the vector elliptic equation by the multi-point stress and added
        degrees of freedom on the fracture faces

        The method computes fluxes over faces in terms of displacements in
        adjacent cells (defined as the two cells sharing the face).

        The name of data in the input dictionary (data) are:
        param : Parameter(Class). Contains the following parameters:
            tensor : fourth_order_tensor
                Permeability defined cell-wise. If not given a identity permeability
                is assumed and a warning arised.
            bc : boundary conditions (optional)
            bc_val : dictionary (optional)
                Values of the boundary conditions. The dictionary has at most the
                following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
                conditions, respectively.
            apertures : (np.ndarray) (optional) apertures of the cells for scaling of
                the face normals.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.
        """

        #    dir_bound = g.get_all_boundary_faces()
        #    bound = bc.BoundaryCondition(g, dir_bound, ['dir'] * dir_bound.size)
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        frac_faces = g.tags["fracture_faces"]

        bound = parameter_dictionary["bc"]

        if bound.bc_type == "scalar":
            bound.is_dir[frac_faces] = True
            bound.is_neu[frac_faces] = False
        elif bound.bc_type == "vectorial":
            bound.is_dir[:, frac_faces] = True
            bound.is_neu[:, frac_faces] = False
        else:
            raise ValueError("Unknow boundary condition type: " + bound.bc_type)
        if np.sum(bound.is_dir * bound.is_neu) != 0:
            raise AssertionError("Found faces that are both dirichlet and neuman")
        # Discretize with normal mpsa
        self.discretize(g, data, **kwargs)

        stress, bound_stress = (
            matrix_dictionary["stress"],
            matrix_dictionary["bound_stress"],
        )

        # Create A and rhs
        div = pp.fvutils.vector_divergence(g)
        a = div * stress
        b = div * bound_stress

        # we find the matrix indices of the fracture
        if faces is None:
            frac_faces = g.frac_pairs
            frac_faces_left = frac_faces[0]
            frac_faces_right = frac_faces[1]
        else:
            raise NotImplementedError("not implemented given faces")

        int_b_left = pp.utils.mcolon.mcolon(
            g.dim * frac_faces_left, g.dim * frac_faces_left + g.dim
        )
        int_b_right = pp.utils.mcolon.mcolon(
            g.dim * frac_faces_right, g.dim * frac_faces_right + g.dim
        )
        int_b_ind = np.ravel((int_b_left, int_b_right), "C")

        # We find the sign of the left and right faces.
        sgn_left = _sign_matrix(g, frac_faces_left)
        sgn_right = _sign_matrix(g, frac_faces_right)
        # The displacement on the internal boundary face are considered unknowns,
        # so we move them over to the lhs. The rhs now only consists of the
        # external boundary faces
        b_internal = b[:, int_b_ind]
        b_external = b.copy()
        pp.utils.sparse_mat.zero_columns(b_external, int_b_ind)

        bound_stress_external = bound_stress.copy().tocsc()
        pp.utils.sparse_mat.zero_columns(bound_stress_external, int_b_ind)
        # We assume that the traction on the left hand side is equal but
        # opisite

        frac_stress_diff = (
            sgn_left * bound_stress[int_b_left, :]
            + sgn_right * bound_stress[int_b_right, :]
        )[:, int_b_ind]
        internal_stress = sps.hstack(
            (
                sgn_left * stress[int_b_left, :] + sgn_right * stress[int_b_right, :],
                frac_stress_diff,
            )
        )

        A = sps.vstack((sps.hstack((a, b_internal)), internal_stress), format="csr")
        # negative sign since we have moved b_external from lhs to rhs
        d_b = -b_external
        # sps.csr_matrix((int_b_left.size, g.num_faces * g.dim))
        d_t = (
            -sgn_left * bound_stress_external[int_b_left]
            - sgn_right * bound_stress_external[int_b_right]
        )

        b_matrix = sps.vstack((d_b, d_t), format="csr")

        matrix_dictionary["b_e"] = b_matrix
        matrix_dictionary["A_e"] = A

    def given_traction(self, g, stress, bound_stress, faces=None, **kwargs):
        # we find the matrix indices of the fracture
        if faces is None:
            frac_faces = g.frac_pairs
            frac_faces_left = frac_faces[0]
            frac_faces_right = frac_faces[1]
        else:
            raise NotImplementedError("not implemented given faces")

        int_b_left = pp.utils.mcolon.mcolon(
            g.dim * frac_faces_left, g.dim * frac_faces_left + g.dim
        )
        int_b_right = pp.utils.mcolon.mcolon(
            g.dim * frac_faces_right, g.dim * frac_faces_right + g.dim
        )
        int_b_ind = np.ravel((int_b_left, int_b_right), "C")

        # We find the sign of the left and right faces.
        sgn_left = _sign_matrix(g, frac_faces_left)
        sgn_right = _sign_matrix(g, frac_faces_right)

        # We obtain the stress from boundary conditions on the domain boundary
        bound_stress_external = bound_stress.copy().tocsc()
        pp.utils.sparse_mat.zero_columns(bound_stress_external, int_b_ind)
        bound_stress_external = bound_stress_external.tocsc()

        # We construct the L matrix, i.e., we set the traction on the left
        # fracture side
        frac_stress = (sgn_left * bound_stress[int_b_left, :])[:, int_b_ind]

        L = sps.hstack((sgn_left * stress[int_b_left, :], frac_stress))

        # negative sign since we have moved b_external from lhs to rhs
        d_t = (
            sps.csr_matrix(
                (np.ones(int_b_left.size), (np.arange(int_b_left.size), int_b_left)),
                (int_b_left.size, g.num_faces * g.dim),
            )
            - sgn_left * bound_stress_external[int_b_left]
        )  # \
        #        + sgn_right * bound_stress_external[int_b_right]

        return L, d_t

    def given_slip_distance(self, g, stress, bound_stress, faces=None):
        # we find the matrix indices of the fracture
        if faces is None:
            frac_faces = g.frac_pairs
            frac_faces_left = frac_faces[0]
            frac_faces_right = frac_faces[1]
        else:
            raise NotImplementedError("not implemented given faces")

        int_b_left = pp.utils.mcolon.mcolon(
            g.dim * frac_faces_left, g.dim * frac_faces_left + g.dim
        )
        int_b_right = pp.utils.mcolon.mcolon(
            g.dim * frac_faces_right, g.dim * frac_faces_right + g.dim
        )
        int_b_ind = np.ravel((int_b_left, int_b_right), "C")

        # We construct the L matrix, by assuming that the relative displacement
        # is given
        L = sps.hstack(
            (
                sps.csr_matrix((int_b_left.size, g.dim * g.num_cells)),
                sps.identity(int_b_left.size),
                -sps.identity(int_b_right.size),
            )
        )

        d_f = sps.csr_matrix(
            (np.ones(int_b_left.size), (np.arange(int_b_left.size), int_b_left)),
            (int_b_left.size, g.num_faces * g.dim),
        )

        return L, d_f


# ------------------------------------------------------------------------------#


def mpsa(
    g,
    constit,
    bound,
    eta=None,
    inverter=None,
    max_memory=None,
    hf_disp=False,
    hf_eta=None,
    **kwargs
):
    """
    Discretize the vector elliptic equation by the multi-point stress
    approximation method, specifically the weakly symmetric MPSA-W method.

    The method computes stresses over faces in terms of displacments in
    adjacent cells (defined as all cells sharing at least one vertex with the
    face).  This corresponds to the MPSA-W method, see

    Keilegavlen, Nordbotten: Finite volume methods for elasticity with weak
        symmetry. Int J Num. Meth. Eng. doi: 10.1002/nme.5538.

    Implementation needs:
        1) The local linear systems should be scaled with the elastic moduli
        and the local grid size, so that we avoid rounding errors accumulating
        under grid refinement / convergence tests.
        2) It should be possible to do a partial update of the discretization
        stensil (say, if we introduce an internal boundary, or modify the
        permeability field).
        3) For large grids, the current implementation will run into memory
        issues, due to the construction of a block diagonal matrix. This can be
        overcome by splitting the discretization into several partial updates.
        4) It probably makes sense to create a wrapper class to store the
        discretization, interface to linear solvers etc.
    Right now, there are concrete plans for 2) - 4).

    Parameters:
        g (core.grids.grid): grid to be discretized
        constit (pp.FourthOrderTensor) Constitutive law
        bound (pp.BoundarCondition) Class for boundary condition
        eta Location of pressure continuity point. Should be 1/3 for simplex
            grids, 0 otherwise. On boundary faces with Dirichlet conditions,
            eta=0 will be enforced.
        inverter (string) Block inverter to be used, either numba (default),
            cython or python. See fvutils.invert_diagonal_blocks for details.
        max_memory (double): Threshold for peak memory during discretization.
            If the **estimated** memory need is larger than the provided
            threshold, the discretization will be split into an appropriate
            number of sub-calculations, using mpsa_partial().
        hf_disp (bool) False: If true two matrices hf_cell, hf_bound is also returned such
            that hf_cell * U + hf_bound * u_bound gives the reconstructed displacement
            at the point on the face hf_eta. U is the cell centered displacement and
            u_bound the boundary conditions
        hf_eta (float) None: The point of displacment on the sub-faces. hf_eta=0 gives the
            displacement at the face centers while hf_eta=1 gives the displacements at
            the nodes. If None is given, the continuity points eta will be used.
    Returns:
        scipy.sparse.csr_matrix (shape num_faces, num_cells): stress
            discretization, in the form of mapping from cell displacement to
            face stresses.
            NOTE: The cell displacements are ordered cellwise (first u_x_1,
            u_y_1, u_x_2 etc)
        scipy.sparse.csr_matrix (shape num_faces, num_faces): discretization of
            boundary conditions. Interpreted as istresses induced by the boundary
            condition (both Dirichlet and Neumann). For Neumann, this will be
            the prescribed stress over the boundary face, and possibly stress
            on faces having nodes on the boundary. For Dirichlet, the values
            will be stresses induced by the prescribed displacement.
            Incorporation as a right hand side in linear system by
            multiplication with divergence operator.
            NOTE: The stresses are ordered facewise (first s_x_1, s_y_1 etc)
        If hf_disp is True the following will also be returned
        scipy.sparse.csr_matrix (g.dim*shape num_hfaces, g.dim*num_cells):
            displacement reconstruction for the displacement at the sub-faces. This is
            the contribution from the cell-center displacements.
            NOTE: The sub-face displacements are ordered cell wise
            (U_x_0, U_x_1, ..., U_x_n, U_y0, U_y1, ...)
        scipy.sparse.csr_matrix (g.dim*shape num_hfaces, g.dim*num_faces):
            displacement reconstruction for the displacement at the half faces.
            This is the contribution from the boundary conditions.
            NOTE: The half-face displacements are ordered cell wise
            (U_x_0, U_x_1, ..., U_x_n, U_y0, U_y1, ...)


    Example:
        # Set up a Cartesian grid
        g = structured.CartGrid([5, 5])
        c =tensor.FourthOrderTensor(g.dim, np.ones(g.num_cells))

        # Dirirchlet boundary conditions
        bound_faces = g.get_all_boundary_faces().ravel()
        bnd = bc.BoundaryCondition(g, bound_faces, ['dir'] * bound_faces.size)

        # Discretization
        stress, bound_stress = mpsa(g, c, bnd)

        # Source in the middle of the domain
        q = np.zeros(g.num_cells * g.dim)
        q[12 * g.dim] = 1

        # Divergence operator for the grid
        div = fvutils.vector_divergence(g)

        # Discretization matrix
        A = div * stress

        # Assign boundary values to all faces on the bounary
        bound_vals = np.zeros(g.num_faces * g.dim)
        bound_vals[bound_faces] = np.arange(bound_faces.size * g.dim)

        # Assemble the right hand side and solve
        rhs = -q - div * bound_stress * bound_vals
        x = sps.linalg.spsolve(A, rhs)
        s = stress * x + bound_stress * bound_vals

    """
    if bound.bc_type != "vectorial":
        raise AttributeError("MPSA must be given a vectorial boundary condition")

    if eta is None:
        eta = pp.fvutils.determine_eta(g)

    if max_memory is None:
        # For the moment nothing to do here, just call main mpfa method for the
        # entire grid.
        # TODO: We may want to estimate the memory need, and give a warning if
        # this seems excessive
        stress, bound_stress, hf_cell, hf_bound = _mpsa_local(
            g,
            constit,
            bound,
            eta=eta,
            inverter=inverter,
            hf_disp=hf_disp,
            hf_eta=hf_eta,
        )
    else:
        # Estimate number of partitions necessary based on prescribed memory
        # usage
        peak_mem = _estimate_peak_memory_mpsa(g)
        num_part = np.ceil(peak_mem / max_memory)

        logger.info("Split MPSA discretization into " + str(num_part) + " parts")

        # Let partitioning module apply the best available method
        part = pp.grid.partition.partition_metis(g, num_part)

        # Empty fields for stress and bound_stress. Will be expanded as we go.
        # Implementation note: It should be relatively straightforward to
        # estimate the memory need of stress (face_nodes -> node_cells ->
        # unique).
        stress = sps.csr_matrix((g.num_faces * g.dim, g.num_cells * g.dim))
        bound_stress = sps.csr_matrix((g.num_faces * g.dim, g.num_faces * g.dim))

        cn = g.cell_nodes()

        face_covered = np.zeros(g.num_faces, dtype=np.bool)

        for p in np.unique(part):
            # Cells in this partitioning
            cell_ind = np.argwhere(part == p).ravel("F")
            # To discretize with as little overlap as possible, we use the
            # keyword nodes to specify the update stencil. Find nodes of the
            # local cells.
            active_cells = np.zeros(g.num_cells, dtype=np.bool)
            active_cells[cell_ind] = 1
            active_nodes = np.squeeze(np.where((cn * active_cells) > 0))

            # Perform local discretization.
            loc_stress, loc_bound_stress, loc_faces = mpsa_partial(
                g, constit, bound, eta=eta, inverter=inverter, nodes=active_nodes
            )

            # Eliminate contribution from faces already covered
            eliminate_ind = pp.fvutils.expand_indices_nd(face_covered, g.dim)
            pp.fvutils.zero_out_sparse_rows(loc_stress, eliminate_ind)
            pp.fvutils.zero_out_sparse_rows(loc_bound_stress, eliminate_ind)

            face_covered[loc_faces] = 1

            stress += loc_stress
            bound_stress += loc_bound_stress

    return stress, bound_stress, hf_cell, hf_bound


def mpsa_update_partial(
    stress,
    bound_stress,
    hf_cell,
    hf_bound,
    g,
    constit,
    bound,
    eta=None,
    hf_eta=None,
    inverter="numba",
    cells=None,
    faces=None,
    nodes=None,
    apertures=None,
):
    """
    Given a discretization this function rediscretize parts of the domain.
    This is a fast way to update the discretization if you change, say the
    boundary conditions, have a growth of fractures, or a change in aperture.

    Parameters:
    stress (scipy.sparse.csr_matrix (shape num_faces, num_cells)): stress
            discretization to be updated. As returned from mpsa(...).
    bound_stress (scipy.sparse.csr_matrix (shape num_faces, num_faces)): bound stress
            discretization of boundary conditions as returned form mpsa(...).
    hf_cell (scipy.sparse.csr_matrix (g.dim*shape num_hfaces, g.dim*num_cells)):
            Sub-face displacement reconstruction at the sub faces.
    hf_bound (scipy.sparse.csr_matrix (g.dim*shape num_hfaces, g.dim*num_faces)):
            Sub-face displacement reconstruction at the sub faces.
    This is just a wrapper for the mpsa_partial(...), see this function for information
    abount the remainding parameters.

    returns:
    stress (scipy.sparse.csr_matrix (shape num_faces, num_cells)): stress
            discretization that has been updated for given cells, faces or nodes.
    bound_stress (scipy.sparse.csr_matrix (shape num_faces, num_faces)): bound stress
            discretization of boundary conditions that has been updated for given cells,
            faces or nodes.
    if hf_cell is not None:
    hf_cell (scipy.sparse.csr_matrix (g.dim*shape num_hfaces, g.dim*num_cells)):
        The matrix for reconstruction the displacement at the sub_faces that has been
        updated for the given cells, faces or nodes
    hf_bound (scipy.sparse.csr_matrix (g.dim*shape num_hfaces, g.dim*num_faces)):
        The matrix for reconstruction the displacement at the sub_faces that has been
        updated for the given cells, faces or nodes
    """
    stress = stress.copy()
    bound_stress = bound_stress.copy()
    hf_cell = hf_cell.copy()
    hf_bound = hf_bound.copy()

    stress_loc, bound_stress_loc, hf_cell_loc, hf_bound_loc, active_faces = mpsa_partial(
        g,
        constit,
        bound,
        eta,
        inverter,
        cells,
        faces,
        nodes=nodes,
        apertures=apertures,
        hf_disp=True,
        hf_eta=hf_eta,
    )

    # Remove old rows
    eliminate_ind = pp.fvutils.expand_indices_nd(active_faces, g.dim)
    pp.fvutils.zero_out_sparse_rows(stress, eliminate_ind)
    pp.fvutils.zero_out_sparse_rows(bound_stress, eliminate_ind)
    stress += stress_loc
    bound_stress += bound_stress_loc

    # We now update the values for the reconstruction of displacement.This is
    # equivalent to what is done for the stress and
    # bound_stress, but as we are working with subfaces some more care has to be
    # taken.
    # First, find the active subfaces associated with the active_faces
    subcell_topology = pp.fvutils.SubcellTopology(g)
    active_subfaces = np.where(np.in1d(subcell_topology.fno_unique, active_faces))[0]
    # We now expand the indices for each dimension.
    # The indices are ordered as first all variables of subface 1 then all variables
    # of subface 2, etc. Duplicate indices for each dimension and multipy by g.dim to
    # obtain correct x-index.
    sub_eliminate_ind = g.dim * np.tile(active_subfaces, (g.dim, 1))
    # Next add an increment to the y (and possible z) dimension to obtain correct index
    # For them
    sub_eliminate_ind += np.atleast_2d(np.arange(0, g.dim)).T
    sub_eliminate_ind = sub_eliminate_ind.ravel("F")
    # Zero out the faces we have updated
    pp.fvutils.zero_out_sparse_rows(hf_cell, sub_eliminate_ind)
    pp.fvutils.zero_out_sparse_rows(hf_bound, sub_eliminate_ind)
    # and add the update.
    hf_cell += hf_cell_loc
    hf_bound += hf_bound_loc
    return stress, bound_stress, hf_cell, hf_bound


def mpsa_partial(
    g,
    constit,
    bound,
    eta=None,
    inverter="numba",
    cells=None,
    faces=None,
    nodes=None,
    apertures=None,
    hf_disp=False,
    hf_eta=None,
):
    """
    Run an MPFA discretization on subgrid, and return discretization in terms
    of global variable numbers.

    Scenarios where the method will be used include updates of permeability,
    and the introduction of an internal boundary (e.g. fracture growth).

    The subgrid can be specified in terms of cells, faces and nodes to be
    updated. For details on the implementation, see
    fv_utils.cell_ind_for_partial_update()

    Parameters:
        g (porepy.grids.grid.Grid): grid to be discretized
        constit (porepy.params.tensor.SecondOrderTensor) permeability tensor
        bnd (porepy.params.bc.BoundaryCondition) class for boundary conditions
        faces (np.ndarray) faces to be considered. Intended for partial
            discretization, may change in the future
        eta Location of pressure continuity point. Should be 1/3 for simplex
            grids, 0 otherwise. On boundary faces with Dirichlet conditions,
            eta=0 will be enforced.
        inverter (string) Block inverter to be used, either numba (default),
            cython or python. See fvutils.invert_diagonal_blocks for details.
        cells (np.array, int, optional): Index of cells on which to base the
            subgrid computation. Defaults to None.
        faces (np.array, int, optional): Index of faces on which to base the
            subgrid computation. Defaults to None.
        nodes (np.array, int, optional): Index of nodes on which to base the
            subgrid computation. Defaults to None.
        apertures (np.array, int, optional): Cell apertures. Defaults to None.
            Unused for now, added for similarity to mpfa_partial.
        hf_disp (bool) False: If true two matrices hf_cell, hf_bound is also returned such
            that hf_cell * U + hf_bound * u_bound gives the reconstructed displacement
            at the point on the face hf_eta. U is the cell centered displacement and
            u_bound the boundary conditions
        hf_eta (float) None: The point of displacment on the half-faces. hf_eta=0 gives the
            displacement at the face centers while hf_eta=1 gives the displacements at
            the nodes. If None is given, the continuity points eta will be used.

        Note that if all of {cells, faces, nodes} are None, empty matrices will
        be returned.

    Returns:
        sps.csr_matrix (g.num_faces x g.num_cells): Stress discretization,
            computed on a subgrid.
        sps.csr_matrix (g,num_faces x g.num_faces): Boundary stress
            discretization, computed on a subgrid
        np.array (int): Global of the faces where the stress discretization is
            computed.
        If hf_disp is True the following will also be returned
        scipy.sparse.csr_matrix (g.dim*shape num_hfaces, g.dim*num_cells):
            displacement reconstruction for the displacement at the sub faces. This is
            the contribution from the cell-center displacements.
            NOTE: The half-face displacements are ordered sub_face wise
            (U_x_0, U_x_1, ..., U_x_n, U_y0, U_y1, ...)
        scipy.sparse.csr_matrix (g.dim*shape num_hfaces, g.dim*num_faces):
            displacement reconstruction for the displacement at the half faces.
            This is the contribution from the boundary conditions.
            NOTE: The half-face displacements are ordered sub_face wise
            (U_x_0, U_x_1, ..., U_x_n, U_y0, U_y1, ...)
    """
    if eta is None:
        eta = pp.fvutils.determine_eta(g)

    if cells is not None:
        warnings.warn("Cells keyword for partial mpfa has not been tested")
    if faces is not None:
        warnings.warn("Faces keyword for partial mpfa has not been tested")

    # Find computational stencil, based on specified cells, faces and nodes.
    ind, active_faces = pp.fvutils.cell_ind_for_partial_update(
        g, cells=cells, faces=faces, nodes=nodes
    )
    if (ind.size + active_faces.size) == 0:
        stress_glob = sps.csr_matrix(
            (g.dim * g.num_faces, g.dim * g.num_cells), dtype="float64"
        )
        bound_stress_glob = sps.csr_matrix(
            (g.dim * g.num_faces, g.dim * g.num_faces), dtype="float64"
        )
        return stress_glob, bound_stress_glob, active_faces
    # Extract subgrid, together with mappings between local and global
    # cells
    sub_g, l2g_faces, _ = pp.partition.extract_subgrid(g, ind)
    l2g_cells = sub_g.parent_cell_ind

    # Copy stiffness tensor, and restrict to local cells
    loc_c = constit.copy()
    loc_c.values = loc_c.values[::, ::, l2g_cells]
    # Also restrict the lambda and mu fields; we will copy the stiffness
    # tensors later.
    loc_c.lmbda = loc_c.lmbda[l2g_cells]
    loc_c.mu = loc_c.mu[l2g_cells]

    # Boundary conditions are slightly more complex. Find local faces
    # that are on the global boundary.
    # Then transfer boundary condition on those faces.

    loc_bnd = pp.BoundaryConditionVectorial(sub_g)
    loc_bnd.is_dir = bound.is_dir[:, l2g_faces]
    loc_bnd.is_rob = bound.is_rob[:, l2g_faces]
    loc_bnd.is_neu[loc_bnd.is_dir + loc_bnd.is_rob] = False

    # Discretization of sub-problem
    stress_loc, bound_stress_loc, hf_cell_loc, hf_bound_loc = _mpsa_local(
        sub_g,
        loc_c,
        loc_bnd,
        eta=eta,
        inverter=inverter,
        hf_disp=hf_disp,
        hf_eta=hf_eta,
    )

    face_map, cell_map = pp.fvutils.map_subgrid_to_grid(
        g, l2g_faces, l2g_cells, is_vector=True
    )
    # Update global face fields.
    stress_glob = face_map * stress_loc * cell_map
    bound_stress_glob = face_map * bound_stress_loc * face_map.transpose()

    # By design of mpfa, and the subgrids, the discretization will update faces
    # outside the active faces. Kill these.
    outside = np.setdiff1d(np.arange(g.num_faces), active_faces, assume_unique=True)
    eliminate_ind = pp.fvutils.expand_indices_nd(outside, g.dim)
    pp.fvutils.zero_out_sparse_rows(stress_glob, eliminate_ind)
    pp.fvutils.zero_out_sparse_rows(bound_stress_glob, eliminate_ind)

    # If we are returning the subface displacement reconstruction matrices we have
    # to do some more work. The following is equivalent to what is done for the stresses,
    # but as they are working on faces, the displacement reconstruction has to work on
    # subfaces.
    # First, we find the mappings from local subfaces to global subfaces
    subcell_topology = pp.fvutils.SubcellTopology(g)
    l2g_sub_faces = np.where(np.in1d(subcell_topology.fno_unique, l2g_faces))[0]
    # We now create a fake grid, just to be able to use the function map_subgrid_to_grid.
    subgrid = pp.CartGrid([1] * g.dim)
    subgrid.num_faces = subcell_topology.fno_unique.size
    subgrid.num_cells = g.num_cells
    sub_face_map, _ = pp.fvutils.map_subgrid_to_grid(
        subgrid, l2g_sub_faces, l2g_cells, is_vector=True
    )
    # The sub_face_map is now a map from local sub_faces to global subfaces.
    # Next we need to mat the the local sub face reconstruction "hf_cell_loc"
    # onto the global grid. The cells are ordered the same, so we can use the
    # cell_map from the stress computation. Similarly for the faces.
    hf_cell_glob = sub_face_map * hf_cell_loc * cell_map
    hf_bound_glob = sub_face_map * hf_bound_loc * face_map.T
    # Next we need to eliminate the subfaces outside the active faces.
    # We map from outside faces to outside subfaces
    sub_outside = np.where(np.in1d(subcell_topology.fno_unique, outside))[0]
    # Then expand the indices.
    # The indices are ordered as first all variables of subface 1 then all variables
    # of subface 2, etc. Duplicate indices for each dimension and multipy by g.dim to
    # obtain correct x-index.
    sub_eliminate_ind = g.dim * np.tile(sub_outside, (g.dim, 1))
    # Next add an increment to the y (and possible z) dimension to obtain correct index
    # For them
    sub_eliminate_ind += np.atleast_2d(np.arange(0, g.dim)).T
    sub_eliminate_ind = sub_eliminate_ind.ravel("F")
    # now kill the contribution of these faces
    pp.fvutils.zero_out_sparse_rows(hf_cell_glob, sub_eliminate_ind)
    pp.fvutils.zero_out_sparse_rows(hf_bound_glob, sub_eliminate_ind)
    return stress_glob, bound_stress_glob, hf_cell_glob, hf_bound_glob, active_faces


def _mpsa_local(
    g, constit, bound, eta=None, inverter="numba", hf_disp=False, hf_eta=None
):
    """
    Actual implementation of the MPSA W-method. To calculate the MPSA
    discretization on a grid, either call this method, or, to respect the
    privacy of this method, call the main mpsa method with no memory
    constraints.

    Implementation details:

    The displacement is discretized as a linear function on sub-cells (see
    reference paper). In this implementation, the displacement is represented by
    its cell center value and the sub-cell gradients.

    The method will give continuous stresses over the faces, and displacement
    continuity for certain points (controlled by the parameter eta). This can
    be expressed as a linear system on the form

        (i)   A * grad_u            = 0
        (ii)  B * grad_u + C * u_cc = 0
        (iii) 0            D * u_cc = I

    Here, the first equation represents stress continuity, and involves only
    the displacement gradients (grad_u). The second equation gives displacement
    continuity over cell faces, thus B will contain distances between cell
    centers and the face continuity points, while C consists of +- 1 (depending
    on which side the cell is relative to the face normal vector). The third
    equation enforces the displacement to be unity in one cell at a time. Thus
    (i)-(iii) can be inverted to express the displacement gradients as in terms
    of the cell center variables, that is, we can compute the basis functions
    on the sub-cells. Because of the method construction (again see reference
    paper), the basis function of a cell c will be non-zero on all sub-cells
    sharing a vertex with c. Finally, the fluxes as functions of cell center
    values are computed by insertion into Hook's law (which is essentially half
    of A from (i), that is, only consider contribution from one side of the
    face.

    Boundary values can be incorporated with appropriate modifications -
    Neumann conditions will have a non-zero right hand side for (i), Robin conditions
    will be on the form E * grad_u + F * u_cc = R, while
    Dirichlet gives a right hand side for (ii).

    In the implementation we will order the rows of the local system as follows;
    first enforce the force balance over the internal faces;
    T^L + T^R = 0.
    Then we will enforce the Neumann conditions
    T = T_NEUMAN,
    and the robin conditions
    T + robin_weight * U = R.
    The displacement continuity and Dirichlet conditions are comming last
    U^+ - U^- = 0.
    Note that for the Dirichlet conditions are not pulled out seperatly as the
    Neumann condition, mainly for legacy reasons. This meens that Dirichlet
    faces and internal faces are mixed together, decided by their face ordering.
    """
    if eta is None:
        eta = pp.fvutils.determine_eta(g)

    if bound.bc_type != "vectorial":
        raise AttributeError("MPSA must be given a vectorial boundary condition")
    # The grid coordinates are always three-dimensional, even if the grid is
    # really 2D. This means that there is not a 1-1 relation between the number
    # of coordinates of a point / vector and the real dimension. This again
    # violates some assumptions tacitly made in the discretization (in
    # particular that the number of faces of a cell that meets in a vertex
    # equals the grid dimension, and that this can be used to construct an
    # index of local variables in the discretization). These issues should be
    # possible to overcome, but for the moment, we simply force 2D grids to be
    # proper 2D.
    if g.dim == 2:
        g = g.copy()
        g.cell_centers = np.delete(g.cell_centers, (2), axis=0)
        g.face_centers = np.delete(g.face_centers, (2), axis=0)
        g.face_normals = np.delete(g.face_normals, (2), axis=0)
        g.nodes = np.delete(g.nodes, (2), axis=0)

        constit = constit.copy()
        constit.values = np.delete(constit.values, (2, 5, 6, 7, 8), axis=0)
        constit.values = np.delete(constit.values, (2, 5, 6, 7, 8), axis=1)

    nd = g.dim

    # Define subcell topology
    subcell_topology = pp.fvutils.SubcellTopology(g)
    # If g is not already a sub-grid we create one
    if bound.num_faces == subcell_topology.num_subfno_unique:
        subface_rhs = True
    else:
        # And we expand the boundary conditions to fit the sub-grid
        bound = pp.fvutils.boundary_to_sub_boundary(bound, subcell_topology)
        subface_rhs = False
    # Obtain mappings to exclude boundary faces
    bound_exclusion = pp.fvutils.ExcludeBoundaries(subcell_topology, bound, nd)
    # Most of the work is done by submethod for elasticity (which is common for
    # elasticity and poro-elasticity).

    hook, igrad, rhs_cells, _, _ = mpsa_elasticity(
        g, constit, subcell_topology, bound_exclusion, eta, inverter
    )

    hook_igrad = hook * igrad
    # NOTE: This is the point where we expect to reach peak memory need.
    del hook
    # Output should be on face-level (not sub-face)
    hf2f = pp.fvutils.map_hf_2_f(
        subcell_topology.fno_unique, subcell_topology.subfno_unique, nd
    )

    # Stress discretization
    stress = hook_igrad * rhs_cells
    # Right hand side for boundary discretization
    rhs_bound = create_bound_rhs(
        bound, bound_exclusion, subcell_topology, g, subface_rhs
    )
    # Discretization of boundary values
    bound_stress = hook_igrad * rhs_bound

    if not subface_rhs:
        bound_stress = hf2f * bound_stress * hf2f.T
        stress = hf2f * stress

    # Calculate the reconstruction of dispacement at faces
    if hf_eta is None:
        hf_eta = eta
    # We obtain the reconstruction of displacments
    dist_grad, cell_centers = reconstruct_displacement(g, subcell_topology, hf_eta)

    hf_cell = dist_grad * igrad * rhs_cells + cell_centers
    hf_bound = dist_grad * igrad * rhs_bound

    # The subface displacement is given by
    # hf_cell * u_cell_centers + hf_bound * u_bound_condition
    if not subface_rhs:
        hf_bound *= hf2f.T
    return stress, bound_stress, hf_cell, hf_bound


def mpsa_elasticity(g, constit, subcell_topology, bound_exclusion, eta, inverter):
    """
    This is the function where the real discretization takes place. It contains
    the parts that are common for elasticity and poro-elasticity, and was thus
    separated out as a helper function.

    The steps in the discretization are the same as in mpfa (although with
    everything being somewhat more complex since this is a vector equation).
    The mpfa function is currently more clean, so confer that for additional
    comments.

    Parameters:
        g: Grid
        constit: Constitutive law
        subcell_topology: Wrapper class for numbering of subcell faces, cells
            etc.
        bound_exclusion: Object that can eliminate faces related to boundary
            conditions.
        eta: Parameter determining the continuity point
        inverter: Parameter determining which method to use for inverting the
            local systems

    Returns:
        hook: Hooks law, ready to be multiplied with inverse gradients
        igrad: Inverse gradients
        rhs_cells: Right hand side used to get basis functions in terms of cell
            center displacements
        cell_node_blocks: Relation between cells and vertexes, used to group
            equations in linear system.
        hook_normal: Hooks law for the term div(I*p) in poro-elasticity
    """
    if bound_exclusion.bc_type != "vectorial":
        raise AttributeError("MPSA must be given a vectorial boundary condition")
    nd = g.dim

    # Compute product between normal vectors and stiffness matrices
    ncsym_all, ncasym, cell_node_blocks, sub_cell_index = _tensor_vector_prod(
        g, constit, subcell_topology
    )

    # Prepare for computation of forces due to cell center pressures (the term
    # div(I*p) in poro-elasticity equations. hook_normal will be used as a right
    # hand side by the biot disretization, but needs to be computed here, since
    # this is where we have access to the relevant data.
    ind_f = np.argsort(np.tile(subcell_topology.subhfno, nd), kind="mergesort")
    hook_normal = sps.coo_matrix(
        (np.ones(ind_f.size), (np.arange(ind_f.size), ind_f)),
        shape=(ind_f.size, ind_f.size),
    ) * (ncsym_all + ncasym)

    del ind_f

    # To avoid singular matrices we are not abe to add the asymetric part of the stress
    # tensor to the Neumann and Robin boundaries for nodes that only has more
    # Neumann-boundary faces than gradients. This will typically happen in the
    # corners where you only can have one gradient for the node. Normally if you
    # have at least one internal face connected to the node you are should be safe.
    # For the Neumann faces we eliminate the asymetic part this does in fact
    # lead to an inconsistency.
    _eliminate_ncasym_neumann(
        ncasym, subcell_topology, bound_exclusion, cell_node_blocks, nd
    )

    # The final expression of Hook's law will involve deformation gradients
    # on one side of the faces only; eliminate the other one.
    # Note that this must be done before we can pair forces from the two
    # sides of the faces.
    hook = __unique_hooks_law(ncsym_all, ncasym, subcell_topology, nd)

    # For the Robin boundary conditions we need to pair the forces with the
    # displacement.
    # The contribution of the displacement at the Robin boundary is
    # rob_grad * G + rob_cell * u (this is the displacement at the boundary scaled
    # with the Robin weight times the area of the faces),
    # where G are the gradients and u the cell center displacement. The Robin
    # condtion is then ncsym_rob * G + rob_grad * G + rob_cell * u
    ncsym_full = subcell_topology.pair_over_subfaces_nd(ncsym_all + ncasym)
    del ncasym
    ncsym_rob = bound_exclusion.keep_robin(ncsym_full)
    ncsym_neu = bound_exclusion.keep_neumann(ncsym_full)

    # Book keeping
    num_sub_cells = cell_node_blocks[0].size
    rob_grad, rob_cell = __get_displacement_submatrices_rob(
        g, subcell_topology, eta, num_sub_cells, bound_exclusion
    )

    # Pair the forces from each side.
    # ncsym * G is in fact (due to pair_over_subfaces)
    # ncsym_L * G_L + ncsym_R * G_R for the left and right faces.
    # We are here using ncsym_all and not the full tensor
    # ncsym_full = ncsym_all + ncasym. This is because
    # ncasym_L * G_L + ncasym_R * G_R = 0 due to symmetry.
    ncsym = subcell_topology.pair_over_subfaces_nd(ncsym_all)
    del ncsym_all
    # Boundary conditions are taken hand of in ncsym_rob, ncsym_neu or as Dirichlet.
    ncsym = bound_exclusion.exclude_boundary(ncsym)

    # The contribution of cell center displacement to stress continuity.
    # This is just zero (T_L + T_R = 0).
    num_subfno = subcell_topology.subfno.max() + 1
    hook_cell = sps.coo_matrix(
        (np.zeros(1), (np.zeros(1), np.zeros(1))),
        shape=(num_subfno * nd, (np.max(subcell_topology.cno) + 1) * nd),
    ).tocsr()
    # Here you have to be carefull if you ever change hook_cell to something else than
    # 0. Because we have pulled the Neumann conditions out of the stress condition
    # the following would give an index error. Instead you would have to make a
    # hook_cell_neu equal the number neumann_sub_faces, and a hook_cell_int equal the number
    # of internal sub_faces and use .keep_neu and .exclude_bnd. But since this is all zeros,
    # thi indexing does not matter.
    hook_cell = bound_exclusion.exclude_robin_dirichlet(hook_cell)

    # Matrices to enforce displacement continuity
    d_cont_grad, d_cont_cell = __get_displacement_submatrices(
        g, subcell_topology, eta, num_sub_cells, bound_exclusion
    )

    grad_eqs = sps.vstack([ncsym, ncsym_neu, ncsym_rob + rob_grad, d_cont_grad])

    del ncsym, d_cont_grad, ncsym_rob, rob_grad, ncsym_neu
    igrad = _inverse_gradient(
        grad_eqs,
        sub_cell_index,
        cell_node_blocks,
        subcell_topology.nno_unique,
        bound_exclusion,
        nd,
        inverter,
    )

    # Right hand side for cell center variables
    rhs_cells = -sps.vstack([hook_cell, rob_cell, d_cont_cell])
    return hook, igrad, rhs_cells, cell_node_blocks, hook_normal


def reconstruct_displacement(g, subcell_topology, eta=None):
    """
    Function for reconstructing the displacement at the half faces given the
    local gradients. For a subcell Ks associated with cell K and node s, the
    displacement at a point x is given by
    U_Ks + G_Ks (x - x_k),
    x_K is the cell center of cell k. The point at which we evaluate the displacement
    is given by eta, which is equivalent to the continuity points in mpsa.
    For an internal subface we will obtain two values for the displacement,
    one for each of the cells associated with the subface. The displacement given
    here is the average of the two. Note that at the continuity points the two
    displacements will by construction be equal.

    Parameters:
    Parameters:
        g: Grid
        subcell_topology: Wrapper class for numbering of subcell faces, cells
            etc.
        eta (float or ndarray, range=[0,1)): Optional. Parameter determining the point
            at which the displacement is evaluated. If eta is a nd-array it should be on
            the size of subcell_topology.num_subfno. If eta is not given the method will
            call fvutils.determine_eta(g) to set it.
    Returns:
        scipy.sparse.csr_matrix (g.dim*num_sub_faces, g.dim*num_cells):
            displacement reconstruction for the displacement at the half faces. This is
            the contribution from the cell-center displacements.
            NOTE: The half-face displacements are ordered sub-face_wise
            (U_x_0, U_x_1, ..., U_x_n, U_y0, U_y1, ...)
        scipy.sparse.csr_matrix (g.dim*num_sub_faces, g.dim*num_faces):
            displacement reconstruction for the displacement at the half faces.
            This is the contribution from the boundary conditions.
            NOTE: The half-face displacements are ordered sub_face wise
            (U_x_0, U_x_1, ..., U_x_n, U_y0, U_y1, ...)
    """
    if eta is None:
        eta = pp.fvutils.determine_eta(g)

    # Calculate the distance from the cell centers to continuity points
    D_g = pp.fvutils.compute_dist_face_cell(
        g, subcell_topology, eta, return_paired=False
    )
    # We here average the contribution on internal sub-faces.
    # If you want to get out both displacements on a sub-face your can remove
    # the averaging.
    _, IC, counts = np.unique(
        subcell_topology.subfno, return_inverse=True, return_counts=True
    )

    avg_over_subfaces = sps.coo_matrix(
        (1 / counts[IC], (subcell_topology.subfno, subcell_topology.subhfno))
    )
    D_g = avg_over_subfaces * D_g
    # expand indices to x-y-z
    D_g = sps.kron(sps.eye(g.dim), D_g)
    D_g = D_g.tocsr()

    # Get a mapping from cell centers to half-faces
    D_c = sps.coo_matrix(
        (1 / counts[IC], (subcell_topology.subfno, subcell_topology.cno))
    ).tocsr()
    # Expand indices to x-y-z
    D_c = sps.kron(sps.eye(g.dim), D_c)
    D_c = D_c.tocsc()
    # book keeping
    cell_node_blocks, _ = pp.utils.matrix_compression.rlencode(
        np.vstack((subcell_topology.cno, subcell_topology.nno))
    )
    num_sub_cells = cell_node_blocks[0].size
    # The column ordering of the displacement equilibrium equations are
    # formed as a Kronecker product of scalar equations. Bring them to the
    # same form as that applied in the force balance equations
    dist_grad, cell_centers = __rearange_columns_displacement_eqs(
        D_g, D_c, num_sub_cells, g.dim
    )
    # The row ordering is now first variable x of all subfaces then
    # variable y of all subfaces, etc. Change the ordering to first all variables
    # of first cell, then all variables of second cell, etc.
    P = row_major_to_col_major(cell_centers.shape, g.dim, 0)
    return P * dist_grad, P * cell_centers


# -----------------------------------------------------------------------------
#
# Below here are helper functions, which tend to be less than well documented.
#
# -----------------------------------------------------------------------------


def _estimate_peak_memory_mpsa(g):
    """ Rough estimate of peak memory need for mpsa discretization.
    """
    nd = g.dim
    num_cell_nodes = g.cell_nodes().sum(axis=1).A

    # Number of unknowns around a vertex: nd^2 per cell that share the vertex
    # for pressure gradients, and one per cell (cell center pressure)
    num_grad_unknowns = nd ** 2 * num_cell_nodes

    # The most expensive field is the storage of igrad, which is block diagonal
    # with num_grad_unknowns sized blocks. The number of elements is the square
    # of the local system size. The factor 2 accounts for matrix storage in
    # sparse format (rows and data; ignore columns since this is in compressed
    # format)
    igrad_size = np.power(num_grad_unknowns, 2).sum() * 2

    # The discretization of Hook's law will require nd^2 (that is, a gradient)
    # per sub-face per dimension
    num_sub_face = g.face_nodes.sum()
    hook_size = nd * num_sub_face * nd ** 2

    # Balancing of stresses will require 2*nd**2 (gradient on both sides)
    # fields per sub-face per dimension
    nk_grad_size = 2 * nd * num_sub_face * nd ** 2
    # Similarly, pressure continuity requires 2 * (nd+1) (gradient on both
    # sides, and cell center pressures) numbers
    pr_cont_size = 2 * (nd ** 2 + 1) * num_sub_face * nd

    total_size = igrad_size + hook_size + nk_grad_size + pr_cont_size

    # Not covered yet is various fields on subcell topology, mapping matrices
    # between local and block ordering etc.
    return total_size


def __get_displacement_submatrices(
    g, subcell_topology, eta, num_sub_cells, bound_exclusion
):
    nd = g.dim
    # Distance from cell centers to face centers, this will be the
    # contribution from gradient unknown to equations for displacement
    # continuity
    d_cont_grad = pp.fvutils.compute_dist_face_cell(g, subcell_topology, eta)

    # For force balance, displacements and stresses on the two sides of the
    # matrices must be paired
    d_cont_grad = sps.kron(sps.eye(nd), d_cont_grad)

    # Contribution from cell center potentials to local systems
    d_cont_cell = __cell_variable_contribution(g, subcell_topology)

    # Expand equations for displacement balance, and eliminate rows
    # associated with neumann boundary conditions
    d_cont_grad = bound_exclusion.exclude_neumann_robin(d_cont_grad)
    d_cont_cell = bound_exclusion.exclude_neumann_robin(d_cont_cell)

    # The column ordering of the displacement equilibrium equations are
    # formed as a Kronecker product of scalar equations. Bring them to the
    # same form as that applied in the force balance equations
    d_cont_grad, d_cont_cell = __rearange_columns_displacement_eqs(
        d_cont_grad, d_cont_cell, num_sub_cells, nd
    )

    return d_cont_grad, d_cont_cell


def __get_displacement_submatrices_rob(
    g, subcell_topology, eta, num_sub_cells, bound_exclusion
):
    nd = g.dim
    # Distance from cell centers to face centers, this will be the
    # contribution from gradient unknown to equations for displacement
    # at the boundary
    rob_grad = pp.fvutils.compute_dist_face_cell(g, subcell_topology, eta)

    # For the Robin condition the distance from the cell centers to face centers
    # will be the contribution from the gradients. We integrate over the subface
    # and multiply by the area
    num_nodes = np.diff(g.face_nodes.indptr)
    sgn = g.cell_faces[subcell_topology.fno_unique, subcell_topology.cno_unique].A
    scaled_sgn = (
        sgn[0]
        * g.face_areas[subcell_topology.fno_unique]
        / num_nodes[subcell_topology.fno_unique]
    )
    # pair_over_subfaces flips the sign so we flip it back
    rob_grad = sps.kron(sps.eye(nd), sps.diags(scaled_sgn) * rob_grad)
    # Contribution from cell center potentials to local systems
    rob_cell = sps.coo_matrix(
        (
            g.face_areas[subcell_topology.fno] / num_nodes[subcell_topology.fno],
            (subcell_topology.subfno, subcell_topology.cno),
        )
    ).tocsr()
    rob_cell = sps.kron(sps.eye(nd), rob_cell)

    # First do a basis transformation
    rob_grad = bound_exclusion.basis_matrix * rob_grad
    rob_cell = bound_exclusion.basis_matrix * rob_cell
    # And apply the robin weight in the rotated basis
    rob_grad = bound_exclusion.robin_weight * rob_grad
    rob_cell = bound_exclusion.robin_weight * rob_cell
    # Expand equations for displacement balance, and keep rows
    # associated with neumann boundary conditions. Remember we have already
    # rotated the basis above
    rob_grad = bound_exclusion.keep_robin(rob_grad, transform=False)
    rob_cell = bound_exclusion.keep_robin(rob_cell, transform=False)

    # The column ordering of the displacement equilibrium equations are
    # formed as a Kronecker product of scalar equations. Bring them to the
    # same form as that applied in the force balance equations
    rob_grad, rob_cell = __rearange_columns_displacement_eqs(
        rob_grad, rob_cell, num_sub_cells, nd
    )
    return rob_grad, rob_cell


def _split_stiffness_matrix(constit):
    """
    Split the stiffness matrix into symmetric and asymetric part

    Parameters
    ----------
    constit stiffness tensor

    Returns
    -------
    csym part of stiffness tensor that enters the local calculation
    casym part of stiffness matrix not included in local calculation
    """
    dim = np.sqrt(constit.values.shape[0])

    # We do not know how constit is used outside the discretization,
    # so create deep copies to avoid overwriting. Not really sure if this is
    # necessary
    csym = 0 * constit.copy().values
    casym = constit.copy().values

    # The copy constructor for the stiffness matrix will represent all
    # dimensions as 3d. If dim==2, delete the redundant rows and columns
    if dim == 2 and csym.shape[0] == 9:
        csym = np.delete(csym, (2, 5, 6, 7, 8), axis=0)
        csym = np.delete(csym, (2, 5, 6, 7, 8), axis=1)
        casym = np.delete(casym, (2, 5, 6, 7, 8), axis=0)
        casym = np.delete(casym, (2, 5, 6, 7, 8), axis=1)

    # The splitting is hard coded based on the ordering of elements in the
    # stiffness matrix
    if dim == 2:
        csym[0, 0] = casym[0, 0]
        csym[1, 1] = casym[1, 1]
        csym[2, 2] = casym[2, 2]
        csym[3, 0] = casym[3, 0]
        csym[0, 3] = casym[0, 3]
        csym[3, 3] = casym[3, 3]
    else:  # dim == 3
        csym[0, 0] = casym[0, 0]
        csym[1, 1] = casym[1, 1]
        csym[2, 2] = casym[2, 2]
        csym[3, 3] = casym[3, 3]
        csym[4, 4] = casym[4, 4]
        csym[5, 5] = casym[5, 5]
        csym[6, 6] = casym[6, 6]
        csym[7, 7] = casym[7, 7]
        csym[8, 8] = casym[8, 8]

        csym[4, 0] = casym[4, 0]
        csym[8, 0] = casym[8, 0]
        csym[0, 4] = casym[0, 4]
        csym[8, 4] = casym[8, 4]
        csym[0, 8] = casym[0, 8]
        csym[4, 8] = casym[4, 8]
    # The asymmetric part is whatever is not in the symmetric part
    casym -= csym
    return csym, casym


def _tensor_vector_prod(g, constit, subcell_topology):
    """ Compute product between stiffness tensor and face normals.

    The method splits the stiffness matrix into a symmetric and asymmetric
    part, and computes the products with normal vectors for each. The method
    also provides a unique identification of sub-cells (in the form of pairs of
    cells and nodes), and a global numbering of subcell gradients.

    Parameters:
        g: grid
        constit: Stiffness matrix, in the form of a fourth order tensor.
        subcell_topology: Numberings of subcell quantities etc.

    Returns:
        ncsym, ncasym: Product with face normals for symmetric and asymmetric
            part of stiffness tensors. On the subcell level. In effect, these
            will be stresses on subfaces, as functions of the subcell gradients
            (to be computed somewhere else). The rows first represent stresses
            in the x-direction for all faces, then y direction etc.
        cell_nodes_blocks: Unique pairing of cell and node numbers for
            subcells. First row: Cell numbers, second node numbers. np.ndarray.
        grad_ind: Numbering scheme for subcell gradients - gives a global
            numbering for the gradients. One column per subcell, the rows gives
            the index for the individual components of the gradients.

    """

    # Stack cells and nodes, and remove duplicate rows. Since subcell_mapping
    # defines cno and nno (and others) working cell-wise, this will
    # correspond to a unique rows (Matlab-style) from what I understand.
    # This also means that the pairs in cell_node_blocks uniquely defines
    # subcells, and can be used to index gradients etc.
    cell_node_blocks, blocksz = pp.utils.matrix_compression.rlencode(
        np.vstack((subcell_topology.cno, subcell_topology.nno))
    )

    nd = g.dim

    # Duplicates in [cno, nno] corresponds to different faces meeting at the
    # same node. There should be exactly nd of these. This test will fail
    # for pyramids in 3D
    if not np.all(blocksz == nd):
        raise AssertionError()

    # Define row and column indices to be used for normal vector matrix
    # Rows are based on sub-face numbers.
    # Columns have nd elements for each sub-cell (to store a vector) and
    # is adjusted according to block sizes
    _, cn = np.meshgrid(subcell_topology.subhfno, np.arange(nd))
    sum_blocksz = np.cumsum(blocksz)
    cn += pp.utils.matrix_compression.rldecode(sum_blocksz - blocksz[0], blocksz)
    ind_ptr_n = np.hstack((np.arange(0, cn.size, nd), cn.size))

    # Distribute faces equally on the sub-faces, and store in a matrix
    num_nodes = np.diff(g.face_nodes.indptr)
    normals = g.face_normals[:, subcell_topology.fno] / num_nodes[subcell_topology.fno]
    normals_mat = sps.csr_matrix((normals.ravel("F"), cn.ravel("F"), ind_ptr_n))

    # Then row and columns for stiffness matrix. There are nd^2 elements in
    # the gradient operator, and so the structure is somewhat different from
    # the normal vectors
    _, cc = np.meshgrid(subcell_topology.subhfno, np.arange(nd ** 2))
    sum_blocksz = np.cumsum(blocksz ** 2)
    cc += pp.utils.matrix_compression.rldecode(sum_blocksz - blocksz[0] ** 2, blocksz)
    ind_ptr_c = np.hstack((np.arange(0, cc.size, nd ** 2), cc.size))

    # Splitt stiffness matrix into symmetric and anti-symmatric part
    sym_tensor, asym_tensor = _split_stiffness_matrix(constit)

    # Getting the right elements out of the constitutive laws was a bit
    # tricky, but the following code turned out to do the trick
    sym_tensor_swp = np.swapaxes(sym_tensor, 2, 0)
    asym_tensor_swp = np.swapaxes(asym_tensor, 2, 0)

    # The first dimension in csym and casym represent the contribution from
    # all dimensions to the stress in one dimension (in 2D, csym[0:2,:,
    # :] together gives stress in the x-direction etc.
    # Define index vector to access the right rows
    rind = np.arange(nd)

    # Empty matrices to initialize matrix-tensor products. Will be expanded
    # as we move on
    zr = np.zeros(0)
    ncsym = sps.coo_matrix((zr, (zr, zr)), shape=(0, cc.max() + 1)).tocsr()
    ncasym = sps.coo_matrix((zr, (zr, zr)), shape=(0, cc.max() + 1)).tocsr()

    # For the asymmetric part of the tensor, we will apply volume averaging.
    # Associate a volume with each sub-cell, and a node-volume as the sum of
    # all surrounding sub-cells
    num_cell_nodes = g.num_cell_nodes()
    cell_vol = g.cell_volumes / num_cell_nodes
    node_vol = (
        np.bincount(subcell_topology.nno, weights=cell_vol[subcell_topology.cno])
        / g.dim
    )

    num_elem = cell_node_blocks.shape[1]
    map_mat = sps.coo_matrix(
        (np.ones(num_elem), (np.arange(num_elem), cell_node_blocks[1]))
    )
    weight_mat = sps.coo_matrix(
        (
            cell_vol[cell_node_blocks[0]] / node_vol[cell_node_blocks[1]],
            (cell_node_blocks[1], np.arange(num_elem)),
        )
    )
    # Operator for carying out the average
    average = sps.kron(map_mat * weight_mat, sps.identity(nd)).tocsr()

    for iter1 in range(nd):
        # Pick out part of Hook's law associated with this dimension
        # The code here looks nasty, it should be possible to get the right
        # format of the submatrices in a simpler way, but I couldn't do it.
        sym_dim = np.hstack(sym_tensor_swp[:, :, rind]).transpose()
        asym_dim = np.hstack(asym_tensor_swp[:, :, rind]).transpose()

        # Distribute (relevant parts of) Hook's law on subcells
        # This will be nd rows, thus cell ci is associated with indices
        # ci*nd+np.arange(nd)
        sub_cell_ind = pp.fvutils.expand_indices_nd(cell_node_blocks[0], nd)
        sym_vals = sym_dim[sub_cell_ind]
        asym_vals = asym_dim[sub_cell_ind]

        # Represent this part of the stiffness matrix in matrix form
        csym_mat = sps.csr_matrix((sym_vals.ravel("C"), cc.ravel("F"), ind_ptr_c))
        casym_mat = sps.csr_matrix((asym_vals.ravel("C"), cc.ravel("F"), ind_ptr_c))

        # Compute average around vertexes
        casym_mat = average * casym_mat

        # Compute products of normal vectors and stiffness tensors,
        # and stack dimensions vertically
        ncsym = sps.vstack((ncsym, normals_mat * csym_mat))
        ncasym = sps.vstack((ncasym, normals_mat * casym_mat))

        # Increase index vector, so that we get rows contributing to forces
        # in the next dimension
        rind += nd

    grad_ind = cc[:, ::nd]

    return ncsym, ncasym, cell_node_blocks, grad_ind


def _inverse_gradient(
    grad_eqs,
    sub_cell_index,
    cell_node_blocks,
    nno_unique,
    bound_exclusion,
    nd,
    inverter,
):
    # Mappings to convert linear system to block diagonal form
    rows2blk_diag, cols2blk_diag, size_of_blocks = _block_diagonal_structure(
        sub_cell_index, cell_node_blocks, nno_unique, bound_exclusion, nd
    )

    grad = rows2blk_diag * grad_eqs * cols2blk_diag
    # Compute inverse gradient operator, and map back again
    igrad = (
        cols2blk_diag
        * pp.fvutils.invert_diagonal_blocks(grad, size_of_blocks, method=inverter)
        * rows2blk_diag
    )
    print("max igrad: ", np.max(np.abs(igrad)))
    return igrad


def _block_diagonal_structure(
    sub_cell_index, cell_node_blocks, nno, bound_exclusion, nd
):
    """
    Define matrices to turn linear system into block-diagonal form.

    Parameters
    ----------
    sub_cell_index
    cell_node_blocks: pairs of cell and node pairs, which defines sub-cells
    nno node numbers associated with balance equations
    exclude_dirichlet mapping to remove rows associated with stress boundary
    exclude_neumann mapping to remove rows associated with displacement boundary

    Returns
    -------
    rows2blk_diag transform rows of linear system to block-diagonal form
    cols2blk_diag transform columns of linear system to block-diagonal form
    size_of_blocks number of equations in each block
    """

    # Stack node numbers of equations on top of each other, and sort them to
    # get block-structure. First eliminate node numbers at the boundary, where
    # the equations are either of flux or pressure continuity (not both)

    nno = np.tile(nno, nd)
    # The node numbers do not have a basis, so no transformation here. We just
    # want to pick out the correct node numbers for the correct equations.
    nno_stress = bound_exclusion.exclude_boundary(nno, transform=False)
    nno_displacement = bound_exclusion.exclude_neumann_robin(nno, transform=False)
    nno_neu = bound_exclusion.keep_neumann(nno, transform=False)
    nno_rob = bound_exclusion.keep_robin(nno, transform=False)
    node_occ = np.hstack((nno_stress, nno_neu, nno_rob, nno_displacement))

    sorted_ind = np.argsort(node_occ, kind="mergesort")
    rows2blk_diag = sps.coo_matrix(
        (np.ones(sorted_ind.size), (np.arange(sorted_ind.size), sorted_ind))
    ).tocsr()
    # Size of block systems
    sorted_nodes_rows = node_occ[sorted_ind]
    size_of_blocks = np.bincount(sorted_nodes_rows.astype("int64"))

    # cell_node_blocks[1] contains the node numbers associated with each
    # sub-cell gradient (and so column of the local linear systems). A sort
    # of these will give a block-diagonal structure
    sorted_nodes_cols = np.argsort(cell_node_blocks[1], kind="mergesort")
    subcind_nodes = sub_cell_index[::, sorted_nodes_cols].ravel("F")
    cols2blk_diag = sps.coo_matrix(
        (np.ones(sub_cell_index.size), (subcind_nodes, np.arange(sub_cell_index.size)))
    ).tocsr()
    return rows2blk_diag, cols2blk_diag, size_of_blocks


def create_bound_rhs(bound, bound_exclusion, subcell_topology, g, subface_rhs):
    """
    Define rhs matrix to get basis functions for boundary
    conditions assigned face-wise

    Parameters
    ----------
    bound
    bound_exclusion
    fno
    sgn : +-1, defining here and there of the faces
    g : grid
    num_stress : number of equations for flux continuity
    num_displ: number of equations for pressure continuity

    Returns
    -------
    rhs_bound: Matrix that can be multiplied with inverse block matrix to get
               basis functions for boundary values
    """
    nd = g.dim

    num_stress = bound_exclusion.exclude_bnd.shape[0]
    num_displ = bound_exclusion.exclude_neu_rob.shape[0]

    num_rob = bound_exclusion.keep_rob.shape[0]
    num_neu = bound_exclusion.keep_neu.shape[0]

    fno = subcell_topology.fno_unique
    subfno = subcell_topology.subfno_unique
    sgn = g.cell_faces[
        subcell_topology.fno_unique, subcell_topology.cno_unique
    ].A.ravel("F")

    num_dir = np.sum(bound.is_dir)
    if not num_rob == np.sum(bound.is_rob):
        raise AssertionError()
    if not num_neu == np.sum(bound.is_neu):
        raise AssertionError()

    num_bound = num_neu + num_dir + num_rob

    # Obtain the face number for each coordinate
    subfno_nd = np.tile(subfno, (nd, 1)) * nd + np.atleast_2d(np.arange(0, nd)).T

    # expand the indices
    # Define right hand side for Neumann boundary conditions
    # First row indices in rhs matrix
    # Pick out the subface indices
    # The boundary conditions should be given in the given basis, therefore no transformation
    subfno_neu = bound_exclusion.keep_neumann(
        subfno_nd.ravel("C"), transform=False
    ).ravel("F")
    # Pick out the Neumann boundary
    is_neu_nd = (
        bound_exclusion.keep_neumann(bound.is_neu.ravel("C"), transform=False)
        .ravel("F")
        .astype(np.bool)
    )

    neu_ind = np.argsort(subfno_neu)
    neu_ind = neu_ind[is_neu_nd[neu_ind]]

    # Robin, same procedure
    subfno_rob = bound_exclusion.keep_robin(
        subfno_nd.ravel("C"), transform=False
    ).ravel("F")

    is_rob_nd = (
        bound_exclusion.keep_robin(bound.is_rob.ravel("C"), transform=False)
        .ravel("F")
        .astype(np.bool)
    )

    rob_ind = np.argsort(subfno_rob)
    rob_ind = rob_ind[is_rob_nd[rob_ind]]

    # Dirichlet, same procedure
    # remove neumann and robin subfno
    subfno_dir = bound_exclusion.exclude_neumann_robin(
        subfno_nd.ravel("C"), transform=False
    ).ravel("F")
    is_dir_nd = (
        bound_exclusion.exclude_neumann_robin(bound.is_dir.ravel("C"), transform=False)
        .ravel("F")
        .astype(np.bool)
    )

    dir_ind = np.argsort(subfno_dir)
    dir_ind = dir_ind[is_dir_nd[dir_ind]]

    # We also need to account for all half faces, that is, do not exclude
    # Dirichlet and Neumann boundaries. This is the global indexing.
    is_neu_all = bound.is_neu.ravel("C")
    neu_ind_all = np.argwhere(
        np.reshape(is_neu_all, (nd, -1), order="C").ravel("F")
    ).ravel("F")
    is_dir_all = bound.is_dir.ravel("C")
    dir_ind_all = np.argwhere(
        np.reshape(is_dir_all, (nd, -1), order="C").ravel("F")
    ).ravel("F")

    is_rob_all = bound.is_rob.ravel("C")
    rob_ind_all = np.argwhere(
        np.reshape(is_rob_all, (nd, -1), order="C").ravel("F")
    ).ravel("F")

    # We now merge the neuman and robin indices since they are treated equivalent.
    # Remember that the first set of local equations are the stress equilibrium for
    # the internall faces. The number of internal stresses therefore has to
    # be added to the Neumann and Robin indices
    if rob_ind.size == 0:
        neu_rob_ind = neu_ind + num_stress
    elif neu_ind.size == 0:
        neu_rob_ind = rob_ind + num_stress
    else:
        neu_rob_ind = np.hstack((neu_ind + num_stress, rob_ind + num_stress + num_neu))

    neu_rob_ind_all = np.hstack((neu_ind_all, rob_ind_all))

    # stack together
    bnd_ind = np.hstack((neu_rob_ind_all, dir_ind_all))

    # Some care is needed to compute coefficients in Neumann matrix: sgn is
    # already defined according to the subcell topology [fno], while areas
    # must be drawn from the grid structure, and thus go through fno
    fno_ext = np.tile(fno, nd)
    num_face_nodes = g.face_nodes.sum(axis=0).A.ravel("F")

    # Coefficients in the matrix. For the Neumann boundary components we set the
    # value as seen from the outside of the domain. Note that they do not
    # have to do
    # so, and we will flip the sign later. This means that a stress [1,1] on a
    # boundary face pushes(or pulls) the face to the top right corner.
    # Note:
    if subface_rhs:
        # In this case we set the rhs for the sub-faces. Note that the rhs values
        # should be integrated over the subfaces, that is
        # stress_neumann *\cdot * normal * subface_area
        neu_val = 1 * np.ones(neu_rob_ind_all.size)
    else:
        # In this case we set the value at a face, thus, we need to distribute the
        #  face values to the subfaces. We do this by an area-weighted average. Note
        # that the rhs values should in this case be integrated over the faces, that is:
        # stress_neumann *\cdot * normal * face_area
        neu_val = 1 / num_face_nodes[fno_ext[neu_rob_ind_all]]

    # The columns will be 0:neu_rob_ind.size
    if neu_rob_ind.size > 0:
        neu_cell = sps.coo_matrix(
            (neu_val.ravel("F"), (neu_rob_ind, np.arange(neu_rob_ind.size))),
            shape=(num_stress + num_neu + num_rob, num_bound),
        ).tocsr()
    else:
        # Special handling when no elements are found. Not sure if this is
        # necessary, or if it is me being stupid
        neu_cell = sps.coo_matrix((num_stress + num_rob, num_bound)).tocsr()

    # For Dirichlet, the coefficients in the matrix should be duplicated the same way as
    # the row indices, but with no increment
    sgn_nd = np.tile(sgn, (nd, 1)).ravel("F")
    dir_val = sgn_nd[dir_ind_all]
    del sgn_nd
    # Column numbering starts right after the last Neumann column. dir_val
    # is ordered [u_x_1, u_y_1, u_x_2, u_y_2, ...], and dir_ind shuffles this
    # ordering. The final matrix will first have the x-coponent of the displacement
    # for each face, then the y-component, etc.
    if dir_ind.size > 0:
        dir_cell = sps.coo_matrix(
            (dir_val, (dir_ind, num_neu + num_rob + np.arange(dir_ind.size))),
            shape=(num_displ, num_bound),
        ).tocsr()
    else:
        # Special handling when no elements are found. Not sure if this is
        # necessary, or if it is me being stupid
        dir_cell = sps.coo_matrix((num_displ, num_bound)).tocsr()

    num_subfno = np.max(subfno) + 1

    # The columns in neu_cell, dir_cell are ordered from 0 to num_bound-1.
    # Map these to all half-face indices
    bnd_2_all_hf = sps.coo_matrix(
        (np.ones(num_bound), (np.arange(num_bound), bnd_ind)),
        shape=(num_bound, num_subfno * nd),
    )

    # The user of the discretization should now nothing about half faces,
    # thus map from half face to face indices.

    hf_2_f = pp.fvutils.map_hf_2_f(fno, subfno, nd).transpose()

    # the rows of rhs_bound will be ordered with first the x-component of all
    # neumann faces, then the y-component of all neumann faces, then the
    # z-component of all neumann faces. Then we will have the equivalent for
    # the dirichlet faces.

    rhs_bound = sps.vstack([neu_cell, dir_cell]) * bnd_2_all_hf

    return rhs_bound


def __unique_hooks_law(csym, casym, subcell_topology, nd):
    """
    Go from products of normal vectors with stiffness matrices (symmetric
    and asymmetric), covering both sides of faces, to a discrete Hook's law,
    that, when multiplied with sub-cell gradients, will give face stresses

    Parameters
    ----------
    csym
    casym
    unique_sub_fno
    subfno
    nd

    Returns
    -------
    hook (sps.csr) nd * (nsubfno, ncells)
    """
    # unique_sub_fno covers scalar equations only. Extend indices to cover
    # multiple dimensions
    num_eqs = csym.shape[0] / nd
    ind_single = np.tile(subcell_topology.unique_subfno, (nd, 1))
    increments = np.arange(nd) * num_eqs
    ind_all = np.reshape(ind_single + increments[:, np.newaxis], -1)

    # Unique part of symmetric and asymmetric products
    hook_sym = csym[ind_all, ::]
    hook_asym = casym[ind_all, ::]

    # Hook's law, as it comes out of the normal-vector * stiffness matrix is
    # sorted with x-component balances first, then y-, etc. Sort this to a
    # face-wise ordering
    comp2face_ind = np.argsort(
        np.tile(subcell_topology.subfno_unique, nd), kind="mergesort"
    )
    comp2face = sps.coo_matrix(
        (np.ones(comp2face_ind.size), (np.arange(comp2face_ind.size), comp2face_ind)),
        shape=(comp2face_ind.size, comp2face_ind.size),
    )
    hook = comp2face * (hook_sym + hook_asym)

    return hook


def __cell_variable_contribution(g, subcell_topology):
    """
    Construct contribution from cell center variables to local systems.
    For stress equations, these are zero, while for cell centers it is +- 1
    Parameters
    ----------
    g
    fno
    cno
    subfno

    Returns
    -------

    """
    nd = g.dim
    sgn = g.cell_faces[subcell_topology.fno, subcell_topology.cno].A

    # Contribution from cell center potentials to local systems
    # For pressure continuity, +-1
    d_cont_cell = sps.coo_matrix(
        (sgn[0], (subcell_topology.subfno, subcell_topology.cno))
    ).tocsr()
    d_cont_cell = sps.kron(sps.eye(nd), d_cont_cell)
    # Zero contribution to stress continuity

    return d_cont_cell


def __rearange_columns_displacement_eqs(d_cont_grad, d_cont_cell, num_sub_cells, nd):
    """ Transform columns of displacement balance from increasing cell
    ordering (first x-variables of all cells, then y) to increasing
    variables (first all variables of the first cells, then...)

    Parameters
    ----------
    d_cont_grad
    d_cont_cell
    num_sub_cells
    nd
    col            If true rearange columns. Else: rearange rows
    Returns
    -------

    """
    # Repeat sub-cell indices nd times. Fortran ordering (column major)
    # gives same ordering of indices as used for the scalar equation (where
    # there are nd gradient variables for each sub-cell), and thus the
    # format of each block in d_cont_grad
    rep_ci_single_blk = np.tile(np.arange(num_sub_cells), (nd, 1)).reshape(
        -1, order="F"
    )
    # Then repeat the single-block indices nd times (corresponding to the
    # way d_cont_grad is constructed by Kronecker product), and find the
    # sorting indices
    d_cont_grad_map = np.argsort(np.tile(rep_ci_single_blk, nd), kind="mergesort")
    # Use sorting indices to bring d_cont_grad to the same order as that
    # used for the columns in the stress continuity equations
    d_cont_grad = d_cont_grad[:, d_cont_grad_map]
    # For the cell displacement variables, we only need a single expansion (
    # corresponding to the second step for the gradient unknowns)
    num_cells = d_cont_cell.shape[1] / nd
    d_cont_cell_map = np.argsort(np.tile(np.arange(num_cells), nd), kind="mergesort")
    d_cont_cell = d_cont_cell[:, d_cont_cell_map]
    return d_cont_grad, d_cont_cell


def row_major_to_col_major(shape, nd, axis):
    """ Transform columns of displacement balance from increasing cell
    ordering (first x-variables of all cells, then y) to increasing
    variables (first all variables of the first cells, then...)

    Parameters
    ----------
    d_cont_grad
    d_cont_cell
    num_sub_cells
    nd
    col            If true rearange columns. Else: rearange rows
    Returns
    -------

    """
    P = sps.diags(np.ones(shape[axis])).tocsr()
    num_var = shape[axis] / nd
    mapping = np.argsort(np.tile(np.arange(num_var), nd), kind="mergesort")
    if axis == 1:
        P = P[:, mapping]
    elif axis == 0:
        P = P[mapping, :]
    else:
        raise ValueError("axis must be 0 or 1")
    return P


def _neu_face_sgn(g, neu_ind):
    neu_sgn = (g.cell_faces[neu_ind, :]).data
    if not neu_sgn.size == neu_ind.size:
        raise AssertionError("A normal sign is only well defined for a boundary face")

    sort_id = np.argsort(g.cell_faces[neu_ind, :].indices)
    return neu_sgn[sort_id]


def _zero_neu_rows(g, stress, bound_stress, bnd):
    """
    We zero out all none-diagonal elements for the neumann boundary faces.
    """
    if bnd.bc_type == "scalar":
        neu_face_x = g.dim * np.ravel(np.argwhere(bnd.is_neu))
        if g.dim == 1:
            neu_face_ind = neu_face_x
        elif g.dim == 2:
            neu_face_y = neu_face_x + 1
            neu_face_ind = np.ravel((neu_face_x, neu_face_y), "F")
        elif g.dim == 3:
            neu_face_y = neu_face_x + 1
            neu_face_z = neu_face_x + 2
            neu_face_ind = np.ravel((neu_face_x, neu_face_y, neu_face_z), "F")
        else:
            raise ValueError("Only support for dimension 1, 2, or 3")
        num_neu = neu_face_ind.size

    elif bnd.bc_type == "vectorial":
        neu_face_x = g.dim * np.ravel(np.argwhere(bnd.is_neu[0, :]))
        neu_face_y = g.dim * np.ravel(np.argwhere(bnd.is_neu[1, :])) + 1
        neu_face_ind = np.sort(np.append(neu_face_x, [neu_face_y]))
        if g.dim == 2:
            pass
        elif g.dim == 3:
            neu_face_z = g.dim * np.ravel(np.argwhere(bnd.is_neu[2, :])) + 2
            neu_face_ind = np.sort(np.append(neu_face_ind, [neu_face_z]))
        else:
            raise ValueError("Only support for dimension 1, 2, or 3")
        num_neu = neu_face_ind.size

    if not num_neu:
        return stress, bound_stress

    # Frist we zero out the boundary stress. We keep the sign of the diagonal
    # element, however we discard its value (e.g. set it to +-1). The sign
    # should be negative if the nomral vector points outwards and positive if
    # the normal vector points inwards. I'm not sure if this is correct (that
    # is, zeroing out none-diagonal elements and putting the diagonal elements
    # to +-1), but it seems to give satisfactory results.
    sgn = np.sign(np.ravel(bound_stress[neu_face_ind, neu_face_ind]))
    # Set all neumann rows to zero
    bound_stress = pp.fvutils.zero_out_sparse_rows(bound_stress, neu_face_ind, sgn)
    # For the stress matrix we zero out any rows corresponding to the Neumann
    # boundary faces (these have been moved over to the bound_stress matrix).
    stress = pp.fvutils.zero_out_sparse_rows(stress, neu_face_ind)

    return stress, bound_stress


def _sign_matrix(g, faces):
    # We find the sign of the given faces
    IA = np.argsort(faces)
    IC = np.argsort(IA)

    fi, _, sgn_d = sps.find(g.cell_faces[faces[IA], :])
    I = np.argsort(fi)
    sgn_d = sgn_d[I]
    sgn_d = sgn_d[IC]
    sgn_d = np.ravel([sgn_d] * g.dim, "F")

    sgn = sps.diags(sgn_d, 0)

    return sgn


# Convenience method for duplicating a list, with a certain increment
def expand_ind(ind, dim, increment):
    # Duplicate rows
    ind_nd = np.tile(ind, (dim, 1))
    # Add same increment to each row (0*incr, 1*incr etc.).
    ind_incr = ind_nd + increment * np.array([np.arange(dim)]).transpose()
    # Back to row vector
    ind_new = ind_incr.reshape(-1, order="F")
    return ind_new


def _eliminate_ncasym_neumann(
    ncasym, subcell_topology, bound_exclusion, cell_node_blocks, nd
):
    """
    Eliminate the asymetric part of the stress tensor such that the local systems are
    invertible.
    """
    # We expand the node indices such that we get one indices for each vector equation.
    # The equations are ordered as first all x, then all y, and so on
    node_blocks_nd = np.tile(cell_node_blocks[1], (nd, 1))
    node_blocks_nd += subcell_topology.num_nodes * np.atleast_2d(np.arange(0, nd)).T
    nno_nd = np.tile(subcell_topology.nno_unique, (nd, 1))
    nno_nd += subcell_topology.num_nodes * np.atleast_2d(np.arange(0, nd)).T

    # Each local system is associated to a node. We count the number of subcells for
    # assoiated with each node.
    _, num_sub_cells = np.unique(node_blocks_nd.ravel("C"), return_counts=True)

    # Then we count the number how many Neumann subfaces there are for each node.
    nno_neu = bound_exclusion.keep_neumann(nno_nd.ravel("C"), transform=False)
    _, idx_neu, count_neu = np.unique(nno_neu, return_inverse=True, return_counts=True)

    # The local system is invertible if the number of sub_cells (remember there is one
    # gradient for each subcell) is larger than the number of Neumann sub_faces.
    # To obtain an invertible system we remove the asymetric part around these nodes.
    count_neu = bound_exclusion.keep_neu.T * count_neu[idx_neu]
    diff_count = num_sub_cells[nno_nd.ravel("C")] - count_neu
    remove_singular = np.argwhere((diff_count < 0)).ravel()

    # remove_singular gives the indices of the subfaces. We now obtain the indices
    # as given in ncasym,
    subfno_nd = np.tile(subcell_topology.unique_subfno, (nd, 1))
    subfno_nd += subcell_topology.fno.size * np.atleast_2d(np.arange(0, nd)).T
    dof_elim = subfno_nd.ravel("C")[remove_singular]
    # and eliminate the rows corresponding to these subfaces
    pp.utils.sparse_mat.zero_rows(ncasym, dof_elim)
    print("number of ncasym eliminated: ", np.sum(dof_elim.size))
    ## the following is some code to enforce symmetric G. Comment for now
    # # Find the equations for the x-values
    # x_row = np.arange(0, round(ncasym.shape[0]/nd))
    # # Only pick out the ones that have to many Neumann conditions
    # move_row = np.in1d(x_row, dof_elim)
    # Find the column index of entries
    # x_ind_s = ncasym.indptr[x_row[move_row]]
    # x_ind_e = ncasym.indptr[x_row[move_row] + 1]
    # x_pntr = pp.utils.mcolon.mcolon(x_ind_s, x_ind_e)
    # x_indices = ncasym.indices[x_pntr]
    # # Find the \partial_x u_y and \partial_x u_z values
    # xuy = np.mod(x_indices - 3, nd*nd) == 0
    # xuz = np.mod(x_indices - 6, nd*nd) == 0
    # # Move these to the \partial_y u_x and \partial_z u_x index
    # ncasym.indices[x_pntr[xuy]] -= 2
    # ncasym.indices[x_pntr[xuz]] -= 4
    #
    # # Similar for the y-coordinates
    # y_row = np.arange(round(ncasym.shape[0]/nd), round(2*ncasym.shape[0]/nd))
    # move_row = np.in1d(y_row, dof_elim)
    # y_ind_s = ncasym.indptr[y_row[move_row]]
    # y_ind_e = ncasym.indptr[y_row[move_row] + 1]
    # y_pntr = pp.utils.mcolon.mcolon(y_ind_s, y_ind_e)
    # y_indices = ncasym.indices[y_pntr]
    # yuz = np.mod(y_indices - 7, nd*nd) == 0

    # ncasym.indices[y_pntr[yuz]] -= 2
