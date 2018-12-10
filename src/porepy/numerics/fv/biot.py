import scipy.sparse as sps
import scipy.sparse.linalg as la
import numpy as np

import porepy as pp

from porepy.numerics.fv import fvutils, mpsa


class Biot:
    def __init__(self, mechanics_keyword="mechanics", flow_keyword="flow"):
        """ Set the two keywords.

        The keywords are used to access and store parameters and discretization
        matrices.
        """
        self.mechanics_keyword = mechanics_keyword
        self.flow_keyword = flow_keyword

    def ndof(self, g):
        """ Return the number of degrees of freedom associated wiht the method.

        In this case, each cell has nd displacement variables, as well as a
        pressure variable.

        Parameters:
            g: grid, or a subclass.

        Returns:
            int: Number of degrees of freedom in the grid.

        """
        return g.num_cells * (1 + g.dim)

    def matrix_rhs(self, g, data, discretize=True):
        if discretize:
            self.discretize(g, data)

        A_biot = self.assemble_matrix(g, data)
        rhs_bound = self.rhs(g, data)
        return A_biot, rhs_bound

    # --------------------------- Helper methods for discretization ----------

    def rhs(self, g, data):
        bnd = self.rhs_bound(g, data)
        tm = self.rhs_time(g, data)
        #        src = data['source']
        return bnd + tm

    def rhs_bound(self, g, data):
        """ Boundary component of the right hand side.

        TODO: Boundary effects of coupling terms.

        Parameters:
            g: grid, or subclass, with geometry fields computed.
            data: dictionary to store the data terms. Must have been through a
                call to discretize() to discretization of right hand side.
            state: np.ndarray, solution vector from previous time step.

        Returns:
            np.ndarray: Contribution to right hand side given the current
            state.

        """
        d = data[pp.PARAMETERS][self.mechanics_keyword]["bc_values"]
        p = data[pp.PARAMETERS][self.flow_keyword]["bc_values"]

        div_flow = fvutils.scalar_divergence(g)
        div_mech = fvutils.vector_divergence(g)

        matrices_m = data[pp.DISCRETIZATION_MATRICES][self.mechanics_keyword]
        matrices_f = data[pp.DISCRETIZATION_MATRICES][self.flow_keyword]
        p_bound = (
            -div_flow * matrices_f["bound_flux"] * p - matrices_m["bound_div_d"] * d
        )
        s_bound = -div_mech * matrices_m["bound_stress"] * d
        return np.hstack((s_bound, p_bound))

    def rhs_time(self, g, data):
        """ Time component of the right hand side (dependency on previous time
        step).

        TODO: 1) Generalize this to allow other methods than Euler backwards?
              2) How about time dependent boundary conditions.

        Parameters:
            g: grid, or subclass, with geometry fields computed.
            data: dictionary to store the data terms. Must have been through a
                call to discretize() to discretization of right hand side.
            state: np.ndarray optional, solution vector from previous time
                step. Defaults to zero.

        Returns:
            np.ndarray: Contribution to right hand side given the current
            state.

        """
        state = data.get("state", None)
        if state is None:
            state = np.zeros((g.dim + 1) * g.num_cells)

        d = self.extractD(g, state, as_vector=True)
        p = self.extractP(g, state)

        parameter_dictionary = data[pp.PARAMETERS][self.mechanics_keyword]
        matrix_dictionaries = data[pp.DISCRETIZATION_MATRICES]

        d_scaling = parameter_dictionary.get("displacement_scaling", 1)
        div_d = matrix_dictionaries[self.mechanics_keyword]["div_d"]

        div_d = np.squeeze(parameter_dictionary["biot_alpha"] * div_d * d * d_scaling)
        p_cmpr = matrix_dictionaries[self.flow_keyword]["mass"] * p

        mech_rhs = np.zeros(g.dim * g.num_cells)

        return np.hstack((mech_rhs, div_d + p_cmpr))

    def discretize(self, g, data):
        """ Discretize flow and mechanics equations using FV methods.

        The parameters needed for the discretization are stored in the
        dictionary data, which should contain the following mandatory keywords:

            Related to flow equation (in data[pp.PARAMETERS][self.flow_keyword]):
                second_order_tensor: Second order tensor representing hydraulic
                    conductivity, i.e. permeability / fluid viscosity
                bc: BoundaryCondition object for flow equation. Used in mpfa.

            Related to mechanics equation (in data[pp.PARAMETERS][self.mechanids_keyword]):
                fourt_order_tensor: Fourth order tensor representing elastic moduli.
                bc: BoundaryCondition object for mechanics equation.
                    Used in mpsa.

        In addition, the following parameters are optional:

            Related to coupling terms:
                biot_alpha (double between 0 and 1): Biot's coefficient.
                    Defaults to 1.

            Related to numerics:
                inverter (str): Which method to use for block inversion. See
                    fvutils.invert_diagonal_blocks for detail, and for default
                    options.
                mpsa_eta, mpfa_eta (double): Location of continuity point in MPSA and MPFA.
                    Defaults to 1/3 for simplex grids, 0 otherwise.

        The discretization is stored in the data dictionary, in the form of
        several matrices representing different coupling terms. For details,
        and how to combine these, see self.assemble_matrix()

        Parameters:
            g (grid): Grid to be discretized.
            data (dictionary): Containing data for discretization. See above
                for specification.

        """
        # Discretization of elasticity / poro-mechanics
        self._discretize_flow(g, data)
        self._discretize_mech(g, data)
        self._discretize_compr(g, data)

    def assemble_matrix(self, g, data):
        """ Assemble the poro-elastic system matrix.

        The discretization is presumed stored in the data dictionary.

        Parameters:
            g (grid): Grid for disrcetization
            data (dictionary): Data for discretization, as well as matrices
                with discretization of the sub-parts of the system.

        Returns:
            scipy.sparse.bmat: Block matrix with the combined MPSA/MPFA
                discretization.

        """
        div_flow = fvutils.scalar_divergence(g)
        div_mech = fvutils.vector_divergence(g)
        param = data[pp.PARAMETERS]

        biot_alpha = param[self.flow_keyword]["biot_alpha"]

        matrices_m = data[pp.DISCRETIZATION_MATRICES][self.mechanics_keyword]
        matrices_f = data[pp.DISCRETIZATION_MATRICES][self.flow_keyword]
        # Put together linear system
        A_flow = div_flow * matrices_f["flux"]
        A_mech = div_mech * matrices_m["stress"]

        # Time step size
        dt = param[self.flow_keyword]["time_step"]

        d_scaling = param[self.mechanics_keyword].get("displacement_scaling", 1)
        # Matrix for left hand side
        A_biot = sps.bmat(
            [
                [A_mech, matrices_m["grad_p"] * biot_alpha],
                [
                    matrices_m["div_d"] * biot_alpha * d_scaling,
                    matrices_f["mass"] + dt * A_flow + matrices_f["biot_stabilization"],
                ],
            ]
        ).tocsr()

        return A_biot

    def _discretize_flow(self, g, data):

        # Discretiztaion using MPFA
        key = self.flow_keyword
        md = pp.Mpfa(key)

        md.discretize(g, data)

    def _discretize_compr(self, g, data):
        """
        TODO: Sort out time step (inconsistent with MassMatrix).
        """
        parameter_dictionary = data[pp.PARAMETERS][self.flow_keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.flow_keyword]
        w = parameter_dictionary["mass_weight"]
        apertures = parameter_dictionary["aperture"]
        volumes = g.cell_volumes * apertures
        matrix_dictionary["mass"] = sps.dia_matrix(
            (volumes * w, 0), shape=(g.num_cells, g.num_cells)
        )

    def _discretize_mech(self, g, data):
        """
        Discretization of poro-elasticity by the MPSA-W method.

        Implementation needs (in addition to those mentioned in mpsa function):
            1) Fields for non-zero boundary conditions. Should be simple.
            2) Split return value grad_p into forces and a divergence operator,
            so that we can compute Biot forces on a face.

        Parameters:
            g (core.grids.grid): grid to be discretized
            k (core.constit.second_order_tensor) permeability tensor
            bound_mech: Boundary condition object for mechancis
            bound_flow: Boundary condition object for flow.
            constit (porepy.bc.bc.BoundaryCondition) class for boundary values
            faces (np.ndarray) faces to be considered. Intended for partial
                discretization, may change in the future
            mpsa_eta Location of pressure continuity point. Should be 1/3 for simplex
                grids, 0 otherwise. On boundary faces with Dirichlet conditions,
                eta=0 will be enforced. Defaults to the values computed by
                fvutils.determine_eta(g).
            inverter (string) Block inverter to be used, either numba (default),
                cython or python. See fvutils.invert_diagonal_blocks for details.

        Returns:
            scipy.sparse.csr_matrix (shape num_faces * dim, num_cells * dim): stres
                discretization, in the form of mapping from cell displacement to
                face stresses.
            scipy.sparse.csr_matrix (shape num_faces * dim, num_faces * dim):
                discretization of boundary conditions. Interpreted as istresses
                induced by the boundary condition (both Dirichlet and Neumann). For
                Neumann, this will be the prescribed stress over the boundary face,
                and possibly stress on faces having nodes on the boundary. For
                Dirichlet, the values will be stresses induced by the prescribed
                displacement.  Incorporation as a right hand side in linear system
                by multiplication with divergence operator.
            scipy.sparse.csr_matrix (shape num_cells * dim, num_cells): Forces from
                the pressure gradient (I*p-term), represented as body forces.
                TODO: Should rather be represented as forces on faces.
            scipy.sparse.csr_matrix (shape num_cells, num_cells * dim): Trace of
                strain matrix, cell-wise.
            scipy.sparse.csr_matrix (shape num_cells x num_cells): Stabilization
                term.

        Example:
            # Set up a Cartesian grid
            g = structured.CartGrid([5, 5])
            c = tensor.FourthOrderTensor(g.dim, np.ones(g.num_cells))
            k = tensor.SecondOrderTensor(g.dim, np.ones(g.num_cells))

            # Dirirchlet boundary conditions for mechanics
            bound_faces = g.get_all_boundary_faces().ravel()
            bnd = bc.BoundaryCondition(g, bound_faces, ['dir'] * bound_faces.size)

            # Use no boundary conditions for flow, will default to homogeneous
            # Neumann.

            # Discretization
            stress, bound_stress, grad_p, div_d, stabilization = biot(g, c, bnd)
            flux, bound_flux = mpfa(g, k, None)

            # Source in the middle of the domain
            q_mech = np.zeros(g.num_cells * g.dim)

            # Divergence operator for the grid
            div_mech = fvutils.vector_divergence(g)
            div_flow = fvutils.scalar_divergence(g)
            a_mech = div_mech * stress
            a_flow = div_flow * flux

            a_biot = sps.bmat([[a_mech, grad_p], [div_d, a_flow +
                                                           stabilization]])

            # Zero boundary conditions by default.

            # Injection in the middle of the domain
            rhs = np.zeros(g.num_cells * (g.dim + 1))
            rhs[g.num_cells * g.dim + np.ceil(g.num_cells / 2)] = 1
            x = sps.linalg.spsolve(A, rhs)

            u_x = x[0:g.num_cells * g.dim: g.dim]
            u_y = x[1:g.num_cells * g.dim: g.dim]
            p = x[g.num_cells * gdim:]

        """
        parameters_m = data[pp.PARAMETERS][self.mechanics_keyword]
        parameters_f = data[pp.PARAMETERS][self.flow_keyword]
        matrices_m = data[pp.DISCRETIZATION_MATRICES][self.mechanics_keyword]
        matrices_f = data[pp.DISCRETIZATION_MATRICES][self.flow_keyword]
        bound_mech = parameters_m["bc"]
        bound_flow = parameters_f["bc"]
        constit = parameters_m["fourth_order_tensor"]

        eta = parameters_m.get("mpsa_eta", fvutils.determine_eta(g))
        inverter = parameters_m.get("inverter", None)

        # The grid coordinates are always three-dimensional, even if the grid
        # is really 2D. This means that there is not a 1-1 relation between the
        # number of coordinates of a point / vector and the real dimension.
        # This again violates some assumptions tacitly made in the
        # discretization (in particular that the number of faces of a cell that
        # meets in a vertex equals the grid dimension, and that this can be
        # used to construct an index of local variables in the discretization).
        # These issues should be possible to overcome, but for the moment, we
        # simply force 2D grids to be proper 2D.
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
        subcell_topology = fvutils.SubcellTopology(g)
        # Obtain mappings to exclude boundary faces for mechanics
        bound_mech_sub = fvutils.boundary_to_sub_boundary(bound_mech, subcell_topology)
        bound_exclusion_mech = fvutils.ExcludeBoundaries(
            subcell_topology, bound_mech_sub, nd
        )
        # ... and flow
        bound_flow_sub = fvutils.boundary_to_sub_boundary(bound_flow, subcell_topology)
        bound_exclusion_flow = fvutils.ExcludeBoundaries(
            subcell_topology, bound_flow_sub, nd
        )

        num_subhfno = subcell_topology.subhfno.size

        num_nodes = np.diff(g.face_nodes.indptr)
        sgn = g.cell_faces[subcell_topology.fno, subcell_topology.cno].A

        # The pressure gradient term in the mechanics equation is discretized
        # as a force on the faces. The right hand side is thus formed of the
        # normal vectors.
        def build_rhs_normals_single_dimension(dim):
            val = (
                g.face_normals[dim, subcell_topology.fno]
                * sgn
                / num_nodes[subcell_topology.fno]
            )
            mat = sps.coo_matrix(
                (val.squeeze(), (subcell_topology.subfno, subcell_topology.cno)),
                shape=(subcell_topology.num_subfno, subcell_topology.num_cno),
            )
            return mat

        rhs_normals = build_rhs_normals_single_dimension(0)
        for iter1 in range(1, nd):
            this_dim = build_rhs_normals_single_dimension(iter1)
            rhs_normals = sps.vstack([rhs_normals, this_dim])

        rhs_normals = bound_exclusion_mech.exclude_dirichlet(rhs_normals)

        num_dir_subface = (
            bound_exclusion_mech.exclude_neu.shape[1]
            - bound_exclusion_mech.exclude_neu.shape[0]
        ) * nd
        # No right hand side for cell displacement equations.
        rhs_normals_displ_var = sps.coo_matrix(
            (
                nd * subcell_topology.num_subfno - num_dir_subface,
                subcell_topology.num_cno,
            )
        )

        # Why minus?
        rhs_normals = -sps.vstack([rhs_normals, rhs_normals_displ_var])
        del rhs_normals_displ_var

        # Call core part of MPSA
        hook, igrad, rhs_cells, cell_node_blocks, hook_normal = mpsa.mpsa_elasticity(
            g, constit, subcell_topology, bound_exclusion_mech, eta, inverter
        )

        # Output should be on face-level (not sub-face)
        hf2f = fvutils.map_hf_2_f(
            subcell_topology.fno_unique, subcell_topology.subfno_unique, nd
        )

        # Stress discretization
        stress = hf2f * hook * igrad * rhs_cells

        # Right hand side for boundary discretization
        rhs_bound = mpsa.create_bound_rhs(
            bound_mech_sub, bound_exclusion_mech, subcell_topology, g, False
        )
        rhs_bound = rhs_bound * hf2f.T
        # Discretization of boundary values
        bound_stress = hf2f * hook * igrad * rhs_bound

        # Face-wise gradient operator. Used for the term grad_p in Biot's
        # equations.
        rows = fvutils.expand_indices_nd(subcell_topology.cno, nd)
        cols = np.arange(num_subhfno * nd)
        vals = np.tile(sgn, (nd, 1)).ravel("F")
        div_gradp = sps.coo_matrix(
            (vals, (rows, cols)),
            shape=(subcell_topology.num_cno * nd, num_subhfno * nd),
        ).tocsr()

        #        del hook, rhs_bound
        del rows, cols, vals

        grad_p = div_gradp * hook_normal * igrad * rhs_normals
        # assert np.allclose(grad_p.sum(axis=0), np.zeros(g.num_cells))

        del hook_normal, div_gradp

        div = self._subcell_gradient_to_cell_scalar(g, cell_node_blocks)

        div_d = div * igrad * rhs_cells

        # The boundary discretization of the div_d term is represented directly
        # on the cells, instead of going via the faces.
        bound_div_d = div * igrad * rhs_bound
        del rhs_cells

        stabilization = div * igrad * rhs_normals

        matrices_m["stress"] = stress
        matrices_m["bound_stress"] = bound_stress
        matrices_m["grad_p"] = grad_p
        matrices_m["div_d"] = div_d
        matrices_f["biot_stabilization"] = stabilization
        matrices_m["bound_div_d"] = bound_div_d

    def _face_vector_to_scalar(self, nf, nd):
        """ Create a mapping from vector quantities on faces (stresses) to scalar
        quantities. The mapping is intended for the boundary discretization of the
        displacement divergence term  (coupling term in the flow equation).

        Parameters:
            nf (int): Number of faces in the grid
        """
        rows = np.tile(np.arange(nf), ((nd, 1))).reshape((1, nd * nf), order="F")[0]

        cols = fvutils.expand_indices_nd(np.arange(nf), nd)
        vals = np.ones(nf * nd)
        return sps.coo_matrix((vals, (rows, cols))).tocsr()

    def _subcell_gradient_to_cell_scalar(self, g, cell_node_blocks):
        """ Create a mapping from sub-cell gradients to cell-wise traces of the gradient
        operator. The mapping is intended for the discretization of the term div(u)
        (coupling term in flow equation).
        """
        # To pick out the trace of the strain tensor, we access elements
        #   (2d): 0 (u_x) and 3 (u_y)
        #   (3d): 0 (u_x), 4 (u_y), 8 (u_z)
        nd = g.dim
        if nd == 2:
            trace = np.array([0, 3])
        elif nd == 3:
            trace = np.array([0, 4, 8])

        # Sub-cell wise trace of strain tensor: One row per sub-cell
        row, col = np.meshgrid(np.arange(cell_node_blocks.shape[1]), trace)
        # Adjust the columns to hit each sub-cell
        incr = np.cumsum(nd ** 2 * np.ones(cell_node_blocks.shape[1])) - nd ** 2
        col += incr.astype("int32")

        # Integrate the trace over the sub-cell, that is, distribute the cell
        # volumes equally over the sub-cells
        num_cell_nodes = g.num_cell_nodes()
        cell_vol = g.cell_volumes / num_cell_nodes
        val = np.tile(cell_vol[cell_node_blocks[0]], (nd, 1))
        # and we have our mapping from vector to scalar values on sub-cells
        vector_2_scalar = sps.coo_matrix(
            (val.ravel("F"), (row.ravel("F"), col.ravel("F")))
        ).tocsr()

        # Mapping from sub-cells to cells
        div_op = sps.coo_matrix(
            (
                np.ones(cell_node_blocks.shape[1]),
                (cell_node_blocks[0], np.arange(cell_node_blocks.shape[1])),
            )
        ).tocsr()
        # and the composed map
        div = div_op * vector_2_scalar
        return div

    # ----------------------- Linear solvers -------------------------------------

    def solve(self, A, solver="direct", **kwargs):

        solver = solver.strip().lower()
        if solver == "direct":

            def slv(b):
                x = la.spsolve(A, b)
                return x

        elif solver == "factorized":
            slv = la.factorized(A.tocsc())

        else:
            raise ValueError("Unknown solver " + solver)

        return slv

    # ----------------------- Methods for post processing -------------------------
    def extractD(self, g, u, dims=None, as_vector=False):
        """ Extract displacement field from solution.

        Parameters:
            g: grid, or a subclass.
            u (np.ndarray): Solution variable, representing displacements and
                pressure.
            dim (list of int, optional): Which dimension to extract. If None,
                all dimensions are returned.
        Returns:
            list of np.ndarray: Displacement variables in the specified
                dimensions.

        """
        if dims is None:
            dims = np.arange(g.dim)
        vals = []

        inds = np.arange(0, g.num_cells * g.dim, g.dim)

        for d in dims:
            vals.append(u[d + inds])
        if as_vector:
            vals = np.asarray(vals).reshape((-1, 1), order="F")
            return vals
        else:
            return vals

    def extractP(self, g, u):
        """ Extract pressure field from solution.

        Parameters:
            g: grid, or a subclass.
            u (np.ndarray): Solution variable, representing displacements and
                pressure.

        Returns:
            np.ndarray: Pressure part of solution vector.

        """
        return u[g.dim * g.num_cells :]

    def compute_flux(self, g, u, data):
        """ Compute flux field corresponding to a solution.

        Parameters:
            g: grid, or a subclass.
            u (np.ndarray): Solution variable, representing displacements and
                pressure.
            bc_flow (np.ndarray): Flux boundary values.
            data (dictionary): Dictionary related to grid and problem. Should
                contain boundary discretization.

        Returns:
            np.ndarray: Flux over all faces

        """
        flux_discr = data[pp.DISCRETIZATION_MATRICES][self.flow_keyword]["flux"]
        bound_flux = data[pp.DISCRETIZATION_MATRICES][self.flow_keyword]["bound_flux"]
        bound_val = data[pp.PARAMETERS][self.flow_keyword]["bc_values"]
        p = self.extractP(g, u)
        flux = flux_discr * p + bound_flux * bound_val
        return flux

    def compute_stress(self, g, u, data):
        """ Compute stress field corresponding to a solution.

        Parameters:
            g: grid, or a subclass.
            u (np.ndarray): Solution variable, representing displacements and
                pressure.
            bc_flow (np.ndarray): Flux boundary values.
            data (dictionary): Dictionary related to grid and problem. Should
                contain boundary discretization.

        Returns:
            np.ndarray, g.dim * g.num_faces: Stress over all faces. Stored as
                all stress values on the first face, then the second etc.

        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.mechanics_keyword]
        stress_discr = matrix_dictionary["stress"]
        bound_stress = matrix_dictionary["bound_stress"]
        bound_val = data[pp.PARAMETERS][self.mechanics_keyword]["bc_values"]
        d = self.extractD(g, u, as_vector=True)
        stress = np.squeeze(stress_discr * d) + (bound_stress * bound_val)
        return stress


class GradP(pp.numerics.interface_laws.elliptic_discretization.VectorEllipticDiscretization):
    """ Class for the pressure gradientdivergence term of the Biot equation.
    """

    def ndof(self, g):
        """ Return the number of degrees of freedom associated to the method.

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
        """ Discretize the pressure gradient term of the Biot equation.

        Parameters:
            g (pp.Grid): grid, or a subclass, with geometry fields computed.
            data (dict): For entries, see above.

        Raises:
            NotImplementedError, the discretization should be performed using the
            discretize method of the Biot class.
        """
        raise NotImplementedError(
            """No discretize method implemented for the GradP
                                  class. See the Biot class."""
        )

    def assemble_matrix_rhs(self, g, data):
        """ Return the matrix and right-hand side for a discretization of the pressure
        gradient term of the Biot equation.

        Parameters:
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data. For details on necessary keywords,
                see method discretize()

        Returns:
            matrix: sparse csr (g.dim * g_num_cells, g.dim * g_num_cells) Discretization
            matrix.
            rhs: array (g.dim * g_num_cells) Right-hand side.
        """
        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

    def assemble_matrix(self, g, data):
        """ Return the matrix and right-hand side for a discretization of the pressure
        gradient term of the Biot equation.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization. The
                size of the matrix will depend on the specific discretization.

        Raises:
            ValueError if the pressure gradient term has not already been discretized.
        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        if not "grad_p" in matrix_dictionary:
            raise ValueError(
                """GradP class requires a pre-computed discretization to be
                             stored in the matrix dictionary."""
            )
        biot_alpha = data[pp.PARAMETERS][self.keyword]["biot_alpha"]
        return matrix_dictionary["grad_p"] * biot_alpha

    def assemble_rhs(self, g, data):
        """ Return the zero right-hand side for a discretization of the pressure
        gradient term.

        @Runar: Is it correct that this is zero.

        Parameters:
            g (Grid): Computational grid.
            data (dictionary): With data stored.

        Returns:
            np.ndarray: Zero right hand side vector with representation of boundary
                conditions.
        """
        return np.zeros(self.ndof(g))


class DivD(pp.numerics.interface_laws.elliptic_discretization.VectorEllipticDiscretization):
    """ Class for the displacement divergence term of the Biot equation.
    """

    def ndof(self, g):
        """ Return the number of degrees of freedom associated to the method.

        In this case number of cells times dimension (stress dof).

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_cells

    def extract_displacement(self, g, solution_array, d):
        """ Extract the displacement part of a solution.

        The method is trivial for finite volume methods, with the displacement being
        the only primary variable.

        Parameters:
            g (grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid obtained from
                either a mono-dimensional or a mixed-dimensional problem.
            d (dictionary): Data dictionary associated with the grid. Not used,
                but included for consistency reasons.
        Returns:
            np.array (g.num_cells): Displacement solution vector. Will be identical
                to solution_array.
        """
        return solution_array

    def discretize(self, g, data):
        """ Discretize the displacement divergence term of the Biot equation.

        Parameters:
            g (pp.Grid): grid, or a subclass, with geometry fields computed.
            data (dict): For entries, see above.

        Raises:
            NotImplementedError, the discretization should be performed using the
            discretize method of the Biot class.
        """
        raise NotImplementedError(
            """No discretize method implemented for the GradP
                                  class. See the Biot class."""
        )

    def assemble_matrix_rhs(self, g, data):
        """ Return the matrix and right-hand side for a discretization of the
        displacement divergence term of the Biot equation.

        Parameters:
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data. For details on necessary keywords,
                see method discretize()

        Returns:
            matrix: sparse csr (g.dim * g_num_cells, g.dim * g_num_cells) Discretization
            matrix.
            rhs: array (g.dim * g_num_cells) Right-hand side.
        """
        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

    def assemble_matrix(self, g, data):
        """ Return the matrix and right-hand side for a discretization of the
        displacement divergence term of the Biot equation.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization. The
                size of the matrix will depend on the specific discretization.

        Raises:
            ValueError if the displacement divergence term has not already been
            discretized.
        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        if not "div_d" in matrix_dictionary:
            raise ValueError(
                """DivD class requires a pre-computed discretization to be
                             stored in the matrix dictionary."""
            )
        biot_alpha = data[pp.PARAMETERS][self.keyword]["biot_alpha"]
        return matrix_dictionary["div_d"] * biot_alpha

    def assemble_rhs(self, g, data):
        """ Return the zero right-hand side for a discretization of the displacement
        divergence term.

        @Runar: Is it correct that this is zero.

        Parameters:
            g (Grid): Computational grid.
            data (dictionary): With data stored.

        Returns:
            np.ndarray: Zero right hand side vector with representation of boundary
                conditions.
        """
        return np.zeros(self.ndof(g))


class BiotStabilization(pp.numerics.interface_laws.elliptic_discretization.VectorEllipticDiscretization):
    """ Class for the stabilization term of the Biot equation.
    """

    def ndof(self, g):
        """ Return the number of degrees of freedom associated to the method.

        In this case number of cells times dimension (stress dof).

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_cells

    def extract_displacement(self, g, solution_array, d):
        """ Extract the displacement part of a solution.

        The method is trivial for finite volume methods, with the displacement being
        the only primary variable.

        Parameters:
            g (grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid obtained from
                either a mono-dimensional or a mixed-dimensional problem.
            d (dictionary): Data dictionary associated with the grid. Not used,
                but included for consistency reasons.
        Returns:
            np.array (g.num_cells): Displacement solution vector. Will be identical
                to solution_array.
        """
        return solution_array

    def discretize(self, g, data):
        """ Discretize the stabilization term of the Biot equation.

        Parameters:
            g (pp.Grid): grid, or a subclass, with geometry fields computed.
            data (dict): For entries, see above.

        Raises:
            NotImplementedError, the discretization should be performed using the
            discretize method of the Biot class.
        """
        raise NotImplementedError(
            """No discretize method implemented for the DivD
                                  class. See the Biot class."""
        )

    def assemble_matrix_rhs(self, g, data):
        """ Return the matrix and right-hand side for a discretization of the
        stabilization term of the Biot equation.

        Parameters:
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data. For details on necessary keywords,
                see method discretize()

        Returns:
            matrix: sparse csr (g.dim * g_num_cells, g.dim * g_num_cells) Discretization
            matrix.
            rhs: array (g.dim * g_num_cells) Right-hand side.
        """
        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

    def assemble_matrix(self, g, data):
        """ Return the matrix and right-hand side for a discretization of the
        stabilization term of the Biot equation.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization. The
                size of the matrix will depend on the specific discretization.

        Raises:
            ValueError if the stabilization term has not already been
            discretized.
        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        if not "biot_stabilization" in matrix_dictionary:
            raise ValueError(
                """BiotStabilization class requires a pre-computed
                             discretization to be stored in the matrix dictionary."""
            )

        return matrix_dictionary["biot_stabilization"]

    def assemble_rhs(self, g, data):
        """ Return the zero right-hand side for a discretization of the displacement
        divergence term.

        @Runar: Is it correct that this is zero.

        Parameters:
            g (Grid): Computational grid.
            data (dictionary): With data stored.

        Returns:
            np.ndarray: Zero right hand side vector with representation of boundary
                conditions.
        """
        return np.zeros(self.ndof(g))
