import scipy.sparse as sps
import scipy.sparse.linalg as la
import numpy as np
import warnings

import porepy as pp

from porepy.numerics.fv import fvutils, mpsa


class Biot:
    def __init__(
        self,
        mechanics_keyword="mechanics",
        flow_keyword="flow",
        vector_variable="displacement",
        scalar_variable="pressure",
    ):
        """ Set the two keywords.

        The keywords are used to access and store parameters and discretization
        matrices.
        """
        self.mechanics_keyword = mechanics_keyword
        self.flow_keyword = flow_keyword
        # Set variable names for the vector and scalar variable, used to access
        # solutions from previous time steps
        self.vector_variable = vector_variable
        self.scalar_variable = scalar_variable

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

        There is an assumption on constant mechanics BCs, see DivU.assemble_matrix().

        Parameters:
            g: grid, or subclass, with geometry fields computed.
            data: dictionary to store the data terms. Must have been through a
                call to discretize() to discretization of right hand side.

        Returns:
            np.ndarray: Contribution to right hand side.

        """
        d = data[pp.PARAMETERS][self.mechanics_keyword]["bc_values"]
        p = data[pp.PARAMETERS][self.flow_keyword]["bc_values"]

        div_flow = fvutils.scalar_divergence(g)
        div_mech = fvutils.vector_divergence(g)

        matrices_m = data[pp.DISCRETIZATION_MATRICES][self.mechanics_keyword]
        matrices_f = data[pp.DISCRETIZATION_MATRICES][self.flow_keyword]

        bound_stress = matrices_m["bound_stress"]
        bound_flux = matrices_f["bound_flux"]
        if bound_stress.shape[0] != g.dim * g.num_faces:
            # If the boundary conditions are given on the
            # subfaces we have to map them to the faces
            hf2f_nd = pp.fvutils.map_hf_2_f(g=g)
            hf2f = pp.fvutils.map_hf_2_f(nd=1, g=g)
            bound_stress = hf2f_nd * bound_stress
            bound_flux = hf2f * bound_flux

        dt = data[pp.PARAMETERS][self.flow_keyword]["time_step"]
        p_bound = -div_flow * bound_flux * p * dt
        s_bound = -div_mech * bound_stress * d
        # Note that the following is zero only if the previous time step is zero.
        # See comment in the DivU class
        biot_alpha = data[pp.PARAMETERS][self.flow_keyword]["biot_alpha"]
        div_u_rhs = -0 * biot_alpha * matrices_f["bound_div_u"] * d
        return np.hstack((s_bound, p_bound + div_u_rhs))

    def rhs_time(self, g, data):
        """ Time component of the right hand side (dependency on previous time
        step).

        TODO: 1) Generalize this to allow other methods than Euler backwards?
              2) How about time dependent boundary conditions.

        Parameters:
            g: grid, or subclass, with geometry fields computed.
            data: dictionary to store the data terms. Must have been through a
                call to discretize() to discretization of right hand side. May
                contain the field pp.STATE, storing the solution vectors from previous
                time step. Defaults to zero.

        Returns:
            np.ndarray: Contribution to right hand side given the current state.

        """
        state = data.get(pp.STATE, None)
        if state is None:
            state = {
                self.vector_variable: np.zeros(g.dim * g.num_cells),
                self.scalar_variable: np.zeros(g.num_cells),
            }

        d = self.extract_vector(g, state[self.vector_variable], as_vector=True)
        p = state[self.scalar_variable]

        parameter_dictionary = data[pp.PARAMETERS][self.mechanics_keyword]
        matrix_dictionaries = data[pp.DISCRETIZATION_MATRICES]

        div_u = matrix_dictionaries[self.flow_keyword]["div_u"]

        div_u_rhs = np.squeeze(parameter_dictionary["biot_alpha"] * div_u * d)
        p_cmpr = matrix_dictionaries[self.flow_keyword]["mass"] * p

        mech_rhs = np.zeros(g.dim * g.num_cells)

        # The stabilization is the pressure contribution to the div u part of the
        # fluid mass conservation, thus, it  need a right hand side in the implicit Euler
        # discretization.
        stab_time = matrix_dictionaries[self.flow_keyword]["biot_stabilization"] * p

        return np.hstack((mech_rhs, div_u_rhs + p_cmpr + stab_time))

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
        if matrices_m["stress"].shape[0] != g.dim * g.num_faces:
            # If we give the boundary conditions for subfaces, the discretization
            # will also be returned for the subfaces. We therefore have to map
            # everything to faces before we proceeds.
            hf2f_nd = pp.fvutils.map_hf_2_f(g=g)
            hf2f = pp.fvutils.map_hf_2_f(nd=1, g=g)
            stress = hf2f_nd * matrices_m["stress"]
            flux = hf2f * matrices_f["flux"]
            grad_p = hf2f_nd * matrices_m["grad_p"]
        else:
            stress = matrices_m["stress"]
            flux = matrices_f["flux"]
            grad_p = matrices_m["grad_p"]

        A_flow = div_flow * flux
        A_mech = div_mech * stress
        grad_p = div_mech * grad_p
        stabilization = matrices_f["biot_stabilization"]

        # Time step size
        dt = param[self.flow_keyword]["time_step"]

        # Matrix for left hand side
        A_biot = sps.bmat(
            [
                [A_mech, grad_p],
                [
                    matrices_f["div_u"] * biot_alpha,
                    matrices_f["mass"] + dt * A_flow + stabilization,
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
            1) Split return value grad_p into forces and a divergence operator,
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
            scipy.sparse.csr_matrix (shape num_faces * dim, num_cells * dim): stress
                discretization, in the form of mapping from cell displacement to
                face stresses.
            scipy.sparse.csr_matrix (shape num_faces * dim, num_faces * dim):
                discretization of boundary conditions. Interpreted as stresses
                induced by the boundary condition (both Dirichlet and Neumann). For
                Neumann, this will be the prescribed stress over the boundary face,
                and possibly stress on faces having nodes on the boundary. For
                Dirichlet, the values will be stresses induced by the prescribed
                displacement.  Incorporation as a right hand side in linear system
                by multiplication with divergence operator.
            scipy.sparse.csr_matrix (shape num_faces * dim, num_cells): Forces from
                the pressure gradient (-I*p-term), represented as forces on the faces.
            scipy.sparse.csr_matrix (shape num_cells, num_cells * dim): Trace of
                strain matrix, cell-wise.
            scipy.sparse.csr_matrix (shape num_cells x num_cells): Stabilization
                term.
       """
        parameters_m = data[pp.PARAMETERS][self.mechanics_keyword]
        matrices_m = data[pp.DISCRETIZATION_MATRICES][self.mechanics_keyword]
        matrices_f = data[pp.DISCRETIZATION_MATRICES][self.flow_keyword]
        bound_mech = parameters_m["bc"]
        constit = parameters_m["fourth_order_tensor"]

        eta = parameters_m.get("mpsa_eta", fvutils.determine_eta(g))
        inverter = parameters_m.get("inverter", None)

        alpha = parameters_m["biot_alpha"]

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
        # The boundary conditions must be given on the subfaces
        if bound_mech.num_faces == subcell_topology.num_subfno_unique:
            subface_rhs = True
        else:
            # If they are given on the faces, expand the boundary conditions
            bound_mech = pp.fvutils.boundary_to_sub_boundary(
                bound_mech, subcell_topology
            )
            subface_rhs = False

        # Obtain mappings to exclude boundary faces for mechanics
        bound_exclusion_mech = fvutils.ExcludeBoundaries(
            subcell_topology, bound_mech, nd
        )

        # Call core part of MPSA
        hook, igrad, rhs_cells, cell_node_blocks = mpsa.mpsa_elasticity(
            g, constit, subcell_topology, bound_exclusion_mech, eta, inverter
        )

        # Stress discretization
        stress = hook * igrad * rhs_cells

        # Right hand side for boundary discretization
        rhs_bound = mpsa.create_bound_rhs(
            bound_mech, bound_exclusion_mech, subcell_topology, g, subface_rhs
        )
        # Discretization of boundary values
        bound_stress = hook * igrad * rhs_bound

        if not subface_rhs:
            # If the boundary condition is given for faces we return the discretization
            # on for the face values. Otherwise it is defined for the subfaces.
            hf2f = fvutils.map_hf_2_f(
                subcell_topology.fno_unique, subcell_topology.subfno_unique, nd
            )
            bound_stress = hf2f * bound_stress * hf2f.T
            stress = hf2f * stress
            rhs_bound = rhs_bound * hf2f.T

        # trace of strain matrix
        div = self._subcell_gradient_to_cell_scalar(g, cell_node_blocks)
        div_u = div * igrad * rhs_cells

        # The boundary discretization of the div_u term is represented directly
        # on the cells, instead of going via the faces.
        bound_div_u = div * igrad * rhs_bound

        # Call discretization of grad_p-term
        rhs_jumps, grad_p_face = self.discretize_biot_grad_p(
            g, subcell_topology, alpha, bound_exclusion_mech
        )

        if subface_rhs:
            # If boundary conditions are given on subfaces we keep the subface
            # discretization
            grad_p = hook * igrad * rhs_jumps + grad_p_face
        else:
            # otherwise we map it to faces
            grad_p = hf2f * (hook * igrad * rhs_jumps + grad_p_face)

        stabilization = div * igrad * rhs_jumps

        # We obtain the reconstruction of displacments. This is equivalent as for
        # mpsa, but we get a contribution from the pressures.
        dist_grad, cell_centers = pp.numerics.fv.mpsa.reconstruct_displacement(
            g, subcell_topology, eta
        )

        disp_cell = dist_grad * igrad * rhs_cells + cell_centers
        disp_bound = dist_grad * igrad * rhs_bound
        disp_pressure = dist_grad * igrad * rhs_jumps

        # Add discretizations to data
        matrices_m["stress"] = stress
        matrices_m["bound_stress"] = bound_stress
        matrices_f["div_u"] = div_u
        matrices_f["bound_div_u"] = bound_div_u
        matrices_m["grad_p"] = grad_p
        matrices_f["biot_stabilization"] = stabilization
        matrices_m["bound_displacement_cell"] = disp_cell
        matrices_m["bound_displacement_face"] = disp_bound
        matrices_m["bound_displacement_pressure"] = disp_pressure

    def discretize_biot_grad_p(self, g, subcell_topology, alpha, bound_exclusion):
        """
        Consistent discretization of grad_p-term in MPSA-W method.

        Parameters:
            g (core.grids.grid): grid to be discretized
            subcell_topology: Wrapper class for numbering of subcell faces, cells
                etc.
            alpha: Biot's coupling coefficient, given as a scalar in input
            bound_exclusion: Object that can eliminate faces related to boundary
                conditions.

        Returns:
            scipy.sparse.csr_matrix (shape num_subcells * dim, num_cells):
            discretization of the jumps in [n alpha p] term,
            ready to be multiplied with inverse gradient
            scipy.sparse.csr_matrix (shape num_subfaces * dim, num_cells):
                discretization of the force on the face due to cell-centre
                pressure from a unique side.

        Method properties and implementation details.
        Basis functions, namely 'stress' and 'bound_stress', for the displacement
        discretization are obtained as in standard MPSA-W method.
        Pressure is represented as forces in the cells.
        However, jumps in pressure forces over a cell face act as force
        imbalance, and thus induce additional displacement gradients in the sub-cells.
        An additional system is set up, which applies non-zero conditions to the
        traction continuity equation. This can be expressed as a linear system on the form

            (i)   A * grad_u            = I
            (ii)  B * grad_u + C * u_cc = 0
            (iii) 0            D * u_cc = 0

        Thus (i)-(iii) can be inverted to express the additional displacement gradients
        due to imbalance in pressure forces as in terms of the cell center variables.
        Thus we can compute the basis functions 'grad_p_jumps' on the sub-cells.
        To ensure traction continuity, as soon as a convention is chosen for what side
        the force evaluation should be considered on, an additional term, called
        'grad_p_face', is added to the full force. This latter term represents the force
        due to cell-center pressure acting on the face from the chosen side.
        The pair subfno_unique-unique_subfno gives the side convention.
        The full force on the face is therefore given by

        t = stress * u + bound_stress * u_b + alpha * (grad_p_jumps + grad_p_face) * p

        The strategy is as follows.
        1. compute product normal_vector * alpha and get a map for vector problems
        2. assemble r.h.s. for the new linear system, needed for the term 'grad_p_jumps'
        3. compute term 'grad_p_face'
        """

        nd = g.dim

        num_subhfno = subcell_topology.subhfno.size
        num_subfno_unique = subcell_topology.num_subfno_unique
        num_subfno = subcell_topology.num_subfno

        # Step 1

        # The implementation is valid for tensor Biot coefficients, but for the
        # moment, we only allow for scalar inputs.
        # Take Biot's alpha as a tensor
        alpha_tensor = pp.SecondOrderTensor(nd, alpha * np.ones(g.num_cells))

        if nd == 2:
            alpha_tensor.values = np.delete(alpha_tensor.values, (2), axis=0)
            alpha_tensor.values = np.delete(alpha_tensor.values, (2), axis=1)

        # Obtain normal_vector * alpha, pairings of cells and nodes (which together
        # uniquely define sub-cells, and thus index for gradients)
        nAlpha_grad, cell_node_blocks, sub_cell_index = fvutils.scalar_tensor_vector_prod(
            g, alpha_tensor, subcell_topology
        )
        # transfer nAlpha to a subface-based quantity by pairing expressions on the
        # two sides of the subface
        unique_nAlpha_grad = subcell_topology.pair_over_subfaces(nAlpha_grad)

        # convenience method for reshaping nAlpha from face-based
        # to component-based. This is to build a block diagonal sparse matrix
        # compatible with igrad * rhs_units, that is first all x-component, then y, and z
        def map_tensor(mat, nd, ind):
            newmat = mat[:, ind[0]]

            for i in range(1, nd):
                this_dim = mat[:, ind[i]]
                newmat = sps.block_diag([newmat, this_dim])

            return newmat

        # Reshape nAlpha component-wise
        nAlpha_grad = map_tensor(nAlpha_grad, nd, sub_cell_index)
        unique_nAlpha_grad = map_tensor(unique_nAlpha_grad, nd, sub_cell_index)

        # Step 2

        # The pressure term in the tractions continuity equation is discretized
        # as a force on the faces. The right hand side is thus formed of the
        # unit vector.
        def build_rhs_units_single_dimension(dim):
            # EK: Can we skip argument dim?
            vals = np.ones(num_subfno_unique)
            ind = subcell_topology.subfno_unique
            mat = sps.coo_matrix(
                (vals, (ind, ind)), shape=(num_subfno_unique, num_subfno_unique)
            )
            return mat

        rhs_units = build_rhs_units_single_dimension(0)

        for i in range(1, nd):
            this_dim = build_rhs_units_single_dimension(i)
            rhs_units = sps.block_diag([rhs_units, this_dim])

        # We get the sign of the subfaces. This will be needed for the boundary
        # faces if the normal vector points inn. This is because boundary
        # conditions always are set as if the normals point out.
        sgn = g.cell_faces[
            subcell_topology.fno_unique, subcell_topology.cno_unique
        ].A.ravel("F")
        # NOTE: For some reason one should not multiply with the sign, but I don't
        # understand why. It should not matter much for the Biot alpha term since
        # by construction the biot_alpha_jumps and biot_alpha_force will cancel for
        # Neumann boundaries. We keep the sign matrix as an Identity matrix to remember
        # where it should be multiplied:
        sgn_nd = np.tile(np.abs(sgn), (g.dim, 1))

        # In the local systems the coordinates are C ordered (first all x, then all y,
        # etc.), while they are ordered as F (first x,y,z of subface 1 then x,y,z of
        # subface 2) elsewhere. If there is a problem with the stabilization or the
        # boundary, this might be the place to start debugging. The fno_unique
        # and cno_unique is chosen such that the face normal of fno points out of cno
        # for internal faces, thus, sgn_diag_F/C will only flip the sign at the boundary.
        sgn_diag_F = sps.diags(sgn_nd.ravel("F"))
        sgn_diag_C = sps.diags(sgn_nd.ravel("C"))

        # Recall the ordering of the local equations:
        # First stress equilibrium for the internal subfaces.
        # Then the stress equilibrium for the Neumann subfaces.
        # Then the Robin subfaces.
        # And last, the displacement continuity on both internal and external subfaces.
        rhs_int = bound_exclusion.exclude_boundary(rhs_units)
        rhs_neu = bound_exclusion.keep_neumann(sgn_diag_C * rhs_units)
        rhs_rob = bound_exclusion.keep_robin(sgn_diag_C * rhs_units)

        num_dir_subface = (
            bound_exclusion.exclude_neu_rob.shape[1]
            - bound_exclusion.exclude_neu_rob.shape[0]
        )

        # No right hand side for cell displacement equations.
        rhs_units_displ_var = sps.coo_matrix(
            (nd * num_subfno - num_dir_subface, num_subfno_unique * nd)
        )

        # We get a pluss because the -n * I * alpha * p term is moved over to the rhs
        # in the local systems
        rhs_units = sps.vstack([rhs_int, rhs_neu, rhs_rob, rhs_units_displ_var])

        del rhs_units_displ_var

        # Output should be on cell-level (not sub-cell)
        sc2c = fvutils.map_sc_2_c(g.dim, sub_cell_index, cell_node_blocks[0])

        # prepare for computation of imbalance coefficients,
        # that is jumps in cell-centers pressures, ready to be
        # multiplied with inverse gradients
        rhs_jumps = rhs_units * unique_nAlpha_grad * sc2c

        # Step 3

        # mapping from subface to unique subface for vector problems.
        # This mapping gives the convention from which side
        # the force should be evaluated on.
        vals = np.ones(num_subfno_unique * nd)
        rows = fvutils.expand_indices_nd(subcell_topology.subfno_unique, nd)
        cols = fvutils.expand_indices_incr(
            subcell_topology.unique_subfno, nd, num_subhfno
        )
        map_unique_subfno = sps.coo_matrix(
            (vals, (rows, cols)), shape=(num_subfno_unique * nd, num_subhfno * nd)
        )

        del vals, rows, cols

        # Prepare for computation of -grad_p_face term
        # Note that sgn_diag_F might only flip the boundary signs. See comment above.
        grad_p_face = -sgn_diag_F * map_unique_subfno * nAlpha_grad * sc2c

        return rhs_jumps, grad_p_face

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
    def extract_vector(self, g, u, dims=None, as_vector=False):
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

    def extract_scalar(self, g, u):
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
        p = self.extract_scalar(g, u)
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
        d = self.extract_vector(g, u, as_vector=True)
        stress = np.squeeze(stress_discr * d) + (bound_stress * bound_val)
        return stress


class GradP:
    """ Class for the pressure gradient term of the Biot equation.
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
        mat_dict = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        if not "grad_p" in mat_dict:
            raise ValueError(
                """GradP class requires a pre-computed discretization to be
                             stored in the matrix dictionary."""
            )
        div_mech = fvutils.vector_divergence(g)
        # Put together linear system
        if mat_dict["grad_p"].shape[0] != g.dim * g.num_faces:
            hf2f_nd = pp.fvutils.map_hf_2_f(g=g)
            grad_p = hf2f_nd * mat_dict["grad_p"]
        else:
            grad_p = mat_dict["grad_p"]
        return div_mech * grad_p

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

    def assemble_int_bound_displacement_trace(
        self, g, data, data_edge, grid_swap, cc, matrix, rhs, self_ind
    ):
        """ Assemble the contribution from the pressure to the the trace of the
        displacement on internal boundaries.

        The intended use is when the internal boundary is coupled to another
        node in the GridBucket sense. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose displacement continuity on an interface.

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
            rhs (block_array 3x1): Right hand side contribution for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.

        """
        mg = data_edge["mortar_grid"]

        # TODO: this should become first or second or something
        if grid_swap:
            proj = mg.slave_to_mortar_avg()
        else:
            proj = mg.master_to_mortar_avg()

        # Expand indices as Fortran indexes
        proj_avg = sps.kron(proj, sps.eye(g.dim)).tocsr()

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        bp = matrix_dictionary["bound_displacement_pressure"]

        if proj_avg.shape[1] == g.dim * g.num_faces:
            # In this case we projection is from faces to cells.
            # The bound_displacement_pressure gives the pressure contribution to the
            # subface displacement. We therefore need to map it to faces.
            hf2f = pp.fvutils.map_hf_2_f(g=g)
            num_nodes = np.diff(g.face_nodes.indptr)
            weight = sps.kron(sps.eye(g.dim), sps.diags(1 / num_nodes))
            # hf2f adds all subface values to one face value. For the displacement we want
            # to take the average, therefore we divide each face by the number of subfaces.
            cc[2, self_ind] += proj_avg * weight * hf2f * bp
        else:
            cc[2, self_ind] += proj_avg * bp

    def enforce_neumann_int_bound(self, *_):
        pass


class DivU:
    """ Class for the displacement divergence term of the Biot equation.
    """

    def __init__(
        self,
        mechanics_keyword="mechanics",
        flow_keyword="flow",
        variable="displacement",
        mortar_variable="mortar_displacement",
    ):
        """ Set the mechanics keyword and specify the variables.

        The keywords are used to access and store parameters and discretization
        matrices.
        The variable names are used to obtain the previous solution for the time
        discretization. Consequently, they are those of the unknowns contributing to
        the DivU term (displacements), not the scalar variable.
        """
        self.flow_keyword = flow_keyword
        self.mechanics_keyword = mechanics_keyword
        # We also need to specify the names of the displacement variables on the node
        # and adjacent edges. T
        # Set variable name for the vector variable (displacement).
        self.variable = variable
        # The following is only used for mixed-dimensional problems.
        # Set the variable used for contact mechanics.
        self.mortar_variable = mortar_variable

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
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.flow_keyword]
        if not "div_u" in matrix_dictionary:
            raise ValueError(
                """DivU class requires a pre-computed discretization to be
                             stored in the matrix dictionary."""
            )
        biot_alpha = data[pp.PARAMETERS][self.flow_keyword]["biot_alpha"]
        return matrix_dictionary["div_u"] * biot_alpha

    def assemble_rhs(self, g, data):
        """ Return the right-hand side for a discretization of the displacement
        divergence term.

        For the time being, we assume an IE temporal discretization.


        Parameters:
            g (Grid): Computational grid.
            data (dictionary): With data stored.

        Returns:
            np.ndarray: Zero right hand side vector with representation of boundary
                conditions.
        """
        parameter_dictionary_mech = data[pp.PARAMETERS][self.mechanics_keyword]
        parameter_dictionary_flow = data[pp.PARAMETERS][self.flow_keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.flow_keyword]

        # For IE and constant BCs, the boundary part cancels, as the contribution from
        # successive timesteps (n and n+1) appear on the rhs with opposite signs. For
        # transient BCs, use the below with the appropriate version of d_bound_i.
        # Get bc values from mechanics
        d_bound_1 = parameter_dictionary_mech["bc_values"]

        d_bound_0 = data[pp.STATE][self.mechanics_keyword]["bc_values"]
        # and coupling parameter from flow
        biot_alpha = parameter_dictionary_flow["biot_alpha"]
        rhs_bound = (
            -matrix_dictionary["bound_div_u"] * (d_bound_1 - d_bound_0) * biot_alpha
        )

        # Time part
        d_cell = data[pp.STATE][self.variable]

        div_u = matrix_dictionary["div_u"]
        rhs_time = np.squeeze(biot_alpha * div_u * d_cell)

        return rhs_bound + rhs_time

    def assemble_int_bound_displacement_trace(
        self, g, data, data_edge, grid_swap, cc, matrix, rhs, self_ind
    ):
        """Assemble the contribution from the displacement mortar on an internal boundary,
        manifested as a displacement boundary condition.

        The intended use is when the internal boundary is coupled to another
        node by an interface law. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose the effect of the displacement mortar on the divergence term on
        the higher-dimensional grid.

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
        # Projection operators to grid
        mg = data_edge["mortar_grid"]

        if grid_swap:
            proj = mg.mortar_to_slave_avg(nd=g.dim)
        else:
            proj = mg.mortar_to_master_avg(nd=g.dim)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.flow_keyword]
        biot_alpha = data[pp.PARAMETERS][self.flow_keyword]["biot_alpha"]
        bound_div_u = matrix_dictionary["bound_div_u"]

        u_bound_previous = data_edge[pp.STATE][self.mortar_variable]

        if bound_div_u.shape[1] != proj.shape[0]:
            raise ValueError(
                """Inconsistent shapes. Did you define a
            sub-face boundary condition but only a face-wise mortar?"""
            )
        # The mortar will act as a boundary condition for the div_u term.
        # We assume implicit Euler in Biot, thus the div_u term appears
        # on the rhs as div_u^{k-1}. This results in a contribution to the
        # rhs for the coupling variable also.
        cc[self_ind, 2] += biot_alpha * bound_div_u * proj
        rhs[self_ind] += biot_alpha * bound_div_u * proj * u_bound_previous

    def assemble_int_bound_displacement_source(
        self, g, data, data_edge, cc, matrix, rhs, self_ind
    ):
        """Assemble the contribution from the displacement mortar on an internal boundary,
        manifested as a source term. Only the normal component of the mortar displacement
        is considered.

        The intended use is when the internal boundary is coupled to another
        node by an interface law. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose the effect of the displacement mortar on the divergence term on
        the lower-dimensional grid.

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

        mg = data_edge["mortar_grid"]

        # From the mortar displacements, we want to
        # 1) Take the jump between the two mortar sides,
        # 2) Project to the slave grid and
        # 3) Extract the normal component.

        # Define projections and rotations
        nd = g.dim + 1
        proj = mg.mortar_to_slave_avg(nd=nd)
        jump_on_slave = proj * mg.sign_of_mortar_sides(nd=nd)
        rotation = data_edge["tangential_normal_projection"]
        normal_component = rotation.project_normal(g.num_cells)

        biot_alpha = data[pp.PARAMETERS][self.flow_keyword]["biot_alpha"]
#        aperture = data[pp.PARAMETERS][self.flow_keyword]["aperture"]
        if biot_alpha != 1:
            warnings.warn(
                "Are you sure you want a non-unitary biot alpha for the fracture?"
            )

        # Project the previous solution to the slave grid
        previous_displacement_jump_global_coord = (
            jump_on_slave * data_edge[pp.STATE][self.mortar_variable]
        )
        # Rotated displacement jumps. These are in the local coordinates, on
        # the lower-dimensional grid
        previous_displacement_jump_normal = (
            normal_component * previous_displacement_jump_global_coord
        )
        # The same procedure is applied to the unknown displacements, by assembling the
        # jump operator, projection and normal component extraction in the coupling matrix.
        # Finally, we integrate over the cell volume.
        # The jump on the slave is defined to be negative for an open fracture (!),
        # hence the negative sign.
        vol = sps.dia_matrix((g.cell_volumes, 0), shape=(g.num_cells, g.num_cells))
        cc[self_ind, 2] -= biot_alpha * vol * normal_component * jump_on_slave

        # We assume implicit Euler in Biot, thus the div_u term appears
        # on the rhs as div_u^{k-1}. This results in a contribution to the
        # rhs for the coupling variable also.
        # See note above on sign. This term is on the rhs, yielding the opposite sign.
        rhs[self_ind] += biot_alpha * vol * previous_displacement_jump_normal

    def enforce_neumann_int_bound(self, *_):
        pass


class BiotStabilization(
    pp.numerics.interface_laws.elliptic_discretization.EllipticDiscretization
):
    """ Class for the stabilization term of the Biot equation.
    """

    def __init__(self, keyword="mechanics", variable="pressure"):
        """ Set the two keywords.

        The keywords are used to access and store parameters and discretization
        matrices.
        """
        super().__init__(keyword)
        # Set variable name for the scalar variable (pressure)
        self.variable = variable

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
            """No discretize method implemented for the DivU
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
        """ Return the right-hand side for the stabilization part of the displacement
        divergence term.

        For the time being, we assume an IE temporal discretization.


        Parameters:
            g (Grid): Computational grid.
            data (dictionary): With data stored.

        Returns:
            np.ndarray: Zero right hand side vector with representation of boundary
                conditions.
        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        # The stabilization is the pressure contribution to the div u part of the
        # fluid mass conservation, thus needs a right hand side in the implicit Euler
        # discretization.
        pressure_0 = data[pp.STATE][self.variable]
        A_stability = matrix_dictionary["biot_stabilization"]
        rhs_time = A_stability * pressure_0

        # The stabilization has no rhs.
        rhs_bound = np.zeros(self.ndof(g))

        return rhs_bound + rhs_time
