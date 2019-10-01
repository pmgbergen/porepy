"""
Implementation of contact conditions for fracture mechanics, using a primal formulation.

We provide a class for coupling the higher-dimensional mechanical discretization to the
tractions on the fractures. Also, in the case of coupled physics (Biot and the like),
classes handling the arising coupling terms are provided.
"""

import numpy as np
import scipy.sparse as sps
import logging
import time

import porepy as pp

logger = logging.getLogger(__name__)


class PrimalContactCoupling(object):
    """ Implement the coupling conditions for the pure mechanics problem.

    The primary variables for this formulation are displacement in the ambient dimension,
    displacements at the boundary of the highest dimensional grid (represented as mortar
    variables), and contact forces on grids of co-dimension 1.

    The conditions represented here are
        1) KKT condition for the traction / displacement in the normal direction.
        2) Conditions for the tangential traction / displacement, according
           to whether the fracture is sliding, sticking or free.
        3) Linear elasticity on the surface displacements, with the tangential contact
           force as a driving force.
        4) The mortar displacements act as Dirichlet boundary conditions for the
           higher-dimensional domain.

    When solving contact problems, the sought fracture displacement (jumps) are defined
    relative to an initial state. For transient problems, this initial state is the
    solution at the previous time step. The state should be available in
        d[pp.STATE][self.mortar_displacement_variable],
    and may usually be set to zero for stationary problems.
    See also contact_conditions.py
    """

    def __init__(self, keyword, discr_master, discr_slave, use_surface_discr=False):
        self.keyword = keyword
        self.mortar_displacement_variable = "mortar_u"
        self.discr_master = discr_master
        self.discr_slave = discr_slave

        self.SURFACE_DISCRETIZATION_KEY = "surface_smoother"

        self.use_surface_discr = use_surface_discr

    def _key(self):
        return self.keyword + "_"

    def _discretization_key(self):
        return self._key() + pp.keywords.DISCRETIZATION

    def ndof(self, mg):
        """ Get the number of dof for this coupling.

        It is assumed that this method will only be called for mortar grids of
        co-dimension 1. If the assumption is broken, this will not work.
        """
        return (mg.dim + 1) * mg.num_cells

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):

        tic = time.time()
        logging.debug("Discretize contact mechanics interface law")
        # Discretize the surface PDE
        parameter_dictionary_edge = data_edge[pp.PARAMETERS][self.keyword]
        matrix_dictionary_edge = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]

        mg = data_edge["mortar_grid"]

        # Projection onto the tangential space of the mortar grid

        # Tangential_normal projection
        tangential_normal_projection = data_edge["tangential_normal_projection"]

        normal_projection = tangential_normal_projection.project_normal()

        # The right hand side of the normal diffusion considers only the tangential part
        # of the normal forces.
        matrix_dictionary_edge["contact_force_map"] = normal_projection

        # Keyword to control if the surface discretization should be rediscretized
        # It is a linear term, so we may save time during Newton iterations here
        discretize_surface = parameter_dictionary_edge.get("discretize_surface", True)

        if self.use_surface_discr and discretize_surface:
            # Discretize the surface pde if asked for.

            # Lame parameters to be used for discretizing the surface elliptic equation.
            mu = parameter_dictionary_edge["mu"]
            lmbda = parameter_dictionary_edge["lambda"]

            # Parameter used when mapping surface grids to their lower-dimensional planes.
            # This is necessary for the mapping function, but at this point in the
            # simulation workflow, it should not really be an issue.
            deviation_from_plane_tol = 1e-5

            # List of surface diffusion discretizations - one per side.
            A_list = []

            for _, side_grid in mg.project_to_side_grids():

                unity = np.ones(side_grid.num_cells)

                # Create an finite volume discretization for elasticity.
                # Define parameters for the surface diffusion in an appropriate form.
                mpsa = pp.Mpsa(self.keyword)

                # The stiffness matrix is istropic, thus we need not care about the
                # basis used for mapping grid coordinates into the tangential space.
                # Simply define the parameters directly in 2d space.
                stiffness = pp.FourthOrderTensor(
                    side_grid.dim, mu * unity, lmbda * unity
                )

                bc = pp.BoundaryConditionVectorial(side_grid)

                mpsa_parameters = pp.initialize_data(
                    side_grid,
                    {},
                    self.keyword,
                    {"fourth_order_tensor": stiffness, "bc": bc},
                )

                # Project the side grid into its natural dimension.
                g = side_grid.copy()
                # Use the same projection matrix as in the projections used on the
                # variables.
                rot = tangential_normal_projection.projection[:, :, 0]
                if rot.shape == (2, 2):
                    rot = np.vstack((np.hstack((rot, np.zeros((2, 1)))), np.zeros((3))))
                cell_centers, face_normals, face_centers, _, _, nodes = pp.map_geometry.map_grid(
                    g, deviation_from_plane_tol, R=rot
                )
                g.cell_centers = cell_centers
                g.face_normals = face_normals
                g.face_centers = face_centers
                g.nodes = nodes

                mpsa.discretize(g, mpsa_parameters)

                # We are only interested in the elasticity discretization as a smoother.
                # Construct the discretiation matrix, and disregard all other output.
                A_loc = (
                    pp.fvutils.vector_divergence(side_grid)
                    * mpsa_parameters[pp.DISCRETIZATION_MATRICES][self.keyword][
                        "stress"
                    ]
                )

                # The local discretization must be mapped to the full mortar degrees of freedom.
                # This entails a projection onto the normal plane, followed by a restriction to this
                # side grid

                # Projection to remove degrees of freedom in the normal direction to the grid
                # This should be used after the projection to the tangent space,
                # when we know which rows are
                tangential_projection = tangential_normal_projection.project_tangential(
                    side_grid.num_cells
                )
                A_list.append(A_loc * tangential_projection)

            # Concatenate discretization matrices
            A = sps.block_diag([mat for mat in A_list])

            # The discretization is still a non-square matrix, it needs to be expanded to
            # be compatible with the block assembler.
            # The final equations should relate to continuity of the normal froces
            matrix_dictionary_edge[self.SURFACE_DISCRETIZATION_KEY] = A

        # Discretization of the contact mechanics is done by a ColumbContact
        # object.
        # The resulting equations are located at the lower-dimensional grid,
        # however, the discretization is inherently linked to the mortar grid.
        # It is therefore constructed here.

        self.discr_slave.discretize(g_h, g_l, data_h, data_l, data_edge)

        logger.debug("Done. Elapsed time {}".format(time.time() - tic))

    def assemble_matrix_rhs(
        self, g_master, g_slave, data_master, data_slave, data_edge, matrix
    ):

        """ Assemble the dicretization of the interface law, and its impact on
        the neighboring domains.
        Parameters:
            g_master: Grid on one neighboring subdomain.
            g_slave: Grid on the other neighboring subdomain.
            data_master: Data dictionary for the master suddomain
            data_slave: Data dictionary for the slave subdomain.
            data_edge: Data dictionary for the edge between the subdomains
            matrix: original discretization matrix, to which the coupling terms will be
                added.

        """
        matrix_dictionary_edge = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]

        ambient_dimension = g_master.dim

        master_ind = 0
        slave_ind = 1
        mortar_ind = 2

        # Generate matrix for the coupling. This can probably be generalized
        # once we have decided on a format for the general variables
        mg = data_edge["mortar_grid"]
        projection = data_edge["tangential_normal_projection"]

        dof_master = self.discr_master.ndof(g_master)
        dof_slave = self.discr_slave.ndof(g_slave)

        if not dof_master == matrix[master_ind, master_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the master discretization given
            in RobinCoupling must match the number of dofs given by the matrix
            """
            )
        elif not dof_slave == matrix[master_ind, slave_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the slave discretization given
            in RobinCoupling must match the number of dofs given by the matrix
            """
            )
        elif not mg.num_cells * ambient_dimension == matrix[master_ind, 2].shape[1]:
            raise ValueError(
                """The number of dofs of the edge discretization given
            in the PrimalContactCoupling must match the number of dofs given by the matrix
            """
            )

        # We know the number of dofs from the master and slave side from their
        # discretizations
        #        dof = np.array([dof_master, dof_slave, mg.num_cells])
        dof = np.array(
            [
                matrix[master_ind, master_ind].shape[1],
                matrix[slave_ind, slave_ind].shape[1],
                mg.num_cells,
            ]
        )
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((3, 3))

        rhs = np.empty(3, dtype=np.object)
        rhs[master_ind] = np.zeros(dof_master)
        rhs[slave_ind] = np.zeros(dof_slave)
        rhs[mortar_ind] = np.zeros(mg.num_cells * ambient_dimension)

        # IMPLEMENTATION NOTE: The current implementation is geared towards
        # using mpsa for the mechanics problem. A more general approach would
        # be possible - for an example see the flow problem with the RobinCoupling
        # and EllipticDiscretization and its subclasses. However, at present such a general
        # framework currently seems over the top, hence this more mundane approach.

        ### Equation for the master side
        # The mortar variable acts as a Dirichlet boundary condition for the master.
        master_bound_stress = data_master[pp.DISCRETIZATION_MATRICES][
            self.discr_master.keyword
        ]["bound_stress"]
        master_stress = data_master[pp.DISCRETIZATION_MATRICES][
            self.discr_master.keyword
        ]["stress"]
        master_bc_values = data_master[pp.PARAMETERS][self.discr_master.keyword][
            "bc_values"
        ]
        master_divergence = pp.fvutils.vector_divergence(g_master)

        # The mortar variable (boundary displacement) takes the form of a Dirichlet
        # condition for the master side. The MPSA convention is to have
        # - div * bound_stress * bc_values
        # on the rhs. Accordingly, the contribution from the mortar variable (boundary
        # displacement) on the left hand side is positive:
        # div * bound_stress * u_mortar
        cc[master_ind, mortar_ind] = (
            master_divergence
            * master_bound_stress
            * mg.mortar_to_master_avg(nd=ambient_dimension)
        )

        ### Equation for the slave side
        #
        # These are the contact conditions, which dictate relations between
        # the contact forces on the slave, and the displacement jumps.
        #
        # NOTE: Both the contact conditions and the contact stresses are defined in the
        # local coordinate system of the surface. The displacements must therefore
        # be rotated to this local coordinate system during assembly.
        traction_discr, displacement_jump_discr, rhs_slave = self.discr_slave.assemble_matrix_rhs(
            g_slave, data_slave
        )
        # The contact forces. Can be applied directly, these are in their own
        # local coordinate systems.
        cc[slave_ind, slave_ind] = traction_discr

        # The contact condition discretization gives coefficients for the mortar
        # variables. To finalize the relation with the contact conditions, we
        # (from the right) 1) assign +- signs to the two sides of the mortar, so that
        # summation in reality is a difference, 2) project to the mortar grid
        # 3) project to the local coordinates of the fracture, 4) assign the
        # coefficients of the displacement jump.
        cc[slave_ind, mortar_ind] = (
            displacement_jump_discr
            * projection.project_tangential_normal(g_slave.num_cells)
            * mg.mortar_to_slave_avg(nd=ambient_dimension)
            * mg.sign_of_mortar_sides(nd=ambient_dimension)
        )

        # Right hand side system. In the local (surface) coordinate system.
        # For transient simulations where the tangential velocity, not displacement, is
        # considered, a term arises on the rhs from the previous time step.
        previous_time_step_displacements = data_edge[pp.STATE][
            self.mortar_displacement_variable
        ].copy()
        rotated_jumps = (
            projection.project_tangential_normal(g_slave.num_cells)
            * mg.mortar_to_slave_avg(nd=ambient_dimension)
            * mg.sign_of_mortar_sides(nd=ambient_dimension)
            * previous_time_step_displacements
        )
        rhs_u = displacement_jump_discr * rotated_jumps
        # Only tangential velocity is considered. Zero out all normal components, as we
        # operate on absolute, not relative, normal jumps.
        rhs_u[(ambient_dimension - 1) :: ambient_dimension] = 0
        rhs[slave_ind] = rhs_slave + rhs_u

        ### Equation for the mortar rows

        # This is first a stress balance: stress from the higher dimensional
        # domain (both interior and bound_stress) should match with the contact stress:
        #
        #     traction_slave + traction_master = 0
        #
        # Optionally, a diffusion term can be added in the tangential direction
        # of the stresses, this is currently under implementation.

        # A diagonal operator is needed to switch the sign of vectors on
        # higher-dimensional faces that point into the fracture surface. The effect is to
        # switch direction of the stress on boundary for the higher dimensional domain: The
        # contact forces are defined as negative in contact, whereas the sign of the higher
        # dimensional stresses are defined according to the direction of the normal vector.
        faces_on_fracture_surface = mg.master_to_mortar_int().tocsr().indices
        sign_switcher = pp.grid_utils.switch_sign_if_inwards_normal(
            g_master, ambient_dimension, faces_on_fracture_surface
        )

        ## First, we obtain T_master = stress * u_master + bound_stress * u_mortar
        # Stress contribution from the higher dimensional domain, projected onto
        # the mortar grid
        # Switch the direction of the vectors to obtain the traction as defined
        # by the outwards pointing normal vector.
        traction_from_master = (
            mg.master_to_mortar_int(nd=ambient_dimension)
            * sign_switcher
            * master_stress
        )
        cc[mortar_ind, master_ind] = traction_from_master
        # Stress contribution from boundary conditions.
        rhs[mortar_ind] = -(
            mg.master_to_mortar_int(nd=ambient_dimension)
            * sign_switcher
            * master_bound_stress
            * master_bc_values
        )
        # The stress contribution from the mortar variables, mapped to the higher
        # dimensional domain via a boundary condition, and back again by a
        # projection operator.
        # Switch the direction of the vectors, so that for all faces, a positive
        # force points into the fracture surface.
        traction_from_mortar = (
            mg.master_to_mortar_int(nd=ambient_dimension)
            * sign_switcher
            * master_bound_stress
            * mg.mortar_to_master_avg(nd=ambient_dimension)
        )
        cc[mortar_ind, mortar_ind] = traction_from_mortar

        ## Second, the contact stress is mapped to the mortar grid.
        # We have for the positive (first) and negative (second) side of the mortar that
        # T_slave = T_master_pos = -T_master_neg,
        # so we need to map the slave traction with the corresponding signs to match the
        # mortar tractions.

        # The contact forces are defined in the surface coordinate system.
        # Map to the mortar grid, and rotate back again to the global coordinates
        # (note the inverse rotation is given by a transpose).
        # Finally, the contact stresses will be felt in different directions by
        # the two sides of the mortar grids (Newton's third law), hence
        # adjust the signs
        contact_traction_to_mortar = (
            mg.sign_of_mortar_sides(nd=ambient_dimension)
            * projection.project_tangential_normal(mg.num_cells).T
            * mg.slave_to_mortar_int(nd=ambient_dimension)
        )
        # Minus to obtain -T_slave + T_master = 0.
        cc[mortar_ind, slave_ind] = -contact_traction_to_mortar

        if self.use_surface_discr:
            restrict_to_tangential_direction = projection.project_tangential(
                mg.num_cells
            )

            # The first block contains the surface diffusion component. This has
            # the surface diffusion operator for the mortar variables, and a
            # mapping of contact forces on the slave variables.
            # The second block gives continuity of forces in the normal direction.
            surface_discr = matrix_dictionary_edge[self.SURFACE_DISCRETIZATION_KEY]

            cc[mortar_ind, mortar_ind] += (
                restrict_to_tangential_direction.T * surface_discr
            )

        matrix += cc

        return matrix, rhs


class MatrixScalarToForceBalance:
    """
    This class adds the matrix scalar (pressure) contribution to the force balance posed
    on the mortar grid by PrimalContactCoupling.

    We account for the scalar variable contribution to the forces on the higher-dimensional
    internal boundary, i.e. the last term of:

        boundary_traction_hat = stress * u_hat + bound_stress * u_mortar + gradP * p_hat

    Note that with this approach to discretization of the boundary pressure force, it
    will only be included for nonzero values of the biot_alpha coefficient.

    If the scalar is e.g. pressure, subtraction of the pressure contribution is needed:

        T_contact - p_check I \dot n = boundary_traction_hat

    This is taken care of by FracturePressureToForceBalance.

    """

    def __init__(self, keyword, discr_master, discr_slave):
        """
        Parameters:
            keyword used for storage of the gradP discretization. If the GradP class is
                used, this is the keyword associated with the mechanical parameters.
            discr_master and
            discr_slave are the discretization objects operating on the master and slave
                pressure, respectively. Used for #DOFs. In FV, one cell variable is
                expected.
        """
        # Set node discretizations
        self.discr_master = discr_master
        self.discr_slave = discr_slave
        # Keyword used to retrieve gradP discretization.
        self.keyword = keyword

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        """
        Nothing to do
        """
        pass

    def assemble_matrix_rhs(
        self, g_master, g_slave, data_master, data_slave, data_edge, matrix
    ):
        """
        Assemble the pressure contributions of the interface force balance law.

        Parameters:
            g_master: Grid on one neighboring subdomain.
            g_slave: Grid on the other neighboring subdomain.
            data_master: Data dictionary for the master suddomain
            data_slave: Data dictionary for the slave subdomain.
            data_edge: Data dictionary for the edge between the subdomains
            matrix: original discretization matrix, to which the coupling terms will be
                added.
        """

        ambient_dimension = g_master.dim

        master_ind = 0
        slave_ind = 1
        mortar_ind = 2

        # Generate matrix for the coupling. This can probably be generalized
        # once we have decided on a format for the general variables
        mg = data_edge["mortar_grid"]

        dof_master = self.discr_master.ndof(g_master)
        dof_slave = self.discr_slave.ndof(g_slave)

        if not dof_master == matrix[master_ind, master_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the master discretization given
            in RobinCoupling must match the number of dofs given by the matrix
            """
            )
        elif not dof_slave == matrix[master_ind, slave_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the slave discretization given
            in RobinCoupling must match the number of dofs given by the matrix
            """
            )
        elif not mg.num_cells * ambient_dimension == matrix[master_ind, 2].shape[1]:
            raise ValueError(
                """The number of dofs of the edge discretization given
            in the PrimalContactCoupling must match the number of dofs given by the matrix
            """
            )

        # We know the number of dofs from the master and slave side from their
        # discretizations
        dof = np.array(
            [
                matrix[master_ind, master_ind].shape[1],
                matrix[slave_ind, slave_ind].shape[1],
                mg.num_cells * ambient_dimension,
            ]
        )
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((3, 3))

        rhs = np.empty(3, dtype=np.object)
        rhs[master_ind] = np.zeros(dof_master)
        rhs[slave_ind] = np.zeros(dof_slave)
        rhs[mortar_ind] = np.zeros(mg.num_cells * ambient_dimension)

        master_scalar_gradient = data_master[pp.DISCRETIZATION_MATRICES][self.keyword][
            "grad_p"
        ]

        # We want to modify the stress balance posed on the edge to account for the
        # scalar (usually pressure) contribution.
        # In the purely mechanical case, stress from the higher dimensional
        # domain (both interior and bound_stress) should match the contact stress:
        # -T_slave + T_master = 0,
        # see PrimalContactCoupling.
        # The following modification is needed:
        # Add the scalar gradient contribution to the traction on the master
        # boundary.

        # A diagonal operator is needed to switch the sign of vectors on
        # higher-dimensional faces that point into the fracture surface, see
        # PrimalContactCoupling.
        faces_on_fracture_surface = mg.master_to_mortar_int().tocsr().indices
        sign_switcher = pp.grid_utils.switch_sign_if_inwards_normal(
            g_master, ambient_dimension, faces_on_fracture_surface
        )

        # i) Obtain pressure stress contribution from the higher dimensional domain.
        # ii) Switch the direction of the vectors, so that for all faces, a positive
        # force points into the fracture surface (along the outwards normal on the
        # boundary).
        # iii) Map to the mortar grid.
        # iv) Minus according to - alpha grad p already in the discretization matrix
        master_scalar_to_master_traction = (
            mg.master_to_mortar_int(nd=ambient_dimension)
            * sign_switcher
            * master_scalar_gradient
        )
        cc[mortar_ind, master_ind] = master_scalar_to_master_traction

        matrix += cc

        return matrix, rhs


class FractureScalarToForceBalance:
    """
    This class adds the fracture pressure contribution to the force balance posed on the
    mortar grid by PrimalContactCoupling and modified to account for matrix pressure by
    MatrixPressureToForceBalance.

    For the contact mechanics, we only want to consider the _contact_ traction. Thus, we
    have to subtract the pressure contribution, i.e.

        T_contact - p_check I \dot n = boundary_traction_hat,

    since the full tractions experienced by a fracture surface are the sum of the
    contact forces and the fracture pressure force.

    """

    def __init__(self, discr_master, discr_slave):
        """
        Parameters:
            keyword used for storage of the gradP discretization. If the GradP class is
                used, this is the keyword associated with the mechanical parameters.
            discr_master and
            discr_slave are the discretization objects operating on the master and slave
                pressure, respectively. Used for #DOFs. In FV, one cell variable is
                expected.
        """
        # Set node discretizations
        self.discr_master = discr_master
        self.discr_slave = discr_slave

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        """
        Nothing to do
        """
        pass

    def assemble_matrix_rhs(
        self, g_master, g_slave, data_master, data_slave, data_edge, matrix
    ):
        """
        Assemble the pressure contributions of the interface force balance law.

        Parameters:
            g_master: Grid on one neighboring subdomain.
            g_slave: Grid on the other neighboring subdomain.
            data_master: Data dictionary for the master suddomain
            data_slave: Data dictionary for the slave subdomain.
            data_edge: Data dictionary for the edge between the subdomains
            matrix: original discretization matrix, to which the coupling terms will be
                added.
        """

        ambient_dimension = g_master.dim

        master_ind = 0
        slave_ind = 1
        mortar_ind = 2

        # Generate matrix for the coupling. This can probably be generalized
        # once we have decided on a format for the general variables
        mg = data_edge["mortar_grid"]

        dof_master = self.discr_master.ndof(g_master)
        dof_slave = self.discr_slave.ndof(g_slave)

        if not dof_master == matrix[master_ind, master_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the master discretization given
            in RobinCoupling must match the number of dofs given by the matrix
            """
            )
        elif not dof_slave == matrix[master_ind, slave_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the slave discretization given
            in RobinCoupling must match the number of dofs given by the matrix
            """
            )
        elif not mg.num_cells * ambient_dimension == matrix[master_ind, 2].shape[1]:
            raise ValueError(
                """The number of dofs of the edge discretization given
            in the PrimalContactCoupling must match the number of dofs given by the matrix
            """
            )

        # We know the number of dofs from the master and slave side from their
        # discretizations
        dof = np.array(
            [
                matrix[master_ind, master_ind].shape[1],
                matrix[slave_ind, slave_ind].shape[1],
                mg.num_cells * ambient_dimension,
            ]
        )
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((3, 3))

        rhs = np.empty(3, dtype=np.object)
        rhs[master_ind] = np.zeros(dof_master)
        rhs[slave_ind] = np.zeros(dof_slave)
        rhs[mortar_ind] = np.zeros(mg.num_cells * ambient_dimension)

        ## Ensure that the contact variable is only the force from the contact of the
        # two sides of the fracture. This requires subtraction of the pressure force.

        # Construct the dot product between normals on fracture faces and the identity
        # matrix. Similar sign switching as above is needed (this one operating on
        # fracture faces only).
        faces_on_fracture_surface = mg.master_to_mortar_int().tocsr().indices
        sgn = g_master.sign_of_faces(faces_on_fracture_surface)
        fracture_normals = g_master.face_normals[
            :ambient_dimension, faces_on_fracture_surface
        ]
        outwards_fracture_normals = sgn * fracture_normals

        data = outwards_fracture_normals.ravel("F")
        row = np.arange(g_master.dim * mg.num_cells)
        col = np.tile(np.arange(mg.num_cells), (g_master.dim, 1)).ravel("F")
        n_dot_I = sps.csc_matrix((data, (row, col)))
        # i) The scalar contribution to the contact stress is mapped to the mortar grid
        # and multiplied by n \dot I, with n being the outwards normals on the two sides.
        # Note that by using different normals for the two sides, we do not need to
        # adjust the slave pressure with the corresponding signs by applying
        # sign_of_mortar_sides as done in PrimalContactCoupling.
        # iii) The contribution should be subtracted so that we balance the master
        # forces by
        # T_contact - n dot I p,
        # hence the minus.
        slave_pressure_to_contact_traction = -(n_dot_I * mg.slave_to_mortar_int(nd=1))
        # Minus to obtain -T_slave + T_master = 0, i.e. from placing the two
        # terms on the same side of the equation, as also done in PrimalContactCoupling.
        cc[mortar_ind, slave_ind] = -slave_pressure_to_contact_traction

        matrix += cc

        return matrix, rhs


class DivUCoupling:
    """
    Coupling conditions for DivU term.

    For mixed-dimensional flow in coupled to matrix mechanics, i.e. Biot in the matrix
    and conservation of a scalar quantity (usually fluid mass) in matrix and fractures.
    We have assumed a primal displacement mortar variable, which will contribute
    to the div u term in fracture ("div aperture") and matrix.
    """

    def __init__(self, variable, discr_master, discr_slave):
        # Set variable names for the vector variable on the nodes (displacement), used
        # to access solutions from previous time steps.
        self.variable = variable
        # The terms are added by calls to assemble methods of DivU discretizations,
        # namely assemble_int_bound_displacement_trace for the master and
        self.discr_master = discr_master
        # assemble_int_bound_displacement_source for the slave.
        self.discr_slave = discr_slave

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        """
        Nothing to do
        """
        pass

    def assemble_matrix_rhs(
        self, g_master, g_slave, data_master, data_slave, data_edge, matrix
    ):
        """
        Assemble the mortar displacement's contribution as a internal Dirichlet
        contribution for the higher dimension, and source term for the lower dimension.
        Parameters:
            g_master: Grid on one neighboring subdomain.
            g_slave: Grid on the other neighboring subdomain.
            data_master: Data dictionary for the master suddomain
            data_slave: Data dictionary for the slave subdomain.
            data_edge: Data dictionary for the edge between the subdomains
            matrix: original discretization matrix, to which the coupling terms will be
                added.
        """
        ambient_dimension = g_master.dim

        master_ind = 0
        slave_ind = 1
        mortar_ind = 2

        # Generate matrix for the coupling. This can probably be generalized
        # once we have decided on a format for the general variables
        mg = data_edge["mortar_grid"]

        dof_master = self.discr_master.ndof(g_master)
        dof_slave = self.discr_slave.ndof(g_slave)

        if not dof_master == matrix[master_ind, master_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the master discretization given
            in RobinCoupling must match the number of dofs given by the matrix
            """
            )
        elif not dof_slave == matrix[master_ind, slave_ind].shape[1]:
            raise ValueError(
                """The number of dofs of the slave discretization given
            in RobinCoupling must match the number of dofs given by the matrix
            """
            )
        elif not mg.num_cells * ambient_dimension == matrix[master_ind, 2].shape[1]:
            raise ValueError(
                """The number of dofs of the edge discretization given
            in the PrimalContactCoupling must match the number of dofs given by the matrix
            """
            )

        # We know the number of dofs from the master and slave side from their
        # discretizations
        dof = np.array(
            [
                matrix[master_ind, master_ind].shape[1],
                matrix[slave_ind, slave_ind].shape[1],
                mg.num_cells * ambient_dimension,
            ]
        )
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((3, 3))
        rhs = np.empty(3, dtype=np.object)
        rhs[master_ind] = np.zeros(dof_master)
        rhs[slave_ind] = np.zeros(dof_slave)
        rhs[mortar_ind] = np.zeros(mg.num_cells * ambient_dimension)

        grid_swap = False
        # Let the DivU class assemble the contribution from mortar to master
        self.discr_master.assemble_int_bound_displacement_trace(
            g_master, data_master, data_edge, grid_swap, cc, matrix, rhs, master_ind
        )
        # and from mortar to slave.
        self.discr_slave.assemble_int_bound_displacement_source(
            g_slave, data_slave, data_edge, cc, matrix, rhs, slave_ind
        )
        matrix += cc

        return matrix, rhs
