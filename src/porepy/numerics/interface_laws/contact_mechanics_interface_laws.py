#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of contact conditions for fracture mechanics, using a primal formulation.


The primal formulation is conceptually similar, but mathematically different from,
the dual formulation, currently located in elliptic_interface_laws.
"""

import numpy as np
import scipy.sparse as sps

import porepy as pp


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

    """

    def __init__(self, keyword, discr_master, discr_slave):
        self.keyword = keyword

        self.discr_master = discr_master
        self.discr_slave = discr_slave

        self.SURFACE_DISCRETIZATION_KEY = "surface_smoother"

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

        # Discretize the surface PDE
        parameter_dictionary_edge = data_edge[pp.PARAMETERS][self.keyword]
        matrix_dictionary_edge = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]

        mg = data_edge["mortar_grid"]

        # Projection onto the tangential space of the mortar grid

        # Tangential_normal projection
        tangential_normal_projection = data_edge["tangential_normal_projection"]

        normal_projection = tangential_normal_projection.project_normal()

        discretize_surface = parameter_dictionary_edge.get("discretize_surface", True)

        # The right hand side of the normal diffusion considers only the tangential part
        # of the normal forces.
        matrix_dictionary_edge["contact_force_map"] = normal_projection

        if discretize_surface:

            # Lame parameters to be used for discretizing the surface elliptic equation.
            mu = parameter_dictionary_edge["mu"]
            lmbda = parameter_dictionary_edge["lambda"]

            deviation_from_plane_tol = 1e-5

            A_list = []

            for proj, side_grid in mg.project_to_side_grids():

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
                cell_centers, face_normals, face_centers, _, _, nodes = pp.cg.map_grid(
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

        # Discretization of the contact mechanics is done externally.
        # The resulting equations are located at the lower-dimensional grid,
        # however, the discretization is inherently linked to the mortar grid.
        # It is therefore constructed here.

        self.discr_slave.discretize(g_h, g_l, data_h, data_l, data_edge)

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
            matrix_master: original discretization for the master subdomain
            matrix_slave: original discretization for the slave subdomain

        """
        matrix_dictionary_edge = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]

        if not "surface_smoother" in matrix_dictionary_edge:
            self.discretize(g_master, g_slave, data_master, data_slave, data_edge)

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
        master_divergence = pp.fvutils.vector_divergence(g_master)

        proj_vector_master = mg.mortar_to_master_avg(nd=ambient_dimension)

        sign_of_mortar_variables = mg.sign_of_mortar_sides(nd=ambient_dimension)

        # The mortar variable (boundary displacement) takes the form of a Dirichlet
        # condition for the master side.
        cc[master_ind, mortar_ind] = (
            master_divergence * master_bound_stress * proj_vector_master
        )

        ### Equation for the slave side
        # These are the contact conditions, which dictate relations between
        # the contact forces on the slave, and the displacement jumps.
        # NOTE: Some more projcetions may be needed here for non-matching grids
        traction_discr, displacement_jump_discr, rhs_slave = self.discr_slave.assemble_matrix_rhs(
            g_slave, data_slave
        )
        cc[slave_ind, slave_ind] = traction_discr
        cc[slave_ind, mortar_ind] = displacement_jump_discr
        rhs[slave_ind] = rhs_slave

        ## Equation for the mortar side.
        # The first nd-1 sets of equations are formed by the surface pde, which
        # acts in the tangential direction of the mortar grid.
        # The final equation is continuity of forces in the normal direction.
        stress_from_mortar = (
            mg.master_to_mortar_int(nd=ambient_dimension) * master_bound_stress
        )
        stress_from_master = (
            mg.master_to_mortar_int(nd=ambient_dimension) * master_stress
        )

        contact_stress_to_mortar = mg.slave_to_mortar_int(nd=ambient_dimension)

        projection = data_edge["tangential_normal_projection"]
        restrict_to_tangential_direction = projection.project_tangential(mg.num_cells)
        restrict_to_normal_direction = projection.project_normal(mg.num_cells)

        ## Equations for the mortars, in a block system:
        # The first block contains the surface diffusion component. This has
        # the surface diffusion operator for the mortar variables, and a
        # mapping of contact forces on the slave variables.
        # The second block gives continuity of forces in the normal direction.
        surface_discr = matrix_dictionary_edge[self.SURFACE_DISCRETIZATION_KEY]

        # Master component: No contribution to the surface diffusion equation.
        # For the normal component, use the discretization of stress from the
        # master side.
        cc[mortar_ind, master_ind] = sps.bmat(
            [
                [
                    sps.csc_matrix(
                        (mg.num_cells * mg.dim, self.discr_master.ndof(g_master))
                    )
                ],
                [restrict_to_normal_direction * stress_from_master],
            ]
        )
        # Contribution from the slave variable (contact forces).
        # The stresses are mapped to the mortar grid, restricted to the tangential
        # and normal direction (for surface diffusion and force continuity, respectively)
        # and the signs are flipped on one side of the fracture (Newton's third law).
        cc[mortar_ind, slave_ind] = sps.bmat(
            [
                [
                    mg.sign_of_mortar_sides(ambient_dimension - 1)
                    * restrict_to_tangential_direction
                    * contact_stress_to_mortar
                ],
                [
                    mg.sign_of_mortar_sides(1)
                    * restrict_to_normal_direction
                    * contact_stress_to_mortar
                ],
            ]
        )
        # Finally the mortar variables: The first block is the surface diffusion.
        # The second block (normal forces) translates the mortar variable into
        # stresses in master, mapped back again to the mortar (see definition of
        # stress_from_mortar) and then restricted to normal direction.
        cc[mortar_ind, mortar_ind] = sps.bmat(
            [
                [surface_discr],
                [
                    restrict_to_normal_direction
                    * stress_from_mortar
                    * proj_vector_master
                ],
            ]
        )

        matrix += cc

        return matrix, rhs
