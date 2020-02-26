"""
For details on the conditions discretized herein, see

Berge et al., 2019: Finite volume discretization for poroelastic media with fractures
modeled by contact mechanics.

When solving contact problems, the sought fracture displacement (jumps) are defined
relative to an initial state. For transient problems, this initial state is the solution
at the previous time step. The state should be available in

    d[pp.STATE][self.mortar_displacement_variable],

and may usually be set to zero for stationary problems. The ColoumbContact
discretization operates on relative tangential jumps and absolute normal jumps.
See also contact_mechanics_interface_laws.py
"""
import numpy as np

import porepy as pp
import logging

logger = logging.getLogger(__name__)


class ColoumbContact:
    def __init__(self, keyword, ambient_dimension, discr_h):
        self.keyword = keyword

        self.dim = ambient_dimension

        self.mortar_displacement_variable = "mortar_u"
        self.contact_variable = "contact_traction"

        self.traction_discretization = "traction_discretization"
        self.displacement_discretization = "displacement_discretization"
        self.rhs_discretization = "contact_rhs"

        self.discr_h = discr_h

    def _key(self):
        return self.keyword + "_"

    def _discretization_key(self):
        return self._key() + pp.keywords.DISCRETIZATION

    def ndof(self, g):
        return g.num_cells * self.dim

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        """ Discretize the contact conditions using a semi-smooth Newton
        approach.

        The function relates the contact forces, represented on the
        lower-dimensional grid, to the jump in displacement between the two
        adjacent mortar grids. The function provides a (linearized)
        disrcetizaiton of the contact conditions, as described in Berge et al.

        The discertization is stated in the coordinate system defined by the
        projection operator associated with the surface. The contact forces
        should be interpreted as tangential and normal to this plane.

        NOTES:
            Quantities stated in the global coordinate system (e.g.
        displacements on the adjacent mortar grids) must be projected to the
        local system, using the same projection operator, when paired with the
        produced discretization (that is, in the global assembly).
            There is a numerical parameter c_num. The sensitivity is currently
        unknown.

        Assumptions and other noteworthy aspects:  TODO: Rewrite this when the
        implementation is ready.
            * The contact surface is planar, so that all cells on the surface can
            be described by a single normal vector.
            * The contact forces are represented directly in the local
            coordinate system of the surface. The first self.dim - 1 elements
            of the contact vector are the tangential components of the first
            cell, then the normal component, then tangential of the second cell
            etc.

        """

        # CLARIFICATIONS NEEDED:
        #   1) Do projection and rotation commute on non-matching grids? The
        #   gut feel says yes, but I'm not sure.

        # Process input
        parameters_l = data_l[pp.PARAMETERS]
        friction_coefficient = parameters_l[self.keyword]["friction_coefficient"]

        cohesion = parameters_l[self.keyword].get("cohesion", 0)

        if np.asarray(friction_coefficient).size == 1:
            friction_coefficient = friction_coefficient * np.ones(g_l.num_cells)

        mg = data_edge["mortar_grid"]

        # Numerical parameter, value and sensitivity is currently unknown.
        # The thesis of Huber is probably a good place to look for information.
        c_num = parameters_l[self.keyword].get(
            "contact_mechanics_numerical_parameter", 100
        )

        # In an attempt to reduce the sensitivity of the numerical parameter on the
        # model parameters, we scale it with area and an order-of-magnitude estimate
        # for the elastic moduli.
        area = g_l.cell_volumes

        parameters_h = data_h[pp.PARAMETERS][self.discr_h.keyword]
        constit_h = parameters_h["fourth_order_tensor"]
        mean_constit = (
            mg.mortar_to_slave_avg()
            * mg.master_to_mortar_avg()
            * 0.5
            * np.abs(g_h.cell_faces * (constit_h.mu + constit_h.lmbda))
        )

        c_num_normal = c_num * mean_constit * area
        c_num_tangential = c_num * mean_constit * area

        # The tractions are scaled with area, so do the same with the cohesion.
        scaled_cohesion = cohesion * area

        # TODO: Implement a single method to get the normal vector with right sign
        # thus the right local coordinate system.

        # Pick the projection operator (defined elsewhere) for this surface.
        # IMPLEMENATION NOTE: It is paramount that this projection is used for all
        # operations relating to this surface, or else directions of normal vectors
        # will get confused.
        projection = data_edge["tangential_normal_projection"]

        # The contact force is already computed in local coordinates
        contact_force = data_l[pp.STATE]["previous_iterate"][self.contact_variable]

        # Pick out the tangential and normal direction of the contact force.
        # The contact force of the first cell is in the first self.dim elements
        # of the vector, second cell has the next self.dim etc.
        # By design the tangential force is the first self.dim-1 components of
        # each cell, while the normal force is the last component.
        normal_indices = np.arange(self.dim - 1, contact_force.size, self.dim)
        tangential_indices = np.setdiff1d(np.arange(contact_force.size), normal_indices)
        contact_force_normal = contact_force[normal_indices]
        contact_force_tangential = contact_force[tangential_indices].reshape(
            (self.dim - 1, g_l.num_cells), order="F"
        )

        # The displacement jump (in global coordinates) is found by switching the
        # sign of the second mortar grid, and then sum the displacements on the
        # two sides (which is really a difference since one of the sides have
        # its sign switched).
        # The tangential displacements are relative to the initial state, which in the
        # transient case equals the previous time step.
        previous_displacement_iterate = data_edge[pp.STATE]["previous_iterate"][
            self.mortar_displacement_variable
        ]
        previous_displacement_time = data_edge[pp.STATE][
            self.mortar_displacement_variable
        ]
        displacement_jump_global_coord_iterate = (
            mg.mortar_to_slave_avg(nd=self.dim)
            * mg.sign_of_mortar_sides(nd=self.dim)
            * previous_displacement_iterate
        )
        displacement_jump_global_coord_time = (
            mg.mortar_to_slave_avg(nd=self.dim)
            * mg.sign_of_mortar_sides(nd=self.dim)
            * previous_displacement_time
        )
        # Rotated displacement jumps. These are in the local coordinates, on
        # the lower-dimensional grid. For the normal direction, we consider the absolute
        # displacement, not that relative to the initial state.
        displacement_jump_normal = projection.project_normal(g_l.num_cells) * (
            displacement_jump_global_coord_iterate
        )
        # The jump in the tangential direction is in g_l.dim columns, one per
        # dimension in the tangential direction. For the tangential direction, we
        # consider the relative displacement.
        displacement_jump_tangential = (
            projection.project_tangential(g_l.num_cells)
            * (
                displacement_jump_global_coord_iterate
                - displacement_jump_global_coord_time
            )
        ).reshape((self.dim - 1, g_l.num_cells), order="F")

        # The friction bound is computed from the previous state of the contact
        # force and normal component of the displacement jump.
        # Note that the displacement jump is rotated before adding to the contact force
        friction_bound = (
            friction_coefficient
            * np.clip(
                -contact_force_normal + c_num_normal * displacement_jump_normal,
                0,
                np.inf,
            )
            + scaled_cohesion
        )

        num_cells = friction_coefficient.size

        # Find contact and sliding region

        # Contact region is determined from the normal direction.
        penetration_bc = self._penetration(
            contact_force_normal, displacement_jump_normal, c_num_normal
        )
        # Check criterion for sliding
        sliding_criterion = self._sliding(
            contact_force_tangential,
            displacement_jump_tangential,
            friction_bound,
            c_num_tangential,
        )
        # Find cells with non-zero tangential traction. This excludes cells that were
        # not in contact in the previous iteration.
        non_zero_tangential_traction = (
            np.sum(contact_force_tangential ** 2, axis=0) > 1e-10
        )

        # The discretization of the sliding state tacitly assumes that the tangential
        # traction in the previous iteration - or else we will divide by zero.
        # Therefore, only allow for sliding if the tangential traciton is non-zero.
        # In practice this means that a cell is not allowed to go directly from
        # non-penetration to sliding.
        sliding_bc = np.logical_and(sliding_criterion, non_zero_tangential_traction)

        # Structures for storing the computed coefficients.
        displacement_weight = []  # Multiplies displacement jump
        traction_weight = []  # Multiplies the normal forces
        rhs = np.array([])  # Goes to the right hand side.

        # Zero vectors of the size of the tangential space and the full space,
        # respectively. These are needed to complement the discretization
        # coefficients to be determined below.
        zer = np.array([0] * (self.dim - 1))
        zer1 = np.array([0] * (self.dim))
        zer1[-1] = 1

        # Loop over all mortar cells, discretize according to the current state of
        # the contact
        # The loop computes three parameters:
        # L will eventually multiply the displacement jump, and be associated with
        #   the coefficient in a Robin boundary condition (using the terminology of
        #   the mpsa implementation)
        # r is the right hand side term
        # IS: Comment about the traction weight?

        for i in range(num_cells):
            if sliding_bc[i] & penetration_bc[i]:  # in contact and sliding
                # This is Eq (31) in Berge et al, including the regularization
                # described in (32) and onwards. The expressions are somewhat complex,
                # and are therefore moved to subfunctions.
                loc_displacement_tangential, r, v = self._sliding_coefficients(
                    contact_force_tangential[:, i],
                    displacement_jump_tangential[:, i],
                    friction_bound[i],
                    c_num_tangential[i],
                )

                # There is no interaction between displacement jumps in normal and
                # tangential direction
                L = np.hstack((loc_displacement_tangential, np.atleast_2d(zer).T))
                loc_displacement_weight = np.vstack((L, zer1))
                # Right hand side is computed from (24-25). In the normal
                # direction, zero displacement is enforced.
                # This assumes that the original distance, g, between the fracture
                # walls is zero.
                r = np.vstack((r + friction_bound[i] * v, 0))
                # Unit contribution from tangential force
                loc_traction_weight = np.eye(self.dim)
                # Zero weight on normal force
                loc_traction_weight[-1, -1] = 0
                # Contribution from normal force
                # NOTE: The sign is different from Berge (31); the paper is wrong
                loc_traction_weight[:-1, -1] = -friction_coefficient[i] * v.ravel()

            elif ~sliding_bc[i] & penetration_bc[i]:  # In contact and sticking
                # Weight for contact force computed according to (30) in Berge.
                # NOTE: There is a sign error in the paper, the coefficient for the
                # normal contact force should have a minus in front of it
                loc_traction_tangential = (
                    -friction_coefficient[i]  # The minus sign is correct
                    * displacement_jump_tangential[:, i].ravel("F")
                    / friction_bound[i]
                )
                # Unit coefficient for all displacement jumps
                loc_displacement_weight = np.eye(self.dim)

                # Tangential traction dependent on normal one
                loc_traction_weight = np.zeros((self.dim, self.dim))
                loc_traction_weight[:-1, -1] = loc_traction_tangential

                # The right hand side is the previous tangential jump, and zero
                # in the normal direction.
                r = np.hstack((displacement_jump_tangential[:, i], 0)).T

            elif ~penetration_bc[i]:  # not in contact
                # This is a free boundary, no conditions on displacement
                loc_displacement_weight = np.zeros((self.dim, self.dim))

                # Free boundary conditions on the forces.
                loc_traction_weight = np.eye(self.dim)
                r = np.zeros(self.dim)

            else:  # should never happen
                raise AssertionError("Should not get here")

            # Depending on the state of the system, the weights in the tangential direction may
            # become huge or tiny compared to the other equations. This will
            # impede convergence of an iterative solver for the linearized
            # system. As a partial remedy, rescale the condition to become
            # closer to unity.
            w_diag = np.diag(loc_displacement_weight) + np.diag(loc_traction_weight)
            W_inv = np.diag(1 / w_diag)
            loc_displacement_weight = W_inv.dot(loc_displacement_weight)
            loc_traction_weight = W_inv.dot(loc_traction_weight)
            r = r.ravel() / w_diag

            # Append to the list of global coefficients.
            displacement_weight.append(loc_displacement_weight)
            traction_weight.append(loc_traction_weight)
            rhs = np.hstack((rhs, r))

        num_blocks = len(traction_weight)
        data_traction = np.array(traction_weight).ravel(order="C")

        data_displacement = np.array(displacement_weight).ravel(order="C")

        data_l[pp.DISCRETIZATION_MATRICES][self.keyword][
            self.traction_discretization
        ] = pp.utils.sparse_mat.csr_matrix_from_blocks(
            data_traction, self.dim, num_blocks
        )
        data_l[pp.DISCRETIZATION_MATRICES][self.keyword][
            self.displacement_discretization
        ] = pp.utils.sparse_mat.csr_matrix_from_blocks(
            data_displacement, self.dim, num_blocks
        )
        data_l[pp.DISCRETIZATION_MATRICES][self.keyword][self.rhs_discretization] = rhs

        # Also store the contact state
        data_l[pp.STATE]["previous_iterate"]["penetration"] = penetration_bc
        data_l[pp.STATE]["previous_iterate"]["sliding"] = sliding_bc

    def assemble_matrix_rhs(self, g, data):
        # Generate matrix for the coupling. This can probably be generalized
        # once we have decided on a format for the general variables
        traction_coefficient = data[pp.DISCRETIZATION_MATRICES][self.keyword][
            self.traction_discretization
        ]
        displacement_coefficient = data[pp.DISCRETIZATION_MATRICES][self.keyword][
            self.displacement_discretization
        ]

        rhs = data[pp.DISCRETIZATION_MATRICES][self.keyword][self.rhs_discretization]

        return traction_coefficient, displacement_coefficient, rhs

    # Active and inactive boundary faces
    def _sliding(self, Tt, ut, bf, ct):
        """ Find faces where the frictional bound is exceeded, that is, the face is
        sliding.

        Arguments:
            Tt (np.array, nd-1 x num_faces): Tangential forces.
            u_hat (np.array, nd-1 x num_faces): Displacements in tangential
                direction.
            bf (np.array, num_faces): Friction bound.
            ct (double): Numerical parameter that relates displacement jump to
                tangential forces. See Huber et al for explanation.

        Returns:
            boolean, size num_faces: True if |-Tt + ct*ut| > bf for a face

        """
        # Use thresholding to not pick up faces that are just about sticking
        # Not sure about the sensitivity to the tolerance parameter here.

        tol = 1e-8
        tol = 1e-6
        #     print("sliding")
        #     print(self._l2(-Tt + ct * ut) - bf)
        return self._l2(-Tt + ct * ut) - bf > tol

    def _penetration(self, Tn, un, cn):
        """ Find faces that are in contact.

        Arguments:
            Tn (np.array, num_faces): Normal forces.
            un (np.array, num_faces): Displament in normal direction.
            cn (double): Numerical parameter that relates displacement jump to
                normal forces. See Huber et al for explanation.

        Returns:
            boolean, size num_faces: True if |-Tu + cn*un| > 0 for a face

        """
        # Not sure about the sensitivity to the tolerance parameter here.
        tol = 1e-6
        #  print("penetration")
        #  print(-Tn + cn * un)
        return (-Tn + cn * un) > tol

    #####
    ## Below here are different help function for calculating the Newton step
    #####

    def _e(self, Tt, cut, bf):
        # Compute part of (32) in Berge et al.
        return bf / self._l2(-Tt + cut)

    def _Q(self, Tt, cut, bf):
        # Implementation of the term Q involved in the calculation of (32) in Berge
        # et al.
        # This is the regularized Q
        numerator = -Tt.dot((-Tt + cut).T)

        # Regularization to avoid issues during the iterations to avoid dividing by
        # zero if the faces are not in contact durign iterations.
        denominator = max(bf, self._l2(-Tt)) * self._l2(-Tt + cut)

        return numerator / denominator

    def _M(self, Tt, cut, bf):
        """ Compute the coefficient M used in Eq. (32) in Berge et al.
        """
        Id = np.eye(Tt.shape[0])
        # M = e * (I - Q)
        return self._e(Tt, cut, bf) * (Id - self._Q(Tt, cut, bf))

    def _hf(self, Tt, cut, bf):
        # This is the product e * Q * (-Tt + cut), used in computation of r in (32)
        return self._e(Tt, cut, bf) * self._Q(Tt, cut, bf).dot(-Tt + cut)

    def _sliding_coefficients(self, Tt, ut, bf, c):
        """
        Compute the regularized versions of coefficients L, v and r, defined in
        Eq. (32) and section 3.2.1 in Berge et al.

        Arguments:
            Tt: Tangential forces. np array, two or three elements
            ut: Tangential displacement. Same size as Tt
            bf: Friction bound for this mortar cell.
            c: Numerical parameter

        """
        if Tt.ndim <= 1:
            Tt = np.atleast_2d(Tt).T
            ut = np.atleast_2d(ut).T

        cut = c * ut
        # Identity matrix
        Id = np.eye(Tt.shape[0])

        # Shortcut if the friction coefficient is effectively zero.
        # Numerical tolerance here is likely somewhat arbitrary.
        if bf <= 1e-3:
            return (
                0 * Id,
                bf * np.ones((Id.shape[0], 1)),
                (-Tt + cut) / self._l2(-Tt + cut),
            )

        # Compute the coefficient M
        coeff_M = self._M(Tt, cut, bf)

        # Regularization during the iterations requires computations of parameters
        # alpha, beta, delta
        alpha = -Tt.T.dot(-Tt + cut) / (self._l2(-Tt) * self._l2(-Tt + cut))

        # Parameter delta.
        # NOTE: The denominator bf is correct. The definition given in Berge is wrong.
        delta = min(self._l2(-Tt) / bf, 1)

        if alpha < 0:
            beta = 1 / (1 - alpha * delta)
        else:
            beta = 1

        # The expression (I - beta * M)^-1
        # NOTE: In the definition of \tilde{L} in Berge, the inverse on the inner
        # paranthesis is missing.
        IdM_inv = np.linalg.inv(Id - beta * coeff_M)

        v = IdM_inv.dot(-Tt + cut) / self._l2(-Tt + cut)

        return c * (IdM_inv - Id), -IdM_inv.dot(self._hf(Tt, cut, bf)), v

    def _l2(self, x):
        x = np.atleast_2d(x)
        return np.sqrt(np.sum(x ** 2, axis=0))


def set_projections(gb):
    """ Define a local coordinate system, and projection matrices, for all
    grids of co-dimension 1.

    The function adds one item to the data dictionary of all GridBucket edges
    that neighbors a co-dimension 1 grid, defined as:
        key: tangential_normal_projection, value: pp.TangentialNormalProjection
            provides projection to the surface of the lower-dimensional grid

    Note that grids of co-dimension 2 and higher are ignored in this construction,
    as we do not plan to do contact mechanics on these objects.

    It is assumed that the surface is planar.

    """
    # Information on the vector normal to the surface is not available directly
    # from the surface grid (it could be constructed from the surface geometry,
    # which spans the tangential plane). We instead get the normal vector from
    # the adjacent higher dimensional grid.
    # We therefore access the grids via the edges of the mixed-dimensional grid.
    for e, d_m in gb.edges():

        mg = d_m["mortar_grid"]
        # Only consider edges where the lower-dimensional neighbor is of co-dimension 1
        if not mg.dim == (gb.dim_max() - 1):
            continue

        # Neigboring grids
        _, g_h = gb.nodes_of_edge(e)

        # Find faces of the higher dimensional grid that coincide with the mortar
        # grid. Go via the master to mortar projection
        # Convert matrix to csr, then the relevant face indices are found from
        # the (column) indices
        faces_on_surface = mg.master_to_mortar_int().tocsr().indices

        # Find out whether the boundary faces have outwards pointing normal vectors
        # Negative sign implies that the normal vector points inwards.
        sgn = g_h.sign_of_faces(faces_on_surface)

        # Unit normal vector
        unit_normal = g_h.face_normals[: g_h.dim] / g_h.face_areas
        # Ensure all normal vectors on the relevant surface points outwards
        unit_normal[:, faces_on_surface] *= sgn

        # Now we need to pick out *one*  normal vector of the higher dimensional grid
        # which coincides with this mortar grid. This could probably have been
        # done with face tags, but we instead project the normal vectors onto the
        # mortar grid to kill off all irrelevant faces. Restriction to a single
        # normal vector is done in the construction of the projection object
        # (below).
        # NOTE: Use a single normal vector to span the tangential and normal space,
        # thus assuming the surface is planar.
        outwards_unit_vector_mortar = mg.master_to_mortar_int().dot(unit_normal.T).T

        # NOTE: The normal vector is based on the first cell in the mortar grid,
        # and will be pointing from that cell towards the other side of the
        # mortar grid. This defines the positive direction in the normal direction.
        # Although a simpler implementation seems to be possible, going via the
        # first element in faces_on_surface, there is no guarantee that this will
        # give us a face on the positive (or negative) side, hence the more general
        # approach is preferred.
        #
        # NOTE: The basis for the tangential direction is determined by the
        # construction internally in TangentialNormalProjection.
        projection = pp.TangentialNormalProjection(
            outwards_unit_vector_mortar[:, 0].reshape((-1, 1))
        )

        # Store the projection operator in the mortar data
        d_m["tangential_normal_projection"] = projection
