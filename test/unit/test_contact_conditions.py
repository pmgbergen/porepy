"""
Unit tests for discretization of contact conditions. Specifically, the tests verify that
the coefficients in front of mortar displacement and contact forces are correctly
calculated for a set of different fracture states (open, contact and sticking, contact
and sliding). Moreover, the projection of contact forces onto the mortar grids are also
verified. Tests are set up for different angles of the fracture compared to the
xy axis, and for different directions of the normal and tangential vectors of the
local coordinate systems in the fracture.

Main shortcoming: Only 2d is considered. It should be simple, but tedious, to set
up a similar problem with a 3d medium consisting of two cells (might be Cartesian).

"""

import unittest

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.grids import mortar_grid
from porepy.models.contact_mechanics_model import ContactMechanics


class ContactConditionColoumb2d(unittest.TestCase):
    """Workflow for the tests:
    Each of the test_* methods first set up a rotation angle for the fracture
    and a previous state for the normal and tangential jumps and the contact
    force. Also the direction of the tangential and normal vectors for the
    local (to the fracture) coordinate system are set. The coefficients in
    the discretization of contact condition and the map from contact forces to
    forces on the mortar grid are then computed in two ways:
        i) Via the ContactCondition discretization class
        ii) With hard coding of the coefficients in the paper describing the
            contact discretization, using the current state and the problem
            geometry
    Errors are raised if the values are not the same.

    """

    def setUp(self):
        # Initialize variables for known values to zeros. Will be modified subsequently
        # by other tests.
        self.known_Acm = np.zeros((2, 4))
        self.known_Amc = np.zeros((4, 2))
        self.known_Acc = np.zeros((2, 2))
        self.known_bc = np.zeros(2)

    def verify(self, model):
        A_mc, A_cm, A_cc, b_c, penetration, sliding = model.get_matrices()
        # print(A_mc, self.known_Amc)
        self.assertTrue(np.allclose(A_mc, self.known_Amc))
        # if not np.allclose(A_cm, self.known_Acm):
        print(A_cm, self.known_Acm)
        self.assertTrue(np.allclose(A_cm, self.known_Acm))
        self.assertTrue(np.allclose(A_cc, self.known_Acc))
        self.assertTrue(np.allclose(b_c, self.known_bc))

        self.assertTrue(np.allclose(penetration, self.known_penetration))
        self.assertTrue(np.allclose(sliding, self.known_sliding))

    def _set_known_contact_to_mortar(self, angle, model):
        # Coefficient in matrix that maps contact forces to the mortar space.
        # The main ingredient is a rotation from tangential-normal space to the global
        # coordinates.
        # The coefficients are independent on the contact state, that is, they
        # are functions of rotation angle only.
        # Also make variables for the direction of the tangential and normal
        # vectors of the fracture.

        # Recover the projection matrix used in the mapping between local and global
        # coordinates.
        # The tangent vector is in the first row, normal in the second
        proj = model.gb.node_props(model.g1)["tangential_normal_projection"].projection[
            :, :, 0
        ]

        # The normal vector is considered positive if it points in the y-direction.
        self.pos_normal = proj[1, 1] > 0
        # Sign convention for tangent vector
        self.pos_tangent = proj[0, 0] > 0
        # Also store the projection matrix for good measure, not clear if we need it
        self.proj = proj

        # short hand for sine and cosine for the rotation angle
        c = np.cos(angle)
        s = np.sin(angle)

        # Matrix, to be filled out
        mat = np.zeros((2, 2))

        # To see the the logic in matrix elements, draw an xy-coordinate system, and
        # a tn-system, where the t-axis is rotation 'angle' in the CCW direction
        # compared to the x-axis. Note that the matrix elements have their sign shifted
        # if the tangential or normal direction is not in the positive direction.
        if self.pos_tangent:
            mat[0, 0] = c
            mat[1, 0] = s
        else:
            mat[0, 0] = -c
            mat[1, 0] = -s

        if self.pos_normal:
            mat[0, 1] = -s
            mat[1, 1] = c
        else:
            mat[0, 1] = s
            mat[1, 1] = -c

        # The full projection matrix is found by stacking two copies of mat, with a
        # sign change on the second copy, to reflect Newton's third law
        # This step corresponds to the jump operator / mg.sign_of_mortar_sides
        full_mat = np.vstack((-mat, mat))

        # Store information
        self.known_Amc = full_mat

    """ Below are tests for no penetration conditions. In this case, the coefficients
    for contact are very simple - the contact force is zero. The non-trivial aspect to
    be tested in this case is the projection of contact forces up to the mortars. This
    one we test for both positive and negative rotation angles. The sign of the angle
    should probably not make a difference, but better safe than sorry.
    """

    def _set_coefficients_no_penetration(self):

        # diagonal matrix for the contact force. zero coefficients for the displacements
        self.known_Acc = np.array([[1, 0], [0, 1]])
        self.known_bc[:] = 0

        self.known_penetration = np.array([False])
        self.known_sliding = np.array([False])

    def test_pos_rot_no_penetration_pos_tangent_pos_normal(self):
        # In contact, no sliding. Tangent in positive direction
        self.setUp()

        # Randomly tilted geometry
        angle = (np.pi / 2 * np.random.rand(1))[0]

        u_mortar = np.zeros(4)
        contact_force = np.array([0, 1])
        model = ContactModel2d(angle, u_mortar, contact_force, True, True)

        self._set_known_contact_to_mortar(angle, model)
        self._set_coefficients_no_penetration()
        self.verify(model)

    def test_pos_rot_no_penetration_neg_tangent_pos_normal(self):
        # In contact, no sliding. Tangent in positive direction
        self.setUp()

        # Randomly tilted geometry
        angle = (np.pi / 2 * np.random.rand(1))[0]

        u_mortar = np.zeros(4)
        contact_force = np.array([0, 1])
        model = ContactModel2d(angle, u_mortar, contact_force, False, True)

        self._set_known_contact_to_mortar(angle, model)
        self._set_coefficients_no_penetration()
        self.verify(model)

    def test_pos_rot_no_penetration_pos_tangent_neg_normal(self):
        # In contact, no sliding. Tangent in positive direction
        self.setUp()

        # Randomly tilted geometry
        angle = (np.pi / 2 * np.random.rand(1))[0]

        u_mortar = np.zeros(4)
        contact_force = np.array([0, 1])
        model = ContactModel2d(angle, u_mortar, contact_force, True, False)

        self._set_known_contact_to_mortar(angle, model)
        self._set_coefficients_no_penetration()
        self.verify(model)

    def test_pos_rot_no_penetration_neg_tangent_neg_normal(self):
        # In contact, no sliding. Tangent in positive direction
        self.setUp()

        # Randomly tilted geometry
        angle = (np.pi / 2 * np.random.rand(1))[0]

        u_mortar = np.zeros(4)
        contact_force = np.array([0, 1])
        model = ContactModel2d(angle, u_mortar, contact_force, True, True)

        self._set_known_contact_to_mortar(angle, model)
        self._set_coefficients_no_penetration()
        self.verify(model)

    def test_neg_rot_no_penetration_pos_tangent_pos_normal(self):
        # In contact, no sliding. Tangent in positive direction
        self.setUp()

        # Randomly tilted geometry
        angle = (-np.pi / 2 * np.random.rand(1))[0]

        u_mortar = np.zeros(4)
        contact_force = np.array([0, 1])
        model = ContactModel2d(angle, u_mortar, contact_force, True, True)

        self._set_known_contact_to_mortar(angle, model)
        self._set_coefficients_no_penetration()
        self.verify(model)

    def test_neg_rot_no_penetration_neg_tangent_pos_normal(self):
        # In contact, no sliding. Tangent in positive direction
        self.setUp()

        # Randomly tilted geometry
        angle = (-np.pi / 2 * np.random.rand(1))[0]

        u_mortar = np.zeros(4)
        contact_force = np.array([0, 1])
        model = ContactModel2d(angle, u_mortar, contact_force, False, True)

        self._set_known_contact_to_mortar(angle, model)
        self._set_coefficients_no_penetration()
        self.verify(model)

    def test_neg_rot_no_penetration_pos_tangent_neg_normal(self):
        # In contact, no sliding. Tangent in positive direction
        self.setUp()

        # Randomly tilted geometry
        angle = (-np.pi / 2 * np.random.rand(1))[0]

        u_mortar = np.zeros(4)
        contact_force = np.array([0, 1])
        model = ContactModel2d(angle, u_mortar, contact_force, True, False)

        self._set_known_contact_to_mortar(angle, model)
        self._set_coefficients_no_penetration()
        self.verify(model)

    def test_neg_rot_no_penetration_neg_tangent_neg_normal(self):
        # In contact, no sliding. Tangent in positive direction
        self.setUp()

        # Randomly tilted geometry
        angle = (-np.pi / 2 * np.random.rand(1))[0]

        u_mortar = np.zeros(4)
        contact_force = np.array([0, 1])
        model = ContactModel2d(angle, u_mortar, contact_force, True, True)

        self._set_known_contact_to_mortar(angle, model)
        self._set_coefficients_no_penetration()
        self.verify(model)

    """ Tests for contact, but sticking.
    """

    def _set_coefficients_contact_sticking(self, angle, u_mortar, contact_force):
        # _set_coefficients_sticking()

        # In addition, we need the coefficients for the normal contact force
        # The tangential force still has a zero coefficient

        # Coefficients set in the model. These should really be set explicitly here,
        # to be in control over any changes
        friction_coefficient = 1
        cnum = 100

        c = np.cos(angle)
        s = np.sin(angle)

        # Projection of xy displacemnet jump onto the normal direction
        u_jump_normal = (
            -s * u_mortar[2] + c * u_mortar[3] - (-s * u_mortar[0] + c * u_mortar[1])
        )

        if not self.pos_normal:
            u_jump_normal *= -1

        u_jump_tangential = (
            c * u_mortar[2] + s * u_mortar[3] - (c * u_mortar[0] + s * u_mortar[1])
        )
        if not self.pos_tangent:
            u_jump_tangential *= -1

        # Definition of the friction bound from Berge et al
        friction_bound = friction_coefficient * (
            -contact_force[1] - cnum * u_jump_normal
        )

        # Test that the initial conditions for the discretization are set so that the
        # fracture is in contact and sticking
        # This is essentially a test that the conditions are consistent with the
        # part of the discretization that is meant to be tested.
        # If these fail, so will comparison with known and computed penetration and
        # sliding states.

        # The condition for open fracture should not be fulfilled
        self.assertTrue(friction_bound >= 0)
        # The condition for contact and sticking should be fulfilled
        self.assertTrue(
            np.abs(-contact_force[0] - cnum * u_jump_tangential) < friction_bound
        )

        # Coefficient set according to (30) in Berge et al
        self.known_Acc[0, 1] = (
            -friction_coefficient * u_jump_tangential / friction_bound
        )
        self.known_bc[0] = u_jump_tangential

        # Coefficient
        # This is the same computation as in the above definition of
        # u_jump_tangential
        self.known_Acm[0, 0] = -c
        self.known_Acm[0, 1] = -s
        self.known_Acm[0, 2] = c
        self.known_Acm[0, 3] = s
        if not self.pos_tangent:
            self.known_Acm[0] *= -1

        # Normal direction: coefficients are the projection of the jump onto the normal
        # direction
        self.known_Acm[1, 0] = s
        self.known_Acm[1, 1] = -c
        self.known_Acm[1, 2] = -s
        self.known_Acm[1, 3] = c
        if not self.pos_normal:
            self.known_Acm[1] *= -1

        self.known_penetration = np.array([True])
        self.known_sliding = np.array([False])

    def test_zero_angle_penetration_sticking(self):
        # In contact, nonzero tangential jump, no sliding.
        # This is the simplest setup for the sticking state.
        self.setUp()

        # The tangential direction coincides with the global x-direction
        # normal direction with global y-direction
        angle = 0
        c = np.cos(angle)
        s = np.sin(angle)

        # Tangential jump in previous iteration
        tj = 0.1

        # The tangential jump should be projected back to the xy-coordinates
        # Divide by two to distribute jump on the two sides of the mortar grid
        u_mortar = 0.5 * np.array([tj * c, tj * s, -tj * c, -tj * s])
        # Make contact force sufficiently strong to in effect always be in contact
        contact_force = np.array([0, -10000])
        model = ContactModel2d(angle, u_mortar, contact_force, True, True)

        self._set_known_contact_to_mortar(angle, model)

        self._set_coefficients_contact_sticking(angle, u_mortar, contact_force)

        self.verify(model)

    def test_rot_penetration_sticking_pos_tangential_pos_normal(self):
        # Rotation, given configuration for normal and tangential directions
        self.setUp()

        # The tangential direction coincides with the global x-direction
        # normal direction with global y-direction
        angle = ((np.pi / 6) * np.random.rand(1))[0]
        c = np.cos(angle)
        s = np.sin(angle)

        # Tangential jump in previous iteration
        tj = 0.1

        # The tangential jump should be projected back to the xy-coordinates
        # Divide by two to distribute jump on the two sides of the mortar grid
        u_mortar = 0.5 * np.array([tj * c, tj * s, -tj * c, -tj * s])
        # Make contact force sufficiently strong to in effect always be in contact
        contact_force = np.array([0, -10000])
        model = ContactModel2d(angle, u_mortar, contact_force, True, True)

        self._set_known_contact_to_mortar(angle, model)

        self._set_coefficients_contact_sticking(angle, u_mortar, contact_force)
        self.verify(model)

    def test_rot_penetration_sticking_pos_tangential_neg_normal(self):
        # Rotation, given configuration for normal and tangential directions
        self.setUp()

        # The tangential direction coincides with the global x-direction
        # normal direction with global y-direction
        angle = ((np.pi / 6) * np.random.rand(1))[0]
        c = np.cos(angle)
        s = np.sin(angle)

        # Tangential jump in previous iteration
        tj = 0.1

        # The tangential jump should be projected back to the xy-coordinates
        # Divide by two to distribute jump on the two sides of the mortar grid
        u_mortar = 0.5 * np.array([tj * c, tj * s, -tj * c, -tj * s])
        # Make contact force sufficiently strong to in effect always be in contact
        contact_force = np.array([0, -10000])
        model = ContactModel2d(angle, u_mortar, contact_force, True, False)

        self._set_known_contact_to_mortar(angle, model)

        self._set_coefficients_contact_sticking(angle, u_mortar, contact_force)
        self.verify(model)

    def test_rot_penetration_sticking_neg_tangential_pos_normal(self):
        # Rotation, given configuration for normal and tangential directions
        self.setUp()

        # The tangential direction coincides with the global x-direction
        # normal direction with global y-direction
        angle = ((np.pi / 6) * np.random.rand(1))[0]
        c = np.cos(angle)
        s = np.sin(angle)

        # Tangential jump in previous iteration
        tj = 0.1

        # The tangential jump should be projected back to the xy-coordinates
        # Divide by two to distribute jump on the two sides of the mortar grid
        u_mortar = 0.5 * np.array([tj * c, tj * s, -tj * c, -tj * s])
        # Make contact force sufficiently strong to in effect always be in contact
        contact_force = np.array([0, -10000])
        model = ContactModel2d(angle, u_mortar, contact_force, False, True)

        self._set_known_contact_to_mortar(angle, model)

        self._set_coefficients_contact_sticking(angle, u_mortar, contact_force)
        self.verify(model)

    def test_rot_penetration_sticking_neg_tangential_neg_normal(self):
        # Rotation, given configuration for normal and tangential directions
        self.setUp()

        # The tangential direction coincides with the global x-direction
        # normal direction with global y-direction
        angle = ((np.pi / 6) * np.random.rand(1))[0]
        c = np.cos(angle)
        s = np.sin(angle)

        # Tangential jump in previous iteration
        tj = 0.1

        # The tangential jump should be projected back to the xy-coordinates
        # Divide by two to distribute jump on the two sides of the mortar grid
        u_mortar = 0.5 * np.array([tj * c, tj * s, -tj * c, -tj * s])
        # Make contact force sufficiently strong to in effect always be in contact
        contact_force = np.array([0, -10000])
        model = ContactModel2d(angle, u_mortar, contact_force, False, False)

        self._set_known_contact_to_mortar(angle, model)

        self._set_coefficients_contact_sticking(angle, u_mortar, contact_force)
        self.verify(model)

    def _set_coefficients_contact_sliding(self, angle, u_mortar, contact_force):
        # _set_coefficients_sticking()

        # In addition, we need the coefficients for the normal contact force
        # The tangential force still has a zero coefficient

        # Coefficients set in the model. These should really be set explicitly here,
        # to be in control over any changes
        friction_coefficient = 1
        cnum = 100

        c = np.cos(angle)
        s = np.sin(angle)

        # Projection of xy displacemnet jump onto the normal direction
        u_jump_normal = (
            -s * u_mortar[2] + c * u_mortar[3] - (-s * u_mortar[0] + c * u_mortar[1])
        )

        if not self.pos_normal:
            u_jump_normal *= -1

        u_jump_tangential = (
            c * u_mortar[2] + s * u_mortar[3] - (c * u_mortar[0] + s * u_mortar[1])
        )
        if not self.pos_tangent:
            u_jump_tangential *= -1

        # Definition of the friction bound from Berge et al
        friction_bound = friction_coefficient * (
            -contact_force[1] - cnum * u_jump_normal
        )

        # Test that the initial conditions for the discretization are set so that the
        # fracture is in contact and sticking
        # This is essentially a test that the conditions are consistent with the
        # part of the discretization that is meant to be tested.
        # If these fail, so will comparison with known and computed penetration and
        # sliding states.

        # The condition for open fracture should not be fulfilled
        self.assertTrue(friction_bound >= 0)
        # The condition for contact and sticking should be fulfilled
        self.assertTrue(
            np.abs(-contact_force[0] - cnum * u_jump_tangential) >= friction_bound
        )

        # Coefficient set according to (30) in Berge et al
        self.known_Acc[0, 1] = (
            -friction_coefficient * u_jump_tangential / friction_bound
        )
        self.known_bc[0] = u_jump_tangential

        alpha = (
            -contact_force[0]
            * (-contact_force[0] - cnum * u_jump_tangential)
            / (
                np.abs(contact_force[0])
                * np.abs(-contact_force[0] - cnum * u_jump_tangential)
            )
        )
        # Delta coefficient. Note that the definition given in Berge is wrong - the
        # below is correct, and what is used in the code
        delta = min(abs(contact_force[0]) / friction_bound, 1)

        if alpha < 0:
            beta = 1 / (1 - alpha * delta)
        else:
            beta = 1

        # Regularized coefficient Q
        Q = (
            -contact_force[0]
            * (-contact_force[0] - cnum * u_jump_tangential)
            / (
                np.abs(contact_force[0])
                * np.abs(-contact_force[0] - cnum * u_jump_tangential)
            )
        )
        # Coefficient e
        e = friction_bound / (np.abs(-contact_force[0] - cnum * u_jump_tangential))

        # Regularized coefficient M
        M = e * (1 - Q)

        # Regularized coefficient L
        L = cnum * (1 / (1 - beta * M) - 1)

        # Coefficient v
        v = (
            1
            / (1 - M)
            * (
                (-contact_force[0] - cnum * u_jump_tangential)
                / (np.abs(-contact_force[0] - cnum * u_jump_tangential))
            )
        )

        # Coefficient r
        r = -1 / (1 - M) * e * Q * (-contact_force[0] - cnum * u_jump_tangential)

        # Finally ready to define the coefficients. First tangential direction
        # (both u and lambda contributions)

        # Contribution to A_cm: Projection from global to tangential direction, scaled
        # with L
        self.known_Acm[0, 0] = -c * L
        self.known_Acm[0, 1] = -s * L
        self.known_Acm[0, 2] = c * L
        self.known_Acm[0, 3] = s * L
        if not self.pos_tangent:
            self.known_Acm[0] *= -1

        self.known_Acc[0, 0] = 1
        # NBNB: There is a sign error for this term in Berge (31), the minus sign is
        # correct
        self.known_Acc[0, 1] = -friction_coefficient * v

        self.known_bc[0] = r + friction_bound * v

        # Normal direction: coefficients are the projection of the jump onto the normal
        # direction. No contribution to A_cc
        self.known_Acm[1, 0] = s
        self.known_Acm[1, 1] = -c
        self.known_Acm[1, 2] = -s
        self.known_Acm[1, 3] = c
        if not self.pos_normal:
            self.known_Acm[1] *= -1

        self.known_penetration = np.array([True])
        self.known_sliding = np.array([True])

    def test_zero_angle_penetration_sliding(self):
        # In contact, nonzero tangential jump, no sliding.
        # This is the simplest setup for the sticking state.
        self.setUp()

        # The tangential direction coincides with the global x-direction
        # normal direction with global y-direction
        angle = 0
        c = np.cos(angle)
        s = np.sin(angle)

        # Tangential jump in previous iteration
        tj = 0.1

        # The tangential jump should be projected back to the xy-coordinates
        # Divide by two to distribute jump on the two sides of the mortar grid
        u_mortar = 0.5 * np.array([tj * c, tj * s, -tj * c, -tj * s])
        # Make contact force sufficiently strong to in effect always be in contact
        contact_force = np.array([-1, -0.01])
        model = ContactModel2d(angle, u_mortar, contact_force, True, True)

        self._set_known_contact_to_mortar(angle, model)

        self._set_coefficients_contact_sliding(angle, u_mortar, contact_force)

        self.verify(model)

    def test_rot_penetration_sliding_pos_tangential_pos_normal(self):
        # Rotation, given configuration for normal and tangential directions
        self.setUp()

        # The tangential direction coincides with the global x-direction
        # normal direction with global y-direction
        angle = ((np.pi / 6) * np.random.rand(1))[0]
        c = np.cos(angle)
        s = np.sin(angle)

        # Tangential jump in previous iteration
        tj = 0.1

        # The tangential jump should be projected back to the xy-coordinates
        # Divide by two to distribute jump on the two sides of the mortar grid
        u_mortar = 0.5 * np.array([tj * c, tj * s, -tj * c, -tj * s])
        # Make contact force sufficiently strong to in effect always be in contact
        contact_force = np.array([-1, -0.01])
        model = ContactModel2d(angle, u_mortar, contact_force, True, True)

        self._set_known_contact_to_mortar(angle, model)

        self._set_coefficients_contact_sliding(angle, u_mortar, contact_force)

        self.verify(model)

    def test_rot_penetration_sliding_pos_tangential_neg_normal(self):
        # Rotation, given configuration for normal and tangential directions
        self.setUp()

        # The tangential direction coincides with the global x-direction
        # normal direction with global y-direction
        angle = ((np.pi / 6) * np.random.rand(1))[0]
        c = np.cos(angle)
        s = np.sin(angle)

        # Tangential jump in previous iteration
        tj = 0.1

        # The tangential jump should be projected back to the xy-coordinates
        # Divide by two to distribute jump on the two sides of the mortar grid
        u_mortar = 0.5 * np.array([tj * c, tj * s, -tj * c, -tj * s])
        # Make contact force sufficiently strong to in effect always be in contact
        contact_force = np.array([-1, -0.01])
        model = ContactModel2d(angle, u_mortar, contact_force, True, True)

        self._set_known_contact_to_mortar(angle, model)

        self._set_coefficients_contact_sliding(angle, u_mortar, contact_force)

        self.verify(model)

    def test_rot_penetration_sliding_neg_tangential_pos_normal(self):
        # Rotation, given configuration for normal and tangential directions
        self.setUp()

        # The tangential direction coincides with the global x-direction
        # normal direction with global y-direction
        angle = ((np.pi / 6) * np.random.rand(1))[0]
        c = np.cos(angle)
        s = np.sin(angle)

        # Tangential jump in previous iteration
        tj = 0.1

        # The tangential jump should be projected back to the xy-coordinates
        # Divide by two to distribute jump on the two sides of the mortar grid
        u_mortar = 0.5 * np.array([tj * c, tj * s, -tj * c, -tj * s])
        # Make contact force sufficiently strong to in effect always be in contact
        contact_force = np.array([-1, -0.01])
        model = ContactModel2d(angle, u_mortar, contact_force, True, True)

        self._set_known_contact_to_mortar(angle, model)

        self._set_coefficients_contact_sliding(angle, u_mortar, contact_force)

        self.verify(model)

    def test_rot_penetration_sliding_neg_tangential_neg_normal(self):
        # Rotation, given configuration for normal and tangential directions
        self.setUp()

        # The tangential direction coincides with the global x-direction
        # normal direction with global y-direction
        angle = ((np.pi / 6) * np.random.rand(1))[0]
        c = np.cos(angle)
        s = np.sin(angle)

        # Tangential jump in previous iteration
        tj = 0.1

        # The tangential jump should be projected back to the xy-coordinates
        # Divide by two to distribute jump on the two sides of the mortar grid
        u_mortar = 0.5 * np.array([tj * c, tj * s, -tj * c, -tj * s])
        # Make contact force sufficiently strong to in effect always be in contact
        contact_force = np.array([-1, -0.01])
        model = ContactModel2d(angle, u_mortar, contact_force, True, True)

        self._set_known_contact_to_mortar(angle, model)

        self._set_coefficients_contact_sliding(angle, u_mortar, contact_force)

        self.verify(model)


class ContactModel2d(ContactMechanics):
    def __init__(self, angle, u_mortar, contact_force, pos_tangent, pos_normal):
        super().__init__({})

        # override the super __init__, but that should be okay
        self.angle = angle

        self.create_grid()

        pp.contact_conditions.set_projections(self.gb)

        proj = self.gb.node_props(self.g1)["tangential_normal_projection"].projection
        if pos_tangent and proj[0, 0, 0] < 0:
            proj[0] *= -1
        elif not pos_tangent and proj[0, 0, 0] > 0:
            proj[0] *= -1
        if pos_normal and proj[1, 1, 0] < 0:
            proj[1] *= -1
        elif not pos_normal and proj[1, 1, 0] > 0:
            proj[1] *= -1

        self._set_parameters()

        data = self.gb.node_props(self.g2)
        data[pp.PARAMETERS][self.mechanics_parameter_key]["inverter"] = "python"

        self._assign_variables()
        self._assign_discretizations()
        self.set_state(u_mortar, contact_force)
        self._discretize()

    def get_matrices(self):

        assembler = self.assembler
        dof_manager = self.dof_manager

        A, b = assembler.assemble_matrix_rhs()

        dof_mortar = dof_manager.dof_ind(self.edge, self.mortar_displacement_variable)
        dof_contact = dof_manager.dof_ind(self.g1, self.contact_traction_variable)

        A_mc = A[dof_mortar][:, dof_contact].toarray()
        A_cm = A[dof_contact][:, dof_mortar].toarray()
        A_cc = A[dof_contact][:, dof_contact].toarray()

        b_c = b[dof_contact]

        data_l = self.gb.node_props(self.g1)[pp.STATE][pp.ITERATE]
        penetration = data_l["penetration"]
        sliding = data_l["sliding"]

        return A_mc, A_cm, A_cc, b_c, penetration, sliding

    def create_grid(self):
        """
        Domain is unit square. One fracture, centered on (0.5, 0.5), tilted
        according to self.angle.

        """

        angle = self.angle

        corners = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])

        # The fracture points always have x coordinates 0.2 and 0.8
        # The y-coordinates are set so that the fracture forms the prescribed angle with
        # the x-axis
        frac_pt = np.array(
            [[0.2, 0.8], [0.5 - 0.3 * np.tan(angle), 0.5 + 0.3 * np.tan(angle)], [0, 0]]
        )

        nodes = np.hstack((corners, frac_pt))

        rows = np.array(
            [
                [0, 1],
                [1, 5],
                [5, 0],
                [1, 2],
                [2, 5],
                [2, 4],
                [2, 3],
                [3, 0],
                [3, 4],
                [4, 0],
                [4, 5],
                [4, 5],
            ]
        ).ravel()
        cols = np.vstack((np.arange(12), np.arange(12))).ravel("F")
        data = np.ones_like(rows)

        fn_2d = sps.coo_matrix((data, (rows, cols)), shape=(6, 12)).tocsc()
        rows = np.array(
            [[0, 1, 2], [1, 3, 4], [4, 5, 10], [5, 6, 8], [7, 8, 9], [2, 11, 9]]
        ).ravel()
        cols = np.tile(np.arange(6), (3, 1)).ravel("F")
        data = np.array(
            [-1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1]
        )
        cf_2d = sps.coo_matrix((data, (rows, cols)), shape=(12, 6)).tocsc()

        g_2d = pp.Grid(2, nodes, fn_2d, cf_2d, "mock_2d_grid")
        g_2d.compute_geometry()

        fn_1d = sps.csc_matrix(np.array([[1, 0], [0, 1]]))
        cf_1d = sps.csc_matrix(np.array([[-1], [1]]))

        g_1d = pp.Grid(1, frac_pt, fn_1d, cf_1d, "mock_1d_grid")
        g_1d.compute_geometry()

        gb = pp.GridBucket()
        gb.add_nodes([g_2d, g_1d])

        # Construct mortar grid
        side_grids = {
            mortar_grid.MortarSides.LEFT_SIDE: g_1d.copy(),
            mortar_grid.MortarSides.RIGHT_SIDE: g_1d.copy(),
        }

        data = np.array([1, 1])
        row = np.array([0, 0])
        col = np.array([10, 11])
        face_cells_mortar = sps.coo_matrix(
            (data, (row, col)), shape=(g_1d.num_cells, g_2d.num_faces)
        ).tocsc()

        mg = pp.MortarGrid(1, side_grids, face_cells_mortar)

        edge = (g_2d, g_1d)
        gb.add_edge(edge, face_cells_mortar)
        d = gb.edge_props(edge)

        d["mortar_grid"] = mg

        self.gb = gb
        self._Nd = 2

        self.g1 = g_1d
        self.g2 = g_2d
        self.edge = edge
        self.mg = mg

    def set_state(self, u_mortar, contact_force):
        # Replacement for the method 'initial_condition'. Change name to avoid
        # conflict with different parameters.

        for g, d in self.gb:
            if g.dim == self._Nd:
                # Initialize displacement variable
                state = {self.displacement_variable: np.zeros(g.num_cells * self._Nd)}

            elif g.dim == self._Nd - 1:
                # Initialize contact variable
                traction = contact_force
                state = {
                    # Set value for previous iterate. This becomes what decides the
                    # state used in discretization.
                    pp.ITERATE: {self.contact_traction_variable: traction},
                    # Zero state in previous time step, just to avoid confusion
                    self.contact_traction_variable: 0 * traction,
                }
            else:
                state = {}
            pp.set_state(d, state)

        for _, d in self.gb.edges():
            mg = d["mortar_grid"]

            if mg.dim == self._Nd - 1:
                state = {
                    # Set a zero state in previous time step
                    self.mortar_displacement_variable: 0 * u_mortar,
                    # Set value for previous iterate. This becomes what decides the
                    # state used in discretization.
                    pp.ITERATE: {self.mortar_displacement_variable: u_mortar},
                }
            else:
                state = {}
            pp.set_state(d, state)


class ContactModel3d(ContactModel2d):
    def __init__(
        self,
        n_angle,
        t_angle,
        u_mortar,
        contact_force,
        pos_tangent_1,
        pos_tangent_2,
        pos_normal,
    ):
        super().__init__({})

        # n_angle is specified rotation of the normal vector. Used both in grid
        # construction and construction of the projection operator for the grid
        # t_angle is rotation in the xy-plane of the first tangential vector, *before*
        # rotation around y-axis
        # pos_tangent_1: Whether the first tangential vector has a positive x-component
        # pos_tangent_2: Whether the second tangential vector has a positive
        # y-component
        # pos_normal: Whether the normal vector has a positive z-component
        # The manual construction of the coordinate system may reduce the power of the
        # tests relating to pure geometry, but the alternative requires quite a bit more
        # thinking regarding trigonometry

        # Angle of perturbation of the normal vector of the fracture. gives the
        # deviation from a pure z-direction fracture
        self.angle = n_angle

        self.create_grid()

        # Rotation of the normal vector. We know this is around the y-axis (by
        # construction of the grid, which nodes we perturbed)
        cn = np.cos(n_angle)
        sn = np.sin(n_angle)
        Rxz = np.array([[cn, 0, -sn], [0, 1, 0], [sn, 0, cn]])

        ct, st = np.cos(t_angle), np.sin(t_angle)
        Rxy = np.array([[ct, 0, -st], [0, 1, 0], [st, 0, ct]])

        proj = Rxz.dot(Rxy.dot(np.identity(3)))

        if pos_tangent_1 and proj[0, 0] < 0:
            proj[0] *= -1
        elif not pos_tangent_1 and proj[0, 0] > 0:
            proj[0] *= -1
        if pos_tangent_2 and proj[1, 1] < 0:
            proj[1] *= -1
        elif not pos_tangent_2 and proj[1, 1] > 0:
            proj[1] *= -1
        if pos_normal and proj[2, 2] < 0:
            proj[2] *= -1
        elif not pos_normal and proj[2, 2] > 0:
            proj[2] *= -1

        self.gb.node_props(self.g1)["tangential_normal_projection"].projection[
            :, :, 0
        ] = proj

        self.set_parameters()

        data = self.gb.node_props(self.g2)
        data[pp.PARAMETERS][self.mechanics_parameter_key]["inverter"] = "python"

        self.assign_variables()
        self.assign_discretizations()
        self.set_state(u_mortar, contact_force)
        self.discretize()

        pass

    def create_grid(self):

        s = np.sin(self.angle)
        df = 1

        f = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]])

        gb = pp.meshing.cart_grid([f], np.array([1, 1, 2]))

        g3 = gb.grids_of_dimension(3)[0]
        g2 = gb.grids_of_dimension(2)[0]

        xn3 = g3.nodes
        hit3 = np.logical_and.reduce((xn3[0] > 0.5, xn3[2] > 0.5, xn3[2] < 1.5))
        assert hit3.sum() == 4

        xn2 = g2.nodes
        hit2 = np.logical_and.reduce((xn2[0] > 0.5, xn2[2] > 0.5, xn2[2] < 1.5))
        assert hit2.sum() == 2

        xn3[2, hit3] += df * s
        xn2[2, hit2] += df * s

        self.gb = gb
        self.g3 = g3
        self.g2 = g2

        self.Nd = 3

        for e, d in gb.edges():
            self.edge = e
            self.mg = e["mortar_grid"]


if __name__ == "__main__":
    unittest.main()
