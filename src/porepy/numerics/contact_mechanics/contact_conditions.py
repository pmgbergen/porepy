#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 08:53:05 2019

@author: eke001
"""
import numpy as np
import scipy.sparse as sps

import porepy as pp


class ColoumbContact:
    def __init__(self, keyword, ambient_dimension):
        self.keyword = keyword

        self.dim = ambient_dimension

        self.surface_variable = "mortar_u"
        self.contact_variable = "contact_force"

        self.friction_parameter_key = "friction"
        self.surface_parameter_key = "surface"

        self.traction_discretization = "traction_discretization"
        self.displacement_discretization = "displacement_discretization"
        self.rhs_discretization = "contact_rhs"

    def _key(self):
        return self.keyword + "_"

    def _discretization_key(self):
        return self._key() + pp.keywords.DISCRETIZATION

    def ndof(self, g):
        return g.num_cells * self.dim

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):

        parameters_l = data_l[pp.PARAMETERS]

        c_num = 100

        mg = data_edge["mortar_grid"]

        nc = g_h.face_normals[: g_h.dim] / g_h.face_areas

        # Map normal vector to the mortar grid
        nc_mortar = mg.master_to_mortar_int().dot(nc.T).T

        # Use a single normal vector to span the tangential and normal space.
        projection = pp.TangentialNormalProjection(nc_mortar[:, 0].reshape((-1, 1)))

        displacement_jump = (
            mg.mortar_to_slave_avg(nd=self.dim)
            * mg.sign_of_mortar_sides(nd=self.dim)
            * data_edge[self.surface_variable]
        )

        contact_force = data_l[self.contact_variable]

        friction_coefficient = parameters_l[self.friction_parameter_key][
            "friction_coefficient"
        ]

        friction_bound = friction_coefficient * np.clip(
            projection.project_normal(g_l.num_cells)
            * (-contact_force + c_num * displacement_jump),
            0,
            np.inf,
        )

        num_cells = friction_coefficient.size
        nd = projection.dim

        # Process input
        if np.asarray(friction_coefficient).size == 1:
            friction_coefficient = friction_coefficient * np.ones(num_cells)

        # Structures for storing the computed coefficients.
        robin_weight = []  # Multiplies displacement jump
        mortar_weight = []  # Multiplies the normal forces
        rhs = np.array([])  # Goes to the right hand side.

        # Change coordinate system to the one alligned to the fractures
        # The rotation matrix is structured so that in the rotated coordinates, the
        # tangential direction is defined in the first mg.dim rows, while the final
        # row is associated with the normal direction.
        tangential_projection = projection.project_tangential(num_cells)
        normal_projection = projection.project_normal(num_cells)

        normal_contact_force = normal_projection * contact_force
        tangential_contact_force = (tangential_projection * contact_force).reshape(
            (nd - 1, num_cells), order="F"
        )

        normal_displacement_jump = normal_projection * displacement_jump
        tangential_displacement_jump = (
            tangential_projection * displacement_jump
        ).reshape((nd - 1, num_cells), order="F")

        # Find contact and sliding region

        # Contact region is determined from the normal direction, stored in the
        # last row of the projected stress and deformation.
        penetration_bc = self._penetration(
            normal_contact_force, normal_displacement_jump, c_num
        )
        sliding_bc = self._sliding(
            tangential_contact_force,
            tangential_displacement_jump,
            friction_bound,
            c_num,
        )

        # Zero vectors of the size of the tangential space and the full space,
        # respectively
        zer = np.array([0] * (nd - 1))
        zer1 = np.array([0] * (nd))
        zer1[-1] = 1

        # Loop over all mortar cells, discretize according to the current state of
        # the contact
        # The loop computes three parameters:
        # L will eventually multiply the displacement jump, and be associated with
        #   the coefficient in a Robin boundary condition (using the terminology of
        #   the mpsa implementation)
        # r is the right hand side term

        import pdb

     #  pdb.set_trace()

        for i in range(num_cells):
            if sliding_bc[i] & penetration_bc[i]:  # in contact and sliding
                # The equation for the normal direction is computed from equation
                # (24)-(25) in Berge et al.
                # Compute coeffecients L, r, v
                L, r, v = self._L_r(
                    tangential_contact_force[:, i],
                    tangential_displacement_jump[:, i],
                    friction_bound[i],
                    c_num,
                )

                # There is no interaction between displacement jumps in normal and
                # tangential direction
                L = np.hstack((L, np.atleast_2d(zer).T))
                L = np.vstack((L, zer1))
                # Right hand side is computed from (24-25). In the normal
                # direction, zero displacement is enforced.
                # This assumes that the original distance, g, between the fracture
                # walls is zero.
                r = np.vstack((r + friction_bound[i] * v, 0))
                # Unit contribution from tangential force
                MW = np.eye(nd)
                # Contribution from normal force
                MW[:-1, -1] = -friction_coefficient[i] * v.ravel()

            elif ~sliding_bc[i] & penetration_bc[i]:  # In contact and sticking
                # Mortar weight computed according to (23)
                mw = (
                    -friction_coefficient[i]
                    * tangential_displacement_jump[:, i].ravel("F")
                    / friction_bound[i]
                )
                # Unit coefficient for all displacement jumps
                L = np.eye(nd)
                MW = np.zeros((nd, nd))
                MW[:-1, -1] = mw
                r = np.hstack((tangential_displacement_jump[:, i], 0)).T

            elif ~penetration_bc[i]:  # not in contact
                # This is a free boundary, no conditions on u
                L = np.zeros((nd, nd))
                # Free boundary conditions on the forces.
                MW = np.eye(nd)
                r = np.zeros(nd)
            else:  # should never happen
                raise AssertionError("Should not get here")

            #  Append a mapping from global to the local coordinate system.
            # The coefficients are already computed in the local coordinates.
            L = L.dot(projection.local_projection(0))
            MW = MW.dot(projection.local_projection(0))
            # Scale equations (helps iterative solver)
            w_diag = np.diag(L) + np.diag(MW)
            W_inv = np.diag(1 / w_diag)
            L = W_inv.dot(L)
            MW = W_inv.dot(MW)
            r = r.ravel() / w_diag
            # Append to the list of global coefficients.
            robin_weight.append(L)
            mortar_weight.append(MW)
            rhs = np.hstack((rhs, r))

        traction_coefficients = sps.block_diag(mortar_weight)
        displacement_coefficients = sps.block_diag(robin_weight)

        data_l[pp.DISCRETIZATION_MATRICES][self.keyword][
            self.traction_discretization
        ] = traction_coefficients
        data_l[pp.DISCRETIZATION_MATRICES][self.keyword][
            self.displacement_discretization
        ] = displacement_coefficients * mg.mortar_to_slave_avg(nd=nd)
        data_l[pp.DISCRETIZATION_MATRICES][self.keyword][self.rhs_discretization] = rhs

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
        return self._l2(-Tt + ct * ut) - bf > 1e-10

    def _penetration(self, Tn, un, cn):
        """ Find faces that are in contact.

        Arguments:
            Tn (np.array, num_faces): Normal forces.
            un (np.array, num_faces): Displament in normal direction.
            ct (double): Numerical parameter that relates displacement jump to
                normal forces. See Huber et al for explanation.

        Returns:
            boolean, size num_faces: True if |-Tt + ct*ut| > bf for a face

        """
        # Not sure about the sensitivity to the tolerance parameter here.
        tol = 1e-8 * cn
        return (-Tn + cn * un) > tol

    # Below here are different help function for calculating the Newton step
    def _ef(self, Tt, cut, bf):
        # Compute part of (25) in Berge et al.
        return bf / self._l2(-Tt + cut)

    def _Ff(self, Tt, cut, bf):
        # Implementation of the term Q involved in the calculation of (25) in Berge
        # et al.
        numerator = -Tt.dot((-Tt + cut).T)

        # Regularization to avoid issues during the iterations to avoid dividing by
        # zero if the faces are not in contact durign iterations.
        denominator = max(bf, self._l2(-Tt)) * self._l2(-Tt + cut)

        return numerator / denominator

    def _M(self, Tt, cut, bf):
        """ Compute the coefficient M used in Eq. (25) in Berge et al.
        """
        Id = np.eye(Tt.shape[0])
        return self._ef(Tt, cut, bf) * (Id - self._Ff(Tt, cut, bf))

    def _hf(self, Tt, cut, bf):
        return self._ef(Tt, cut, bf) * self._Ff(Tt, cut, bf).dot(-Tt + cut)

    def _L_r(self, Tt, ut, bf, c):
        """
        Compute the coefficient L, defined in Eq. (25) in Berge et al.

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
        if bf <= 1e-10:
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
        delta = min(self._l2(-Tt) / bf, 1)

        if alpha < 0:
            beta = 1 / (1 - alpha * delta)
        else:
            beta = 1

        # The expression (I - beta * M)^-1
        IdM_inv = np.linalg.inv(Id - beta * coeff_M)

        v = IdM_inv.dot(-Tt + cut) / self._l2(-Tt + cut)

        return c * (IdM_inv - Id), -IdM_inv.dot(self._hf(Tt, cut, bf)), v

    def _l2(self, x):
        x = np.atleast_2d(x)
        return np.sqrt(np.sum(x ** 2, axis=0))
