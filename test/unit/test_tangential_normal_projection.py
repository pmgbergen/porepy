#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for class pp.TangentialNormalProjection
"""

import numpy as np
import unittest

import porepy as pp


class TestTangentialNormalProjection(unittest.TestCase):
    def setUp(self):
        # 2d vectors
        self.n2 = np.array([[0, 1, -2], [1, 1, 0]])

        self.n3 = np.vstack((self.n2, np.array([0, 1, 0])))

        s2 = np.sqrt(2)
        self.n2_normalized = np.array([[0, 1.0 / s2, -1.0], [1, 1.0 / s2, 0]])
        s3 = np.sqrt(3)
        self.n3_normalized = np.array(
            [[0, 1.0 / s3, -1.0], [1, 1.0 / s3, 0], [0, 1.0 / s3, 0]]
        )

    def test_optional_arguments(self):
        # Test automatic check of ambient dimension
        proj = pp.TangentialNormalProjection(self.n2)
        self.assertTrue(proj.dim == 2)

        proj = pp.TangentialNormalProjection(self.n3)
        self.assertTrue(proj.dim == 3)

    def test_normal_vectors(self):
        # Test that the normal vectors have
        proj = pp.TangentialNormalProjection(self.n2)
        for i in range(self.n2.shape[1]):
            self.assertTrue(
                np.allclose(np.sum(proj.normals[:, i] * self.n2_normalized[:, i]), 1)
            )

        proj = pp.TangentialNormalProjection(self.n3)

        for i in range(self.n3.shape[1]):
            self.assertTrue(
                np.allclose(np.sum(proj.normals[:, i] * self.n3_normalized[:, i]), 1)
            )

    def _verify_orthonormal(self, proj):
        # Check that the basis is an orthonormal set
        for i in range(proj.num_vecs):
            for j in range(proj.dim):
                for k in range(proj.dim):
                    if j == k:
                        truth = 1
                    else:
                        truth = 0
                    self.assertTrue(
                        np.allclose(
                            proj.projection[:, j, i].dot(proj.projection[:, k, i]),
                            truth,
                        )
                    )

    def test_computed_basis_2d(self):
        # Test that the computed basis functions are orthonormal, and that the
        # correct normal vector is constructed
        proj = pp.TangentialNormalProjection(self.n2)
        self._verify_orthonormal(proj)

        known_projection_of_normal = np.array([0, 1])
        for i in range(self.n2.shape[1]):

            # Check that the projection of the normal vector only has a component in the normal direction
            projected_normal = proj.projection[:, :, i].dot(self.n2_normalized[:, i])
            self.assertTrue(np.allclose(projected_normal, known_projection_of_normal))

    def test_computed_basis_3d(self):
        # Test that the computed basis functions are orthonormal, and that the
        # correct normal vector is constructed
        proj = pp.TangentialNormalProjection(self.n3)
        self._verify_orthonormal(proj)

        known_projection_of_normal = np.array([0, 0, 1])
        for i in range(self.n3.shape[1]):

            # Check that the projection of the normal vector only has a component in the normal direction
            projected_normal = proj.projection[:, :, i].dot(self.n3_normalized[:, i])
            self.assertTrue(np.allclose(projected_normal, known_projection_of_normal))

    def test_tangential_normal_projection_2d(self):
        # Tests of the projection operators in 2d. Several normal vectors are specified.
        proj = pp.TangentialNormalProjection(self.n2)

        vector = np.arange(1, 7)

        normal_projection = proj.project_normal() * vector
        known_normal_projection = np.array([2, (3 + 4) / np.sqrt(2), -5])
        self.assertTrue(np.allclose(normal_projection, known_normal_projection))

        tangential_projection = proj.project_tangential() * vector
        known_tangential_projection = np.array([1, (3 - 4) / np.sqrt(2), 6])
        # The basis function for the tangential plane is determined up to a sign
        # hence we check both options, independently for each normal vector
        self.assertTrue(
            np.allclose(
                np.abs(tangential_projection), np.abs(known_tangential_projection)
            )
        )

    def test_tangential_normal_projection_3d(self):
        # Tests of the projection operators in 3d. Several normal vectors are specified.
        proj = pp.TangentialNormalProjection(self.n3)

        vector = np.arange(1, 10)

        normal_projection = proj.project_normal() * vector
        known_normal_projection = np.array([2, (4 + 5 + 6) / np.sqrt(3), -7])
        self.assertTrue(np.allclose(normal_projection, known_normal_projection))

        tangential_projection = proj.project_tangential() * vector

        # The directions of the basis functions in the tangential plane are
        # unknown. The only thing we can check is that the projected vectors
        # have the right length
        known_tangential_length = np.array([np.sqrt(10), np.sqrt(2), np.sqrt(64 + 81)])
        computed_tangential_length = np.linalg.norm(
            tangential_projection.reshape((2, 3), order="F"), axis=0
        )

        # The basis function for the tangential plane is determined up to a sign
        # hence we check both options, independently for each normal vector
        self.assertTrue(
            np.allclose(computed_tangential_length, known_tangential_length)
        )

    def test_projections_num_keyword(self):
        # Tests of the generated projection operators, using a single tangential/
        # normal space, but generating several (equal) projection matrices.

        dim = 3

        # Random normal and tangential space
        proj = pp.TangentialNormalProjection(np.random.rand(dim, 1))

        num_reps = 4

        # Random vector to be generated
        vector = np.random.rand(dim, 1)

        projection = proj.project_tangential_normal(num=num_reps)

        proj_vector = projection * np.tile(vector, (num_reps, 1))

        for i in range(dim):
            for j in range(num_reps):
                self.assertTrue(proj_vector[i + j * dim], proj_vector[i])


if __name__ == "__main__":
    unittest.main()
