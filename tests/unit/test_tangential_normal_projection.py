"""Unit tests for class pp.TangentialNormalProjection
"""
import pytest
import unittest

import numpy as np

import porepy as pp


@pytest.mark.parametrize(
    "normal, tangent",
    [
        # 2d vector, normal along the x-axis
        (np.array([[1, 0]]).T, np.array([[0, 1]]).T),
        # 2d vector, normal along the x-axis, tangent in negative y-direction
        (np.array([[1, 0]]).T, np.array([[0, -1]]).T),
        # 2d vector, normal along the negative y-axis
        (np.array([[0, -1]]).T, np.array([[1, 0]]).T),
        # 2d vector, normal in the first quadrant,
        (np.array([[1, 1]]).T, np.array([[-1, 1]]).T),
        # 2d vector, two normals
        (np.array([[1, 0], [1, 1]]).T, np.array([[0, 1], [1, -1]]).T),
        # 3d vector, normal along the x-axis
        (np.array([[1, 0, 0]]).T, np.array([[0, 0, 2]]).T),
        # 3d vector, normal along the negative z-axis
        (np.array([[0, 0, -1]]).T, np.array([[1, 0, 0]]).T),
        # 3d vector, normal along the negative z-axis, tangent in the x-y plane
        (np.array([[0, 0, -1]]).T, np.array([[1, 1, 0]]).T),
        # 3d vector, normal in the first octant
        (np.array([[1, 1, 1]]).T, np.array([[0, 1, -1]]).T),
        # 3d vector, two normals
        (np.array([[1, 0, 0], [1, 1, 1]]).T, np.array([[0, 0, 2], [0, 1, -1]]).T),
    ],
)
def test_tangential_normal_projection(normal: np.ndarray, tangent: np.ndarray):
    """Test that the tangential-normal projection works as expected.

    The test covers:
    * Test of the local projection matrix (attribute
        TangentialNormalProjection.projection)
    * Test of the global projection matrices in the tangential and normal directions,
        (constructed by methods projection_normal(), project_tangential()), and the
        matrix which decomposes a vector into tangential and normal parts (method
        project_tangential_normal())

    The tests project the given normal and tangential vectors into the tangent and
    normal spaces, and verify the following properties:
        * When projected onto the tagent space, the normal vector is zero while the
            tangential vector has the same length as the original vector (note: we
            cannot test the direction of the tangential vector, since the tangential
            basis is arbitrary).
        * When projected onto the normal space, the tangential vector is zero while the
            normal vector has the same length as the original vector.
        * When decomposed into tangential and normal parts, the tangential and normal
            vectors have their expected length in their 'own' spaces and are zero in the
            opposite spaces.
        * When projected onto the tangent space, the combined vector (normal + tangent)
            has the length equal to the tangent vector.
        * When projected onto the normal space, the combined vector (normal + tangent)
            has the length equal to the normal vector.

    The comparisons heavily exploit the fact that, when decomposed into tangential and
    normal parts, the tangential part will be found in the first nd-1 components of the
    vector, while the normal part will be found in the last component. This is true by
    construction of the projection matrix.

    Parameters:
        normal (np.ndarray): Normal vectors, stored as nd x num_vec arrays tangent
        (np.ndarray): Tangential vectors, stored as nd x num_vec arrays. Each
            tangential vector should be orthogonal to the corresponding (i.e., in the
            corresponding column) normal vector.

    """
    # Projections should work for vectors of arbritrary length. If the normal and
    # tangent vectors have the same length, we risk that errors in the directions
    # (normal and tangential is mixed) are masked. We therefore first normalize the
    # vectors, and then scale them to different lengths.
    normal = 2 * normal / np.linalg.norm(normal)
    tangent = 3 * tangent / np.linalg.norm(tangent)

    # Ravel the vectors, to test the global projection matrices
    raveled_normal = normal.ravel("f")
    raveled_tangent = tangent.ravel("f")

    dim, num_vec = normal.shape

    # Create a projection object
    proj = pp.TangentialNormalProjection(normals=normal)

    # Test that the normal vectors are projected onto themselves, using the attribute
    # projection. This tests the local projection matrix
    for i in range(num_vec):
        assert np.allclose(
            (proj.projection[:, :, i] @ normal[:, i])[dim - 1],
            np.linalg.norm(normal[:, i]),
        )
    # Test that the normal vectors are projected onto themselves, using the global
    # projection matrix.
    assert np.allclose(
        # No need to take the norm of the projected matrix, since there is only one
        # component per normal vector (the returned value will be a num_vec array which
        # fits with the result of np.linalg.norm with the axis=0 option).
        proj.project_normal() @ raveled_normal,
        np.linalg.norm(normal, axis=0),
    )
    # Test that, when projected onto the tangent space, the normal vector is zero.
    for i in range(num_vec):
        assert np.allclose(proj.project_tangential() @ raveled_normal, 0)

    # Decompose the normal vector into tangential and normal parts, and test that the
    # tangential part is zero, while the normal part has the same length as the original
    # vector
    decomposed_normal = (proj.project_tangential_normal() @ raveled_normal).reshape(
        (-1, num_vec), order="f"
    )
    assert np.allclose(np.linalg.norm(decomposed_normal[: dim - 1]), 0)
    assert np.allclose(decomposed_normal[dim - 1], np.linalg.norm(normal, axis=0))

    # Test that the projection of the tangential vectors have the same length as the
    # original vector, using the local projection matrix.
    for i in range(num_vec):
        assert np.allclose(
            np.linalg.norm((proj.projection[:, :, i] @ tangent[:, i])[: dim - 1]),
            np.linalg.norm(tangent[:, i]),
        )

    # Test that the projection of the tangential vector onto the tangent space has the
    # same length as the original vector, using the global projection matrix. Here we
    # need to reshape the projected matrix back to a (nd-1) x num_vec array and take the
    # column-wise norm (in contrast to the normal direction, where there is only one
    # component per normal vector).
    assert np.allclose(
        np.linalg.norm(
            (proj.project_tangential() @ raveled_tangent).reshape(
                (-1, num_vec), order="f"
            ),
            axis=0,
        ),
        np.linalg.norm(tangent, axis=0),
    )

    # Test that, when projected onto the normal space, the tangential vector is zero.
    for i in range(num_vec):
        assert np.allclose((proj.project_normal() @ raveled_tangent), 0)

    # Test that, when decomposed by a tangential-normal projection, the tangential
    # vector is non-zero only in the tangential direction
    decomposed_tangent = (proj.project_tangential_normal() @ raveled_tangent).reshape(
        (-1, num_vec), order="f"
    )
    assert np.allclose(
        np.linalg.norm(decomposed_tangent[: dim - 1]), np.linalg.norm(tangent)
    )
    assert np.allclose(decomposed_tangent[dim - 1], 0)

    # Create a vector with both normal and tangential components
    raveled_combined = (normal + tangent).ravel("f")

    # Test that the projection of the combined vector onto the tangent space is the same
    # as the projection of the tangential vector
    assert np.allclose(
        proj.project_tangential() @ raveled_combined,
        proj.project_tangential() @ raveled_tangent,
    )
    # Test that the projection of the combined vector onto the normal space is the same
    # as the projection of the normal vector
    assert np.allclose(
        proj.project_normal() @ raveled_combined, proj.project_normal() @ raveled_normal
    )


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
            # Check that the projection of the normal vector only has a component in the
            # normal direction
            projected_normal = proj.projection[:, :, i].dot(self.n2_normalized[:, i])
            self.assertTrue(np.allclose(projected_normal, known_projection_of_normal))

    def test_computed_basis_3d(self):
        # Test that the computed basis functions are orthonormal, and that the
        # correct normal vector is constructed
        proj = pp.TangentialNormalProjection(self.n3)
        self._verify_orthonormal(proj)

        known_projection_of_normal = np.array([0, 0, 1])
        for i in range(self.n3.shape[1]):
            # Check that the projection of the normal vector only has a component in the
            # normal direction
            projected_normal = proj.projection[:, :, i].dot(self.n3_normalized[:, i])
            self.assertTrue(np.allclose(projected_normal, known_projection_of_normal))

    def test_project_tangential_normal_2d(self):
        # Test projections into normal and tangential directions in 2d.
        # Only a single vector is projected, the multi-vector option is handled in
        # another test.
        s = 1 / np.sqrt(2)
        n = np.array([[s], [s]])

        proj = pp.TangentialNormalProjection(n)

        # Vectors to be projected
        vecs = [n, np.array([[1], [0]]), np.array([[0], [1]]), np.array([[-s], [s]])]
        # Length of the vectors when projected onto the tangent space
        known_tangential_projections = [0, s, s, 1]
        # length of the vectors when projected onto the normal space
        known_normal_projections = [1, s, s, 0]

        known_projected_vecs = [
            np.array([[0], [1]]),
            np.array([[s], [s]]),
            np.array([[s], [s]]),
            np.array([[1], [0]]),
        ]

        for i, v in enumerate(vecs):
            # Project tangential direction
            # Add an absolute value since we don't know the positive direction chosen
            # for the tangential space (this is chosen by random in the construction
            # of the projection object).
            tangential_length = np.abs(proj.project_tangential() * v)
            self.assertTrue(
                np.allclose(known_tangential_projections[i], tangential_length)
            )

            # project in normal direction
            # No need for absolute values here
            normal_length = proj.project_normal() * v
            self.assertTrue(np.allclose(known_normal_projections[i], normal_length))

            pv = np.abs(proj.project_tangential_normal() * v)
            self.assertTrue(np.allclose(known_projected_vecs[i], pv))

    def test_project_tangential_normal_30deg_2d(self):
        # Test projections into normal and tangential directions in 2d.
        # Only a single vector is projected, the multi-vector option is handled in
        # another test.
        c = np.cos(np.pi / 6)
        s = np.sin(np.pi / 6)
        n = np.array([[c], [s]])

        proj = pp.TangentialNormalProjection(n)

        # Vectors to be projected
        vecs = [n, np.array([[1], [0]]), np.array([[0], [1]]), np.array([[-s], [c]])]
        # Length of the vectors when projected onto the tangent space
        known_tangential_projections = [0, s, c, 1]
        # length of the vectors when projected onto the normal space
        known_normal_projections = [1, c, s, 0]

        known_projected_vecs = [
            np.array([[0], [1]]),
            np.array([[s], [c]]),
            np.array([[c], [s]]),
            np.array([[1], [0]]),
        ]

        for i, v in enumerate(vecs):
            # Project tangential direction
            # Add an absolute value since we don't know the positive direction chosen
            # for the tangential space (this is chosen by random in the construction
            # of the projection object).
            tangential_length = np.abs(proj.project_tangential() * v)
            self.assertTrue(
                np.allclose(known_tangential_projections[i], tangential_length)
            )

            # project in normal direction
            # No need for absolute values here
            normal_length = proj.project_normal() * v
            self.assertTrue(np.allclose(known_normal_projections[i], normal_length))

            pv = np.abs(proj.project_tangential_normal() * v)
            self.assertTrue(np.allclose(known_projected_vecs[i], pv))

    def test_project_tangential_normal_2d_several_vectors(self):
        # Test projection of several vectors, using the same projection direction
        n = np.array([[1], [0]])

        proj = pp.TangentialNormalProjection(n)

        # Vector to be projected. When interpreted as three stacked vectors, the first
        # has only normal component, the second only tangential component, and the third
        # equal components of both
        vec = np.array([1, 0, 0, 1, 1, 1]).reshape((-1, 1))

        known_tangential_parts = np.array([0, 1, 1]).reshape((-1, 1))
        known_normal_parts = np.array([1, 0, 1]).reshape((-1, 1))

        # Compute projection in tangential direction.
        # absolute value is needed to account for randomness in positive tangential
        # direction
        proj_tangential = np.abs(proj.project_tangential(3) * vec)
        self.assertTrue(np.allclose(known_tangential_parts, proj_tangential))

        # No absolute value for normal direction
        proj_normal = proj.project_normal(3) * vec
        self.assertTrue(np.allclose(proj_normal, known_normal_parts))

        known_tangential_normal = np.array([[0, 1, 1, 0, 1, 1]]).reshape((-1, 1))

        proj_full = proj.project_tangential_normal(3) * vec
        self.assertTrue(np.allclose(proj_full, known_tangential_normal))

    def test_project_tangential_normal_3d(self):
        # Test projections into normal and tangential directions in 2d.
        # Only a single vector is projected, the multi-vector option is handled in
        # another test.
        s = 1 / np.sqrt(3)
        n = np.array([[s], [s], [s]])

        proj = pp.TangentialNormalProjection(n)

        # Vectors to be projected
        vecs = [
            n,
            np.array([[1], [0], [0]]),
            np.array([[0], [1], [0]]),
            np.array([[-1 / np.sqrt(2)], [1 / np.sqrt(2)], [0]]),
        ]

        # Length of the vectors when projected onto the tangent space
        known_tangential_projections = [0, np.sqrt(2 / 3), np.sqrt(2 / 3), 1]
        # length of the vectors when projected onto the normal space
        known_normal_projections = [1, s, s, 0]

        for i, v in enumerate(vecs):
            # Project tangential direciton
            tangential_length = np.linalg.norm(proj.project_tangential() * v)
            self.assertTrue(
                np.allclose(known_tangential_projections[i], tangential_length)
            )

            # project in normal direction
            normal_length = np.abs(proj.project_normal() * v)
            self.assertTrue(np.allclose(known_normal_projections[i], normal_length))

    def test_project_tangential_normal_3d_several_vectors(self):
        # Test projection of several vectors, using the same projection direction
        n = np.array([[1], [0], [0]])

        proj = pp.TangentialNormalProjection(n)

        # Vector to be projected. When interpreted as three stacked vectors, the first
        # has only normal component, the second only tangential component, and the third
        # equal components of both
        vec = np.array([1, 0, 0, 0, 1, 0, 1, 1, 0]).reshape((-1, 1))

        # Compute projection in tangential direction.
        proj_tangential = np.abs(proj.project_tangential(3) * vec)
        # The first two components should both be zero
        self.assertTrue(np.allclose(proj_tangential[:2], 0))
        # The components corresponding to the second vector should together have unit
        # length
        self.assertTrue(np.allclose(np.linalg.norm(proj_tangential[2:4]), 1))
        # The components corresponding to the third vector should together have unit
        # length
        self.assertTrue(np.allclose(np.linalg.norm(proj_tangential[4:]), 1))

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

    def test_several_normal_vectors(self):
        dim = 2

        s2 = np.sqrt(2)
        s3 = np.sqrt(3)
        n1 = np.array([[0.5], [s3 / 2]])
        n2 = np.array([[s2 / 2], [s2 / 2]])

        proj = pp.TangentialNormalProjection(np.hstack((n1, n2)))

        # Two 2d vectors stacked. Both have x-component 1, y 0
        v = np.array([1, 0, 1, 0]).reshape((-1, 1))

        t_proj = proj.project_tangential() * v
        n_proj = proj.project_normal() * v

        self.assertTrue(np.allclose(np.abs(t_proj), np.array([[s3 / 2], [s2 / 2]])))

        self.assertTrue(np.allclose(n_proj, np.array([[0.5], [s2 / 2]])))


if __name__ == "__main__":
    unittest.main()
