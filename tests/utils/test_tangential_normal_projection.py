"""Unit tests for class pp.TangentialNormalProjection
"""
import numpy as np
import pytest

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
        # 2d vector, normal in the second quadrant
        (np.array([[-0.1, 1]]).T, np.array([[-1, -0.1]]).T),
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
        # 3d vector, negative components of the normal vector
        (np.array([[1, -2, -3]]).T, np.array([[0, 3, -2]]).T),
        # 3d vector, two normals
        (np.array([[1, 0, 0], [1, 1, 1]]).T, np.array([[0, 0, 2], [0, 1, -1]]).T),
        # 3d vector, almost aligned with the z-axis
        (np.array([[1e-15, 5e-15, np.sqrt(1 - 6e-15)]]).T, np.array([[5, -1, 0]]).T),
        # 3d vector, two almost equal large components, one small component
        (np.array([[1, 1 - (1e-15), 1e-15]]).T, np.array([[1e-15, 0, 1]]).T),
    ],
)
def test_tangential_normal_projection(normal: np.ndarray, tangent: np.ndarray):
    """Test that the tangential-normal projection works as expected.

    The test covers:
    * Test of the local projection matrix (attribute
        TangentialNormalProjection._projection)
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
        normal: Normal vectors, stored as nd x num_vec arrays tangent
        tangent: Tangential vectors, stored as nd x num_vec arrays. Each
            tangential vector should be orthogonal to the corresponding (i.e., in the
            corresponding column) normal vector.

    """

    # Projections should work for vectors of arbritrary length. If the normal and
    # tangent vectors have the same length, we risk that errors in the directions
    # (normal and tangential is mixed) are masked. We therefore first normalize the
    # vectors, and then scale them to different lengths.
    normal = np.sqrt(2) * normal / np.linalg.norm(normal)
    tangent = np.sqrt(3) * tangent / np.linalg.norm(tangent)

    # Ravel the vectors, to test the global projection matrices
    raveled_normal = normal.ravel("F")
    raveled_tangent = tangent.ravel("F")

    dim, num_vec = normal.shape

    # Create a projection object
    proj = pp.TangentialNormalProjection(normals=normal)

    # Test that the projection matrix is orthonormal
    for i in range(num_vec):
        loc_proj = proj._projection[:, :, i]
        for j in range(dim):
            for k in range(dim):
                if j == k:
                    truth = 1
                else:
                    truth = 0
                assert np.allclose(loc_proj[:, k].dot(loc_proj[:, j]), truth)

    # Test that the normal vectors are projected onto themselves, using the attribute
    # projection. This tests the local projection matrix
    for i in range(num_vec):
        assert np.allclose(
            (proj._projection[:, :, i] @ normal[:, i])[dim - 1],
            np.linalg.norm(normal[:, i]),
            rtol=1e-14,
        )
    # Test that the normal vectors are projected onto themselves, using the global
    # projection matrix.
    assert np.allclose(
        # No need to take the norm of the projected matrix, since there is only one
        # component per normal vector (the returned value will be a num_vec array which
        # fits with the result of np.linalg.norm with the axis=0 option).
        proj.project_normal() @ raveled_normal,
        np.linalg.norm(normal, axis=0),
        rtol=1e-14,
    )
    # Test that, when projected onto the tangent space, the normal vector is zero.
    for i in range(num_vec):
        assert np.allclose(proj.project_tangential() @ raveled_normal, 0, rtol=1e-14)

    # Decompose the normal vector into tangential and normal parts, and test that the
    # tangential part is zero, while the normal part has the same length as the original
    # vector
    decomposed_normal = (proj.project_tangential_normal() @ raveled_normal).reshape(
        (-1, num_vec), order="f"
    )
    assert np.allclose(np.linalg.norm(decomposed_normal[: dim - 1]), 0, rtol=1e-14)
    assert np.allclose(
        decomposed_normal[dim - 1], np.linalg.norm(normal, axis=0), rtol=1e-14
    )

    # Test that the projection of the tangential vectors have the same length as the
    # original vector, using the local projection matrix.
    for i in range(num_vec):
        assert np.allclose(
            np.linalg.norm((proj._projection[:, :, i] @ tangent[:, i])[: dim - 1]),
            np.linalg.norm(tangent[:, i]),
            rtol=1e-14,
        )

    # Test that the projection of the tangential vector onto the tangent space has the
    # same length as the original vector, using the global projection matrix. Here we
    # need to reshape the projected matrix back to a (nd-1) x num_vec array and take the
    # column-wise norm (in contrast to the normal direction, where there is only one
    # component per normal vector).
    assert np.allclose(
        np.linalg.norm(
            (proj.project_tangential() @ raveled_tangent).reshape(
                (-1, num_vec), order="F"
            ),
            axis=0,
        ),
        np.linalg.norm(tangent, axis=0),
        rtol=1e-14,
    )

    # Test that, when projected onto the normal space, the tangential vector is zero.
    for i in range(num_vec):
        assert np.allclose((proj.project_normal() @ raveled_tangent), 0, rtol=1e-14)

    # Test that, when decomposed by a tangential-normal projection, the tangential
    # vector is non-zero only in the tangential direction
    decomposed_tangent = (proj.project_tangential_normal() @ raveled_tangent).reshape(
        (-1, num_vec), order="F"
    )
    assert np.allclose(
        np.linalg.norm(decomposed_tangent[: dim - 1]),
        np.linalg.norm(tangent),
        rtol=1e-14,
    )
    assert np.allclose(decomposed_tangent[dim - 1], 0, rtol=1e-14)

    # Create a vector with both normal and tangential components
    raveled_combined = (normal + tangent).ravel("F")

    # Test that the projection of the combined vector onto the tangent space is the same
    # as the projection of the tangential vector
    assert np.allclose(
        proj.project_tangential() @ raveled_combined,
        proj.project_tangential() @ raveled_tangent,
        rtol=1e-14,
    )
    # Test that the projection of the combined vector onto the normal space is the same
    # as the projection of the normal vector
    assert np.allclose(
        proj.project_normal() @ raveled_combined,
        proj.project_normal() @ raveled_normal,
        rtol=1e-12,
    )
