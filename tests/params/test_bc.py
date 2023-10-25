"""Tests for boundary conditions.

Currently only tests the bases for 2d and 3d Vectorial BCs.

"""
import numpy as np

import porepy as pp


def test_default_basis_2d():
    g = pp.StructuredTriangleGrid([1, 1])
    bc = pp.BoundaryConditionVectorial(g)
    basis_known = np.array(
        [
            [[1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
        ]
    )

    assert np.allclose(bc.basis, basis_known)


def test_default_basis_3d():
    g = pp.StructuredTetrahedralGrid([1, 1, 1])
    bc = pp.BoundaryConditionVectorial(g)
    basis_known = np.array(
        [
            [[1] * 18, [0] * 18, [0] * 18],
            [[0] * 18, [1] * 18, [0] * 18],
            [[0] * 18, [0] * 18, [1] * 18],
        ]
    )

    assert np.allclose(bc.basis, basis_known)
