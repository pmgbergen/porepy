"""Tests for grid utility functions.

These tests cover the computation of circumcenters in 2D and 3D grids,
including cases that require replacement of cell centers based on angle criteria,
degenerate triangles, and tetrahedra with circumcenters outside the cell.

"""

import numpy as np
import pytest

import porepy as pp
from porepy.utils.grid_utils import compute_circumcenters


@pytest.mark.parametrize(
    ["dim", "p", "tetris", "expected_shift"],
    [
        # 2D grids.
        (
            # 1 equilateral triangle, shift value should be 1, but centers should
            # coincide after computation.
            2,
            np.array([[0, 1, 0.5], [0, 0, np.sqrt(3.0) / 2.0]]),
            None,
            0.0,
        ),
        (
            # 2 equilateral triangle in diamond-constillation, shift value should be 1,
            # but centers should coincide after computation.
            2,
            np.array(
                [
                    [0.0, 1.0, 0.5, 0.5],
                    [0.0, 0.0, np.sqrt(3.0) / 2.0, -np.sqrt(3.0) / 2.0],
                ]
            ),
            np.array([[0, 0], [1, 1], [2, 3]]),
            0.0,
        ),
        (
            # 1 acute triangle, full shift.
            2,
            np.array([[0, 1, 0.5], [0, 0, 1]]),
            None,
            1.0,
        ),
        (
            # 2 acute triangles, in a diamond-like constellation. Full shift.
            2,
            np.array(
                [
                    [0.0, 1.0, 0.5, 0.5],
                    [0.0, 0.0, np.sqrt(3.0) / 2.0 + 1e-2, -np.sqrt(3.0) / 2.0 - 1e-2],
                ]
            ),
            np.array([[0, 0], [1, 1], [2, 3]]),
            1,
        ),
        (
            # 1 right triangle, shift is equal to default value of threshold argument
            # since circumcenter would be placed right on face.
            2,
            np.array([[0, 1, 0], [0, 0, 1]]),
            None,
            0.95,
        ),
        (
            # 2 right triangles, all shifts are equal to default value of threshold
            # argument.
            2,
            np.array([[0, 1, 0, 1], [0, 0, 1, 1]]),
            None,
            0.95,
        ),
        (
            # 1 obtuse triangle, specified shift.
            2,
            np.array([[0, 1, -0.5], [0, 0, 1]]),
            None,
            np.float64(0.2763636363636363),
        ),
        (
            # 2 obtuse triangles, specified shift.
            2,
            np.array([[0, 1, 2, 1.5], [0, 0, 0.5, 0]]),
            None,
            np.array([np.float64(0.0358490566037736), np.float64(0.11176470588235295)]),
        ),
        # 3D grids.
        (
            # 1 equilateral tetrahedron: barycenter = circumcenter.
            3,
            np.array(
                [
                    [1.0, -1.0, -1.0, 1.0],
                    [1.0, -1.0, 1.0, -1.0],
                    [1.0, 1.0, -1.0, -1.0],
                ]
            ),
            np.array([[0], [1], [2], [3]]),
            0.0,
        ),
        (
            # 1 acute tetrahedron: Expecting full shift.
            3,
            np.array(
                [
                    [1.0, -1.0, -1.0, 1.0],
                    [1.0, -1.0, 1.0, -1.0],
                    [1.0, 1.0, -1.2, -1.0],
                ]
            ),
            np.array([[0], [1], [2], [3]]),
            1.0,
        ),
        (
            # 1 right tetrahedron: Shift value equal to default threshold.
            3,
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0],
                    [0.0, -1.0, -1.0, -2.0],
                ]
            ),
            np.array([[0], [1], [2], [3]]),
            0.95,
        ),
        (
            # 3D normal simplex: Specific shift value.
            3,
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array([[0], [1], [2], [3]]),
            np.float64(0.31666666666666654),
        ),
        (
            # 2 normal simplices forming unit cube: the lower shifts as above, the
            # upper has barycenters equal to circumcenters.
            3,
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                ]
            ),
            np.array([[0, 1], [1, 2], [2, 3], [3, 4]]),
            np.array([np.float64(0.31666666666666654), 0.0]),
        ),
        (
            # 1 obtuse tetrahedron: Specified shift value.
            3,
            np.array(
                [
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array([[0], [1], [2], [3]]),
            np.float64(0.31666666666666665),
        ),
    ],
)
def test_compute_circumcenters(
    dim: int,
    p: np.ndarray,
    tetris: np.ndarray | None,
    expected_shift: np.ndarray | float,
) -> None:
    """Tests the circumcenter computation for triangular 2D grids.

    A triangle grid with default arguments will be created using the provided nodes.

    Parameters:
        dim: Dimension of simplex grid (2 or 3).
        p: 2D array of nodes of shape ``(dim, nc)``.
        tetris: Explicit triangulation/tetrahedron selection, if given.
        expected_shift: 1D of shift values of shape ``(num_cells,)`` or a float if
            shift is expected to be uniform.
            If None, asserts that the returned shift value is 1 and that the
            cell-centers did not change (only the case for equilaterals!).

    """
    # Computational absolute tolerance.
    tol = 1e-14

    if dim == 2:
        sd = pp.TriangleGrid(p, tri=tetris)
    elif dim == 3:
        sd = pp.TetrahedralGrid(p, tet=tetris)
    else:
        assert False, "Test set up for dimension 2 and 3 only."

    sd.compute_geometry()
    cc = sd.cell_centers.copy()
    nc = sd.num_cells

    new_cc, shift, is_changed = compute_circumcenters(sd, tol=tol)

    assert cc.shape == new_cc.shape, "Expecting same shapes for cell centers."
    assert shift.shape == (nc,), "Expecting array of shape (nc,)."
    assert is_changed.shape == (nc,), "Expecting array of shape (nc,)."
    assert np.all(shift >= 0) and np.all(shift <= 1.0), "Shifts not bound in (0,1)."

    if not isinstance(expected_shift, np.ndarray):
        expected_shift = np.ones(nc) * expected_shift

    assert isinstance(expected_shift, np.ndarray), "Expecting array at this point."
    assert expected_shift.shape == (nc,), "Expecting shift array of shape (nc,)."

    # Sanity check for indicators.
    assert np.all(is_changed[shift > tol]), (
        "Expecting change indicator where shift value greater 0."
    )
    assert np.all(~is_changed[shift <= tol]), (
        "Expecting no change indicator where shift value 0."
    )

    # Where shift is zero, new cell centers should coinice with old (bary) centers.
    idx = shift <= tol
    np.testing.assert_allclose(new_cc[:, idx], cc[:, idx], rtol=0, atol=tol)

    # Test shift values are as expected.
    np.testing.assert_allclose(shift, expected_shift, rtol=0, atol=tol)


def test_compute_circumcenters_raises_expected_errors():
    """The method should raise value errors if a grid is degenerate, or the threshold
    parameter is not in (0,  1)."""

    # Testing degenerate 3D grid.
    # Degenerate tetrahedron: all 4 points lie in the plane z = 0.
    points = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    tet = np.array([[0], [1], [2], [3]])
    sd = pp.TetrahedralGrid(points, tet)

    # compute_geometry may raise errors for degenerate grids. Manually set cell
    # centers to avoid that, expect error.
    sd.cell_centers = np.array([[0.25], [0.25], [0.0]])

    with pytest.raises(ValueError):
        compute_circumcenters(sd)

    # Testing degenerate 2D grid.
    p = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
        ]
    )
    tri = np.array([[0], [1], [2]])
    sd = pp.TriangleGrid(p, tri)
    sd.cell_centers = np.array([[1.0], [0.0], [0.0]])

    with pytest.raises(ValueError):
        compute_circumcenters(sd)

    # Testing threshold value bounds.
    for t in [-0.1, 0, 1, 1.1]:
        with pytest.raises(ValueError):
            compute_circumcenters(sd, threshold=t)

def _rotation_matrix_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )


def _rotation_matrix_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


def _rotation_matrix_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _triangle_type(p: np.ndarray) -> str:
    """Classify a triangle from points of shape (2, 3)."""
    a2 = np.sum((p[:, 1] - p[:, 2]) ** 2)
    b2 = np.sum((p[:, 0] - p[:, 2]) ** 2)
    c2 = np.sum((p[:, 0] - p[:, 1]) ** 2)
    s = np.sort(np.array([a2, b2, c2]))

    if np.isclose(s[0], s[1]) and np.isclose(s[1], s[2]):
        return "equilateral"
    if np.isclose(s[0] + s[1], s[2]):
        return "right"
    if s[0] + s[1] < s[2]:
        return "obtuse"
    return "acute"


def _make_rotated_triangle_grid(
    p2d: np.ndarray, tri: np.ndarray, R: np.ndarray
) -> pp.TriangleGrid:
    """Create a TriangleGrid and embed/rotate it in 3D."""
    sd = pp.TriangleGrid(p2d, tri=tri)
    p3d = np.vstack((p2d, np.zeros(p2d.shape[1])))
    sd.nodes = R @ p3d
    sd.compute_geometry()
    return sd


def test_compute_circumcenters_trianglegrid_rotation_invariant_in_3d():
    """Circumcenter construction should be invariant under rigid 3D rotations.

    The grid consists of four connected triangles in the plane z=0:
    - one right
    - one obtuse
    - one equilateral
    - one acute non-equilateral
    We compute the result in the original plane, then rotate the whole grid in 3D,
    compute again, inverse-rotate only the returned centers, and compare with the
    original result.
    """
    # Connected fan around the origin:
    # T0 = (0,1,2): right
    # T1 = (0,2,3): obtuse
    # T2 = (0,3,4): equilateral
    # T3 = (0,4,5): acute non-equilateral
    p2d = np.array(
        [
            [
                0.0,
                1.0,
                0.0,
                np.cos(np.deg2rad(210.0)),
                0.0,
                1.3 * np.cos(np.deg2rad(320.0)),
            ],
            [
                0.0,
                0.0,
                1.0,
                np.sin(np.deg2rad(210.0)),
                -1.0,
                1.3 * np.sin(np.deg2rad(320.0)),
            ],
        ]
    )
    tri = np.array([[0, 0, 0, 0], [1, 2, 3, 4], [2, 3, 4, 5]])

    # Sanity-check the intended triangle types.
    types = [_triangle_type(p2d[:, tri[:, i]]) for i in range(tri.shape[1])]
    assert types == ["right", "obtuse", "equilateral", "acute"]

    # Base result in the plane z = 0.
    sd0 = _make_rotated_triangle_grid(p2d, tri, np.eye(3))
    new_cc0, shift0, changed0 = compute_circumcenters(sd0)

    # Rotations about x, y, z, and one combined rotation.
    rotations = [
        _rotation_matrix_x(np.deg2rad(37.0)),
        _rotation_matrix_y(np.deg2rad(-29.0)),
        _rotation_matrix_z(np.deg2rad(61.0)),
        _rotation_matrix_z(np.deg2rad(23.0))
        @ _rotation_matrix_y(np.deg2rad(-41.0))
        @ _rotation_matrix_x(np.deg2rad(17.0)),
    ]

    for R in rotations:
        sd_rot = _make_rotated_triangle_grid(p2d, tri, R)
        new_cc_rot, shift_rot, changed_rot = compute_circumcenters(sd_rot)

        # Inverse-rotate only the returned centers.
        new_cc_back = R.T @ new_cc_rot

        np.testing.assert_allclose(new_cc_back, new_cc0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(shift_rot, shift0, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(changed_rot, changed0)