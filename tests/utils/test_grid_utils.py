"""Tests for grid utility functions.

These tests cover the computation of circumcenters in 2D and 3D grids,
including cases that require replacement of cell centers based on angle criteria,
degenerate triangles, and tetrahedra with circumcenters outside the cell.

"""

import numpy as np
import pytest

import porepy as pp
from porepy.utils.grid_utils import compute_circumcenter_2d, compute_circumcenter_3d


@pytest.mark.parametrize(
    ["p", "tri", "expected_shift"],
    [
        (
            # 1 equilateral triangle, shift value should be 1, but centers should
            # coincide after computation.
            np.array([[0, 1, 0.5], [0, 0, np.sqrt(3.0) / 2.0]]),
            None,
            None,
        ),
        (
            # 2 equilateral triangle in diamond-constillation, shift value should be 1,
            # but centers should coincide after computation.
            np.array(
                [
                    [0.0, 1.0, 0.5, 0.5],
                    [0.0, 0.0, np.sqrt(3.0) / 2.0, -np.sqrt(3.0) / 2.0],
                ]
            ),
            np.array([[0, 0], [1, 1], [2, 3]]),
            None,
        ),
        (
            # 1 acute triangle, full shift.
            np.array([[0, 1, 0.5], [0, 0, 1]]),
            None,
            1.0,
        ),
        (
            # 2 acute triangles, in a diamond-like constellation. Full shift.
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
            # 1 right triangles, shift is equal to default value of threshold argument
            # since circumcenter would be placed right on face.
            np.array([[0, 1, 0], [0, 0, 1]]),
            None,
            0.95,
        ),
        (
            # 2 right triangles, all shifts are equal to default value of threshold
            # argument.
            np.array([[0, 1, 0, 1], [0, 0, 1, 1]]),
            None,
            0.95,
        ),
        (
            # 1 obtuse triangle, specified shift.
            np.array([[0, 1, -0.5], [0, 0, 1]]),
            None,
            np.float64(0.2763636363636363),
        ),
        (
            # 2 obtuse triangles, specified shift.
            np.array([[0, 1, 2, 1.5], [0, 0, 0.5, 0]]),
            None,
            np.array([np.float64(0.0358490566037736), np.float64(0.11176470588235295)]),
        ),
    ],
)
def test_compute_circumcenter_2d(
    p: np.ndarray,
    tri: np.ndarray | None,
    expected_shift: np.ndarray | float | None,
) -> None:
    """Tests the circumcenter computation for triangular 2D grids.

    A triangle grid with default arguments will be created using the provided nodes.

    Parameters:
        p: 2D array of nodes of shape ``(2, N)``, with ``N>=3``.
        tri: Explicit triangulation, if given.
        expected_shift: 1D of shift values of shape ``(num_cells,)`` or a float if
            shift is expected to be uniform.
            If None, asserts that the returned shift value is 1 and that the
            cell-centers did not change (only the case for equilaterals!).

    """
    sd = pp.TriangleGrid(p, tri=tri)
    sd.compute_geometry()
    cc = sd.cell_centers

    new_cc, shift, is_changed = compute_circumcenter_2d(sd)

    assert cc.shape == new_cc.shape, "Expecting same shapes."
    if expected_shift is not None:
        if not isinstance(expected_shift, np.ndarray):
            expected_shift = np.ones(sd.num_cells) * expected_shift
        np.testing.assert_allclose(shift, expected_shift, rtol=0, atol=1e-14)
        assert np.all(is_changed)
    else:
        np.testing.assert_allclose(shift, 1.0, rtol=0, atol=1e-14)
        np.testing.assert_allclose(new_cc, cc, rtol=0, atol=1e-14)
        assert np.all(~is_changed)


def test_compute_circumcenter_2d_raises_expected_errors():
    """The method should raise value errors if a grid is degenerate, or the threshold
    parameter is not in (0,  1)."""
    p = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
        ]
    )
    tri = np.array([[0], [1], [2]])
    sd = pp.TriangleGrid(p, tri)
    # compute_geometry raises for a degenerate (colinear) triangle. Manually set cell
    # centers to avoid that. The value does not matter, any point will take us to the
    # line in compute_circumcenter_2d raising the error.
    sd.cell_centers = np.array([[1.0], [0.0], [0.0]])

    with pytest.raises(ValueError):
        compute_circumcenter_2d(sd)

    for t in [-0.1, 0, 1, 1.1]:
        with pytest.raises(ValueError):
            compute_circumcenter_2d(sd, threshold=t)


def test_compute_circumcenter_3d_regular_tetrahedron_replaces_and_matches():
    # Regular tetrahedron with circumcenter at the origin.
    pts = np.array(
        [
            [1.0, -1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
        ]
    )
    tet = np.array([[0], [1], [2], [3]])
    g = pp.TetrahedralGrid(pts, tet)
    g.compute_geometry()

    cc_new, replace = compute_circumcenter_3d(g)

    assert replace.size == 1 and bool(replace[0])
    # Expected circumcenter at the origin for this symmetric tetrahedron.
    expected = np.array([0.0, 0.0, 0.0])
    assert np.allclose(cc_new[:, 0], expected, rtol=1e-12, atol=1e-12)


def test_compute_circumcenter_3d_outside_tetra_no_replacement():
    # Create an obtuse tetrahedron where the circumcenter is outside the cell.
    # No replacement should occur.
    pts = np.array(
        [
            [0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 0.01],
        ]
    )
    tet = np.array([[0], [1], [2], [3]])
    g = pp.TetrahedralGrid(pts, tet)
    g.compute_geometry()

    _cc_new, replace = compute_circumcenter_3d(g)

    assert replace.size == 1 and not bool(replace[0])


def test_compute_circumcenter_3d_benign_inside_replaces_and_not_centroid():
    # Skewed but acute-ish tetra where circumcenter is inside; replacement occurs
    # and circumcenter differs from centroid.
    # Start from a regular tetra and perturb one vertex slightly to keep it acute
    # while moving the circumcenter away from the centroid
    points = np.array(
        [
            [1.0, -1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.2, -1.0],
        ]
    )
    tet = np.array([[0], [1], [2], [3]])
    g = pp.TetrahedralGrid(points, tet)
    g.compute_geometry()
    new_centers, replace = compute_circumcenter_3d(g)
    assert replace.size == 1 and bool(replace[0])
    centroid = np.mean(points, axis=1)
    # Ensure circumcenter is not the centroid (nontrivial case).
    assert np.linalg.norm(new_centers[:, 0] - centroid) > 1e-3


def test_compute_circumcenter_3d_large_dihedral_angle_no_replacement():
    """Tetrahedron with one very flat face pair producing a large dihedral angle.

    By making one edge extremely long while keeping the opposite face small, we can
    induce a dihedral angle between two faces that exceeds the default threshold
    (0.45*pi). Expect no replacement.
    """
    # Construct points: Start with a near-regular base triangle and stretch one
    # vertex far along x to create a large angle between faces sharing that edge.
    pts = np.array(
        [
            [0.0, 1.0, 0.0, 10.0],  # Stretch last vertex far in x
            [0.0, 0.0, 1.0, 0.2],
            [0.0, 0.0, 0.0, 0.1],
        ]
    )
    tet = np.array([[0], [1], [2], [3]])
    g = pp.TetrahedralGrid(pts, tet)
    g.compute_geometry()
    cc_new, replace = compute_circumcenter_3d(g)
    assert replace.size == 1 and not bool(replace[0])
    # If no replacement, cc_new should match old cell center.
    assert np.allclose(cc_new, g.cell_centers, rtol=1e-13, atol=1e-13)


def test_compute_circumcenter_3d_two_tetra_internal_alignment():
    """Create two tetrahedra sharing a face and test alignment and replacement.

    We build two moderately acute tetrahedra sharing a face; both should satisfy the
    dihedral threshold so both are replaced. Then verify the circumcenter vector
    across the internal face is parallel to the face normal (as enforced in the
    utility function). We cannot directly access the internal check's intermediate
    data, but we can recompute the cross product and ensure near-zero magnitude.
    """
    # Build two regular tetrahedra sharing the same equilateral base face.
    s = 1.0
    h = np.sqrt(2.0 / 3.0) * s  # Height for a regular tetra with side length s
    A = np.array([[0.0], [0.0], [0.0]])
    B = np.array([[s], [0.0], [0.0]])
    C = np.array([[0.5 * s], [np.sqrt(3.0) / 2.0 * s], [0.0]])
    centroid_xy = np.array([[0.5 * s], [np.sqrt(3.0) / 6.0 * s], [0.0]])
    apex_up = centroid_xy + np.array([[0.0], [0.0], [h]])
    apex_down = centroid_xy + np.array([[0.0], [0.0], [-h]])
    pts = np.concatenate((A, B, C, apex_up, apex_down), axis=1)
    # Two tetrahedra: (0,1,2,3) and (0,1,2,4) share face (0,1,2).
    tets = np.array([[0, 0], [1, 1], [2, 2], [3, 4]])
    g = pp.TetrahedralGrid(pts, tets)
    g.compute_geometry()
    cc_new, replace = compute_circumcenter_3d(g)
    # Both replaced expected.
    assert replace.size == 2 and bool(replace[0]) and bool(replace[1])
    # Internal face normal should be parallel to difference of circumcenters.
    # Identify shared face nodes (0,1,2). Compute face normal via cross product of
    # edges using original coordinates.
    p0, p1, p2 = pts[:, 0], pts[:, 1], pts[:, 2]
    v1 = p1 - p0
    v2 = p2 - p0
    face_normal = np.cross(v1, v2)
    face_normal /= np.linalg.norm(face_normal) + 1e-15
    cc_vec = cc_new[:, 0] - cc_new[:, 1]
    cross_mag = np.linalg.norm(np.cross(cc_vec, face_normal))
    denom = np.linalg.norm(cc_vec) * np.linalg.norm(face_normal) + 1e-15
    # Use same tolerance as implementation (1e-10), with small safety factor.
    assert cross_mag / denom < 5e-10
