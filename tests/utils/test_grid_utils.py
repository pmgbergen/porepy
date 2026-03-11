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




def _make_single_tetra_grid(points: np.ndarray) -> pp.TetrahedralGrid:
    """Create a single-cell tetrahedral grid from points of shape (3, 4)."""
    tet = np.array([[0], [1], [2], [3]])
    g = pp.TetrahedralGrid(points, tet)
    g.compute_geometry()
    return g

def _tetra_barycenter(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=1)

def _tetra_circumcenter(points: np.ndarray) -> np.ndarray:
    """Circumcenter of a tetrahedron with points shape (3, 4)."""
    A = points[:, 0]
    B = points[:, 1]
    C = points[:, 2]
    D = points[:, 3]

    M = np.vstack((B - A, C - A, D - A))
    rhs = 0.5 * np.array(
        [
            np.dot(B, B) - np.dot(A, A),
            np.dot(C, C) - np.dot(A, A),
            np.dot(D, D) - np.dot(A, A),
        ]
    )
    return np.linalg.solve(M, rhs)


def _tetra_barycentric_coords(points: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Return barycentric coordinates of p wrt tetrahedron points."""
    A = points[:, 0]
    B = points[:, 1]
    C = points[:, 2]
    D = points[:, 3]

    M = np.column_stack((B - A, C - A, D - A))
    u, v, w = np.linalg.solve(M, p - A)
    return np.array([1.0 - u - v - w, u, v, w])


def _point_in_tetra(points: np.ndarray, p: np.ndarray, tol: float = 1e-12) -> bool:
    lam = _tetra_barycentric_coords(points, p)
    return np.all(lam >= -tol) and np.all(lam <= 1.0 + tol)


def _face_data(points: np.ndarray):
    """Faces as (vertex_ids, opposite_id, name)."""
    return [
        ([1, 2, 3], 0, "BCD"),
        ([0, 3, 2], 1, "ADC"),
        ([0, 1, 3], 2, "ABD"),
        ([0, 2, 1], 3, "ACB"),  # same geometric face as ABC
    ]

def _outward_unit_normal(points: np.ndarray, face_ids: list[int], opp_id: int) -> np.ndarray:
    p0 = points[:, face_ids[0]]
    p1 = points[:, face_ids[1]]
    p2 = points[:, face_ids[2]]
    opp = points[:, opp_id]
    n = np.cross(p1 - p0, p2 - p0)
    if np.dot(n, opp - p0) > 0:
        n = -n
    return n / np.linalg.norm(n)


def _max_dot_shift(points: np.ndarray, threshold: float) -> float:
    G = _tetra_barycenter(points)
    Cc = _tetra_circumcenter(points)
    V = Cc - G

    best_dot = -np.inf
    best_t = None
    for face_ids, opp_id, _ in _face_data(points):
        n = _outward_unit_normal(points, face_ids, opp_id)
        p0 = points[:, face_ids[0]]
        denom = np.dot(n, V)
        if denom > best_dot:
            best_dot = denom
            if denom > 1e-14:
                best_t = np.dot(n, p0 - G) / denom

    assert best_t is not None
    return threshold * best_t


def test_compute_circumcenter_3d_regular_tetra_shift_one_not_changed():
    """Regular tetrahedron: circumcenter = barycenter = centroid."""
    points = np.array(
        [
            [1.0, -1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
        ]
    )
    sd = _make_single_tetra_grid(points)
    cc = sd.cell_centers.copy()

    new_cc, shift, is_changed = compute_circumcenter_3d(sd)

    np.testing.assert_allclose(shift, np.array([1.0]), rtol=0, atol=1e-14)
    np.testing.assert_allclose(new_cc, cc, rtol=0, atol=1e-14)
    assert np.all(~is_changed)

    Cc = _tetra_circumcenter(points)
    G = _tetra_barycenter(points)
    np.testing.assert_allclose(Cc, G, rtol=0, atol=1e-14)


def test_compute_circumcenter_3d_acute_nonregular_full_shift_changed():
    """Acute non-regular tetrahedron: circumcenter strictly inside, shift = 1."""
    points = np.array(
        [
            [1.0, -1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.2, -1.0],
        ]
    )
    sd = _make_single_tetra_grid(points)
    cc = sd.cell_centers.copy()

    new_cc, shift, is_changed = compute_circumcenter_3d(sd)

    Cc = _tetra_circumcenter(points)

    np.testing.assert_allclose(shift, np.array([1.0]), rtol=0, atol=1e-14)
    np.testing.assert_allclose(new_cc[:, 0], Cc, rtol=0, atol=1e-14)
    assert np.all(is_changed)

    # Nontrivial: circumcenter should differ from old center / barycenter.
    assert np.linalg.norm(Cc - cc[:, 0]) > 1e-3
    assert _point_in_tetra(points, Cc)


def test_compute_circumcenter_3d_boundary_circumcenter_shift_equals_threshold():
    """Circumcenter on a face: expected shift equals threshold."""
    points = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, -1.0, -1.0, -2.0],
        ]
    )
    threshold = 0.95
    sd = _make_single_tetra_grid(points)

    new_cc, shift, is_changed = compute_circumcenter_3d(sd, threshold=threshold)

    Cc = _tetra_circumcenter(points)
    G = _tetra_barycenter(points)
    expected_shift = _max_dot_shift(points, threshold)
    expected_center = G + expected_shift * (Cc - G)

    np.testing.assert_allclose(Cc, np.array([0.0, 0.75, -1.0]), rtol=0, atol=1e-14)
    np.testing.assert_allclose(shift, np.array([threshold]), rtol=0, atol=1e-14)
    np.testing.assert_allclose(shift, np.array([expected_shift]), rtol=0, atol=1e-14)
    np.testing.assert_allclose(new_cc[:, 0], expected_center, rtol=0, atol=1e-14)
    assert np.all(is_changed)
    assert _point_in_tetra(points, new_cc[:, 0])


def test_compute_circumcenter_3d_outside_circumcenter_partial_shift():
    """Circumcenter outside tetrahedron: expected partial shift in (0, 1)."""
    points = np.array(
        [
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    threshold = 0.95
    sd = _make_single_tetra_grid(points)

    new_cc, shift, is_changed = compute_circumcenter_3d(sd, threshold=threshold)

    Cc = _tetra_circumcenter(points)
    G = _tetra_barycenter(points)
    expected_shift = _max_dot_shift(points, threshold)
    expected_center = G + expected_shift * (Cc - G)

    assert not _point_in_tetra(points, Cc)
    assert 0.0 < expected_shift < 1.0

    np.testing.assert_allclose(shift, np.array([expected_shift]), rtol=0, atol=1e-14)
    np.testing.assert_allclose(new_cc[:, 0], expected_center, rtol=0, atol=1e-14)
    assert np.all(is_changed)
    assert _point_in_tetra(points, new_cc[:, 0])


def test_compute_circumcenter_3d_two_tetras_full_shift():
    """Two tetrahedra sharing one face; both have interior circumcenters.

    This checks multi-cell handling, full shifts, and that the returned centers
    equal the true circumcenters cell-by-cell.
    """
    points = np.array(
        [
            [0.0, 1.0, 0.0, 0.63421812, 0.69305481],
            [0.0, 0.0, 1.0, 0.38912899, 0.45283182],
            [0.0, 0.0, 0.0, 0.96604107, -0.81093435],
        ]
    )
    # Two tetrahedra sharing face (0, 1, 2):
    # cell 0 = (0,1,2,3), cell 1 = (0,1,2,4)
    tet = np.array([[0, 0], [1, 1], [2, 2], [3, 4]])

    sd = pp.TetrahedralGrid(points, tet)
    sd.compute_geometry()
    cc = sd.cell_centers.copy()

    new_cc, shift, is_changed = compute_circumcenter_3d(sd)

    assert new_cc.shape == cc.shape
    assert shift.shape == (2,)
    assert is_changed.shape == (2,)

    np.testing.assert_allclose(shift, np.ones(2), rtol=0, atol=1e-14)
    assert np.all(is_changed)

    # Check returned centers against true circumcenters of the two cells.
    for i, cell_nodes in enumerate(tet.T):
        pts = points[:, cell_nodes]
        Cc = _tetra_circumcenter(pts)
        np.testing.assert_allclose(new_cc[:, i], Cc, rtol=0, atol=1e-14)
        assert _point_in_tetra(pts, Cc)
        assert np.linalg.norm(new_cc[:, i] - cc[:, i]) > 1e-8


def test_compute_circumcenter_3d_raises_expected_errors():
    """The method should raise value errors if a grid is degenerate, or the threshold
    parameter is not in (0, 1).
    """
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

    # compute_geometry may raise for degenerate tetrahedra. As in the 2D test,
    # manually set cell centers so the function itself is what raises.
    sd.cell_centers = np.array([[0.25], [0.25], [0.0]])

    with pytest.raises(ValueError):
        compute_circumcenter_3d(sd)

    # Invalid threshold values.
    # Use a valid tetrahedron here so the threshold check is what is tested.
    good_points = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    good_sd = _make_single_tetra_grid(good_points)

    for t in [-0.1, 0.0, 1.0, 1.1]:
        with pytest.raises(ValueError):
            compute_circumcenter_3d(good_sd, threshold=t)


def test_compute_circumcenter_3d_two_tetras_mixed_shift():
    """Two tetrahedra sharing a face where one has interior circumcenter
    and the other has exterior circumcenter."""
    
    points = np.array(
        [
            [0.0, 1.0, 0.0, 0.56928343, 0.77728280],
            [0.0, 0.0, 1.0, 0.53093376, 1.70282285],
            [0.0, 0.0, 0.0, 0.73827729, -0.13988499],
        ]
    )

    # Shared face is (0, 1, 2), but reversed in the second tetra.
    tet = np.array([[0, 0], [1, 2], [2, 1], [3, 4]])

    sd = pp.TetrahedralGrid(points, tet)
    sd.compute_geometry()
    cc = sd.cell_centers.copy()

    threshold = 0.95
    new_cc, shift, is_changed = compute_circumcenter_3d(sd, threshold=threshold)

    assert new_cc.shape == cc.shape
    assert shift.shape == (2,)
    assert is_changed.shape == (2,)

    # --- cell 0 ---
    pts0 = points[:, tet[:, 0]]
    Cc0 = _tetra_circumcenter(pts0)

    assert _point_in_tetra(pts0, Cc0)
    np.testing.assert_allclose(shift[0], 1.0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(new_cc[:, 0], Cc0, rtol=0, atol=1e-14)
    assert is_changed[0]

    # --- cell 1 ---
    pts1 = points[:, tet[:, 1]]
    Cc1 = _tetra_circumcenter(pts1)

    assert not _point_in_tetra(pts1, Cc1)

    G1 = _tetra_barycenter(pts1)
    expected_shift = _max_dot_shift(pts1, threshold)
    expected_center = G1 + expected_shift * (Cc1 - G1)

    np.testing.assert_allclose(shift[1], expected_shift, rtol=0, atol=1e-14)
    np.testing.assert_allclose(new_cc[:, 1], expected_center, rtol=0, atol=1e-14)

    assert is_changed[1]
    assert _point_in_tetra(pts1, new_cc[:, 1])