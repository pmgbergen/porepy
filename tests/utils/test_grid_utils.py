"""Tests for grid utility functions.

These tests cover the computation of circumcenters in 2D and 3D grids,
including cases that require replacement of cell centers based on angle criteria,
degenerate triangles, and tetrahedra with circumcenters outside the cell.

"""

import numpy as np
import pytest
import porepy as pp

from porepy.utils.grid_utils import compute_circumcenter_2d, compute_circumcenter_3d


def test_compute_circumcenter_2d_equilateral_triangle_replaces_center():
    # Equilateral triangle: angles are 60° < 0.45*pi, so replacement should occur.
    h = np.sqrt(3.0) / 2.0
    p = np.array(
        [
            [0.0, 1.0, 0.5],
            [0.0, 0.0, h],
        ]
    )
    tri = np.array([[0], [1], [2]])
    g = pp.TriangleGrid(p, tri)
    g.compute_geometry()

    cc_new, replace = compute_circumcenter_2d(g)

    assert replace.size == 1 and bool(replace[0])
    # Circumcenter of equilateral triangle equals centroid.
    expected = np.array([0.5, h / 3.0])
    assert np.allclose(cc_new[:2, 0], expected, rtol=1e-12, atol=1e-12)
    # z-coordinate should remain zero.
    assert np.isclose(cc_new[2, 0], 0.0)


def test_compute_circumcenter_2d_right_triangle_no_replacement():
    # Right isosceles triangle from a unit right triangle has a 90° angle
    # which is > 0.45*pi threshold; no replacement should occur.
    points = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    tri = np.array([[0], [1], [2]])
    g = pp.TriangleGrid(points, tri)
    g.compute_geometry()
    cc_original = g.cell_centers.copy()
    cc_new, replace = compute_circumcenter_2d(g)
    assert replace.size == 1 and not bool(replace[0])
    assert np.allclose(cc_new, cc_original, rtol=1e-13, atol=1e-13)


def test_compute_circumcenter_2d_degenerate_raises():
    # Colinear points -> degenerate triangle; function should raise ValueError.
    p = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
        ]
    )
    tri = np.array([[0], [1], [2]])
    g = pp.TriangleGrid(p, tri)
    # compute_geometry raises for a degenerate (colinear) triangle. Manually set cell
    # centers to avoid that. The value does not matter, any point will take us to the
    # line in compute_circumcenter_2d raising the error.
    g.cell_centers = np.array([[1.0], [0.0], [0.0]])
    with pytest.raises((ValueError)):
        compute_circumcenter_2d(g)


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

@pytest.mark.parametrize("perturb", [1e-2, 0.0])
def test_compute_circumcenter_2d_two_triangles_internal_alignment(perturb):
    """Two acute triangles sharing an edge: both should be replaced.

    Use two equilateral triangles sharing an edge (forming a symmetric "diamond").
    All internal angles are 60°, well below the default threshold (~81°). We verify
    both centers replaced and that the vector between circumcenters is perpendicular
    to the shared edge.
    """
    # Equilateral triangle side length 1: height sqrt(3)/2.
    h = np.sqrt(3.0) / 2.0
    # Perturb the height slightly to avoid perfect symmetry that could mask issues and
    # would imply no replacement needed, i.e., circumcenters equal centroids.
    h += perturb
    # Points: A(0,0), B(1,0), C(0.5,h) (upper), D(0.5,-h) (lower)
    pts = np.array(
        [
            [0.0, 1.0, 0.5, 0.5],  # x
            [0.0, 0.0, h, -h],     # y
        ]
    )
    # Triangles: T0 = (A,B,C) and T1 = (A,B,D) share edge (A,B) indices (0,1)
    tris = np.array([[0, 0], [1, 1], [2, 3]])
    g = pp.TriangleGrid(pts, tris)
    g.compute_geometry()
    cc_new, replace = compute_circumcenter_2d(g)
    assert replace.size == 2 and bool(replace[0]) and bool(replace[1])
    # Shared edge endpoints indices 0 and 1 (A,B).
    pA = pts[:, 0]
    pB = pts[:, 1]
    edge_vec = pB - pA
    # Vector between circumcenters (should be vertical, perpendicular to AB)
    cc_vec = cc_new[:2, 0] - cc_new[:2, 1]
    # Check perpendicular: dot ~ 0
    dot = float(np.dot(edge_vec, cc_vec))
    norm_prod = np.linalg.norm(edge_vec) * np.linalg.norm(cc_vec) + 1e-15
    assert abs(dot) / norm_prod < 1e-10
    match = np.allclose(cc_new, g.cell_centers, rtol=1e-13, atol=1e-13)
    if perturb != 0.0:
        # Check that new and old centers differ.
        assert not match
    else:
        # In the unperturbed case, circumcenters equal centroids.
        assert match
