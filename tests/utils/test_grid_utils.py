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
	# Equilateral triangle: angles are 60° < 0.45*pi, so replacement should occur
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
	# Circumcenter of equilateral triangle equals centroid
	expected = np.array([0.5, h / 3.0])
	assert np.allclose(cc_new[:2, 0], expected, rtol=1e-12, atol=1e-12)
	# z-coordinate should remain zero
	assert np.isclose(cc_new[2, 0], 0.0)


def test_compute_circumcenter_2d_right_triangle_no_replacement():
    # Right isosceles triangle from a unit right triangle has a 90° angle
    # which is > 0.45*pi threshold; no replacement should occur
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
    # Colinear points -> degenerate triangle; function should raise ValueError
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
	# Regular tetrahedron with circumcenter at the origin
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
	# Expected circumcenter at the origin for this symmetric tetrahedron
	expected = np.array([0.0, 0.0, 0.0])
	assert np.allclose(cc_new[:, 0], expected, rtol=1e-12, atol=1e-12)


def test_compute_circumcenter_3d_outside_tetra_no_replacement():
	# Create an obtuse tetrahedron where the circumcenter is outside the cell
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
    # and circumcenter differs from centroid
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
    # Ensure circumcenter is not the centroid (nontrivial case)
    assert np.linalg.norm(new_centers[:, 0] - centroid) > 1e-3
