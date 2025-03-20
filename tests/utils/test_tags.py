"""Tests of tagging of mesh quantities.

The tags are tested for simplex and cartesian grids, grids of all three dimensions,
mixed dimensional grids and fractured/non-fractured domains.

The tests have been developed gradually, and are not coherent, but should give
reasonable coverage.
"""

import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils import reference_dense_arrays
from porepy.fracs import meshing


def _compare_tip_nodes(g, known_tips):
    # Compare nodes tagged as being on a fracture boundary (seen from the
    # grid in the host medium) and known coordinates of such nodes.

    tip_ind = []

    # For each known coordinate find its closest representation in the grid.
    # This should be more or less identical, or something is wrong
    for i in range(known_tips.shape[1]):
        p = known_tips[:, i].reshape((-1, 1))
        tip_ind.append(np.argmin(np.sum(np.power(p - g.nodes, 2), axis=0)))

    tip_ind = np.array(tip_ind)

    # Check that the nodes with the known coordinates are tagged as tips
    is_tip = g.tags["node_is_fracture_tip"]
    assert np.all(is_tip[tip_ind])

    # Check that all other nodes are tagged as non-tips
    other_ind = np.setdiff1d(np.arange(g.num_nodes), tip_ind)
    assert np.all(np.logical_not(is_tip[other_ind]))


def test_node_is_fracture_tip_2d():
    # Test that nodes in the highest dimensional grids are correctly labeled as tip nodes

    f1 = np.array([[1, 3], [2, 2]])
    f2 = np.array([[1, 3], [3, 3]])
    f3 = np.array([[3, 3], [1, 2]])
    f4 = np.array([[2, 2], [2, 4]])

    # T-intersection between 1 and 4.
    # L-intersection between 1 and 3
    # X-intersection between 2 and 4
    # 2 has two endings in the domain 1 and 3 have one ending in the domain.
    # 4 has one node at the domain boundary - should not be a tip node

    fracs = [f1, f2, f3, f4]
    mdg = pp.meshing.cart_grid(fracs, nx=np.array([4, 4]))

    sd = mdg.subdomains(dim=2)[0]

    # Base comparison on coordinates (safe on Cartesian grids), then we don't have to deal
    # with changing node indices
    known_tips = np.array([[1, 1, 3, 3], [2, 3, 3, 1], [0, 0, 0, 0]])

    _compare_tip_nodes(sd, known_tips)


def test_node_is_fracture_tip_3d():
    dims = np.array([6, 5, 5])

    # f1 is isolated, all tip nodes should be marked in gh
    f1 = np.array([[4, 4, 4, 4], [2, 3, 3, 2], [1, 1, 2, 2]])

    # f2 has several intersections, see below for description
    f2 = np.array([[2, 2, 2, 2], [1, 4, 4, 1], [1, 1, 4, 4]])

    # Nodes 1 and 2 of f3 are tips, 0 and 3 ends in a T-intersection with f2
    f3 = np.array([[2, 3, 3, 2], [2, 2, 2, 2], [2, 2, 3, 3]])

    # f4 has an X-intersection with f2, and L-intersection with f5
    f4 = np.array([[1, 3, 3, 1], [3, 3, 3, 3], [1, 1, 3, 3]])

    # f5 has an L-intersection with f4, but since f5 is taller than f4, the
    # node at (1, 3, 4) is a tip. f5 extends to the domain boundary, so the
    # nodes at z=1 and z=4 are tips, but not z=2, z=3.
    f5 = np.array([[1, 1, 1, 1], [3, 3, 5, 5], [1, 4, 4, 1]])

    mdg = pp.meshing.cart_grid([f1, f2, f3, f4, f5], dims)
    sd = mdg.subdomains(dim=3)[0]

    # Gather the tip nodes from one fracture at a time
    known_tips_1 = f1
    known_tips_2 = np.array(
        [
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 2],
            [1, 2, 3, 4, 4, 4, 4, 3, 2, 1, 1],
        ]
    )
    known_tips_3 = np.array([[3, 3], [2, 2], [2, 3]])
    known_tips_4 = np.array([[3, 3, 3], [3, 3, 3], [1, 2, 3]])
    known_tips_5 = np.array([[1, 1, 1, 1, 1], [3, 4, 5, 5, 4], [4, 4, 4, 1, 1]])

    known_tips = np.hstack(
        (known_tips_1, known_tips_2, known_tips_3, known_tips_4, known_tips_5)
    )
    _compare_tip_nodes(sd, known_tips)


# Here follows several functions that return grids and known faces/nodes. They are used
# for testing the tags of non-fractured domains further down in this file.


def func_1d():
    g = pp.CartGrid(3, 1)
    known_faces = known_nodes = [True, False, False, True]
    return g, known_faces, known_nodes


def func_2d_simplex():
    g = pp.StructuredTriangleGrid([3] * 2, [1] * 2)
    known_faces = reference_dense_arrays.test_tags["func_2d_simplex"]["known_faces"]
    known_nodes = reference_dense_arrays.test_tags["func_2d_simplex"]["known_nodes"]
    return g, known_faces, known_nodes


def func_2d_cartesian():
    g = pp.CartGrid([4] * 2, [1] * 2)
    known_faces = reference_dense_arrays.test_tags["func_2d_cartesian"]["known_faces"]
    known_nodes = reference_dense_arrays.test_tags["func_2d_cartesian"]["known_nodes"]
    return g, known_faces, known_nodes


def func_3d_simplex():
    g = pp.StructuredTetrahedralGrid([2] * 3, [1] * 3)
    known_faces = reference_dense_arrays.test_tags["func_3d_simplex"]["known_faces"]
    known_nodes = reference_dense_arrays.test_tags["func_3d_simplex"]["known_nodes"]

    return g, known_faces, known_nodes


def func_3d_cartesian():
    g = pp.CartGrid([4] * 3, [1] * 3)
    known_faces = reference_dense_arrays.test_tags["func_3d_cartesian"]["known_faces"]
    known_nodes = reference_dense_arrays.test_tags["func_3d_cartesian"]["known_nodes"]
    return g, known_faces, known_nodes


@pytest.mark.parametrize(
    "g, known_faces, known_nodes",
    [
        func_1d(),
        func_2d_simplex(),
        func_2d_cartesian(),
        func_3d_simplex(),
        func_3d_cartesian(),
    ],
)
def test_face_and_node_tags_for_non_fractured_domains(g, known_faces, known_nodes):
    """Tests tags of 1d/2d/3d simplex and cartesian grids with no fractures."""

    # Makes sure that there are no faces or nodes that are related to fractures/fracture
    # tips. A fractureless domain should not result in fracture-related tags being True.
    assert np.array_equal(g.tags["fracture_faces"], [False] * g.num_faces)
    assert np.array_equal(g.tags["fracture_nodes"], [False] * g.num_nodes)

    assert np.array_equal(g.tags["tip_faces"], [False] * g.num_faces)
    assert np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes)

    # Asserts that the faces and nodes with tags "domain_boundary_faces" and
    # "domain_boundary_nodes" are as they are expected to be.
    assert np.array_equal(g.tags["domain_boundary_faces"], known_faces)
    assert np.array_equal(g.tags["domain_boundary_nodes"], known_nodes)


# ------------------------------------------------------------------------------#


def test_tag_2d_1d_cartesian_with_one_fracture():
    """Testing tags for 2d domain with a 1d fracture."""
    mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "cartesian",
        {"cell_size": 1 / 4},
        fracture_indices=[1],
    )

    for sd in mdg.subdomains():
        if sd.dim == 1:
            assert np.array_equal(sd.tags["fracture_faces"], [False] * sd.num_faces)
            assert np.array_equal(sd.tags["fracture_nodes"], [False] * sd.num_nodes)
            assert np.array_equal(sd.tags["tip_faces"], [False] * sd.num_faces)
            assert np.array_equal(sd.tags["tip_nodes"], [False] * sd.num_nodes)
            known = [0, 4]
            computed = np.where(sd.tags["domain_boundary_faces"])[0]
            assert np.array_equal(computed, known)
            known = [0, 4]
            computed = np.where(sd.tags["domain_boundary_nodes"])[0]
            assert np.array_equal(computed, known)

        if sd.dim == 2:
            known = [28, 29, 30, 31, 40, 41, 42, 43]
            computed = np.where(sd.tags["fracture_faces"])[0]
            assert np.array_equal(computed, known)
            known = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            computed = np.where(sd.tags["fracture_nodes"])[0]

            assert np.array_equal(computed, known)
            assert np.array_equal(sd.tags["tip_faces"], [False] * sd.num_faces)
            assert np.array_equal(sd.tags["tip_nodes"], [False] * sd.num_nodes)
            known = [0, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 36, 37, 38, 39]
            computed = np.where(sd.tags["domain_boundary_faces"])[0]
            assert np.array_equal(computed, known)
            known = [
                0,
                1,
                2,
                3,
                4,
                5,
                9,
                10,
                11,
                18,
                19,
                20,
                24,
                25,
                26,
                27,
                28,
                29,
            ]
            computed = np.where(sd.tags["domain_boundary_nodes"])[0]
            assert np.array_equal(computed, known)


def test_tags_2d_1d_cartesian_with_crossing_fractures():
    mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "cartesian",
        meshing_args={"cell_size_x": 0.25, "cell_size_y": 0.25},
        fracture_indices=[0, 1],
        fracture_endpoints=[np.array([0.25, 0.75]), np.array([0, 1])],
    )

    for sd in mdg.subdomains():
        if sd.dim == 0:
            assert np.sum(sd.tags["fracture_faces"]) == 0
            assert np.sum(sd.tags["fracture_nodes"]) == 0
            assert np.sum(sd.tags["tip_faces"]) == 0
            assert np.sum(sd.tags["tip_nodes"]) == 0
            assert np.sum(sd.tags["domain_boundary_faces"]) == 0
            assert np.sum(sd.tags["domain_boundary_nodes"]) == 0

        if sd.dim == 1 and sd.nodes[1, 0] == 0.5:
            known = [2, 5]
            computed = np.where(sd.tags["fracture_faces"])[0]
            assert np.array_equal(known, computed)
            known = [2, 3]
            computed = np.where(sd.tags["fracture_nodes"])[0]
            assert np.array_equal(known, computed)
            assert np.array_equal(sd.tags["tip_faces"], [False] * sd.num_faces)
            assert np.array_equal(sd.tags["tip_nodes"], [False] * sd.num_nodes)
            known = [0, 4]
            computed = np.where(sd.tags["domain_boundary_faces"])[0]
            assert np.array_equal(computed, known)
            known = [0, 5]
            computed = np.where(sd.tags["domain_boundary_nodes"])[0]
            assert np.array_equal(computed, known)

        if sd.dim == 1 and sd.nodes[0, 0] == 0.5:
            known = [1, 3]
            computed = np.where(sd.tags["fracture_faces"])[0]
            assert np.array_equal(known, computed)
            known = [1, 2]
            computed = np.where(sd.tags["fracture_nodes"])[0]
            assert np.array_equal(known, computed)
            known = [0, 2]
            computed = np.where(sd.tags["tip_faces"])[0]
            assert np.array_equal(known, computed)
            known = [0, 3]
            computed = np.where(sd.tags["tip_nodes"])[0]
            assert np.array_equal(known, computed)
            assert np.sum(sd.tags["domain_boundary_faces"]) == 0
            assert np.sum(sd.tags["domain_boundary_nodes"]) == 0

        if sd.dim == 2:
            known = [7, 12, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45]
            computed = np.where(sd.tags["fracture_faces"])[0]
            assert np.array_equal(computed, known)
            known = [7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24]
            computed = np.where(sd.tags["fracture_nodes"])[0]
            assert np.array_equal(computed, known)
            assert np.array_equal(sd.tags["tip_faces"], [False] * sd.num_faces)
            assert np.array_equal(sd.tags["tip_nodes"], [False] * sd.num_nodes)
            known = [0, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 36, 37, 38, 39]
            computed = np.where(sd.tags["domain_boundary_faces"])[0]
            assert np.array_equal(computed, known)
            known = [
                0,
                1,
                2,
                3,
                4,
                5,
                9,
                10,
                11,
                20,
                21,
                22,
                26,
                27,
                28,
                29,
                30,
                31,
            ]
            computed = np.where(sd.tags["domain_boundary_nodes"])[0]
            assert np.array_equal(computed, known)


# ------------------------------------------------------------------------------#


def test_x_intersection_2d():
    """Check that the faces has correct tags for a 2D grid."""

    f_1 = np.array([[0, 2], [1, 1]])
    f_2 = np.array([[1, 1], [0, 2]])

    f_set = [f_1, f_2]
    nx = [3, 3]

    grids = meshing.cart_grid(f_set, nx, physdims=nx)

    # 2D grid:
    g_2d = grids.subdomains(dim=2)[0]

    f_tags_2d = np.array(
        [
            False,
            True,
            False,
            False,  # first row
            False,
            True,
            False,
            False,  # Second row
            False,
            False,
            False,
            False,  # third row
            False,
            False,
            False,  # Bottom column
            True,
            True,
            False,  # Second column
            False,
            False,
            False,  # Third column
            False,
            False,
            False,  # Top column
            True,
            True,
            True,
            True,
        ]
    )  # Added faces

    d_tags_2d = np.array(
        [
            True,
            False,
            False,
            True,  # first row
            True,
            False,
            False,
            True,  # Second row
            True,
            False,
            False,
            True,  # third row
            True,
            True,
            True,  # Bottom column
            False,
            False,
            False,  # Second column
            False,
            False,
            False,  # Third column
            True,
            True,
            True,  # Top column
            False,
            False,
            False,
            False,
        ]
    )  # Added Faces
    t_tags_2d = np.zeros(f_tags_2d.size, dtype=bool)

    assert np.all(g_2d.tags["tip_faces"] == t_tags_2d)
    assert np.all(g_2d.tags["fracture_faces"] == f_tags_2d)
    assert np.all(g_2d.tags["domain_boundary_faces"] == d_tags_2d)

    # 1D grids:
    for g_1d in grids.subdomains(dim=1):
        f_tags_1d = np.array([False, True, False, True])
        if g_1d.face_centers[0, 0] > 0.1:
            t_tags_1d = np.array([True, False, False, False])
            d_tags_1d = np.array([False, False, True, False])
        else:
            t_tags_1d = np.array([False, False, True, False])
            d_tags_1d = np.array([True, False, False, False])

        assert np.all(g_1d.tags["tip_faces"] == t_tags_1d)
        assert np.all(g_1d.tags["fracture_faces"] == f_tags_1d)
        assert np.all(g_1d.tags["domain_boundary_faces"] == d_tags_1d)
