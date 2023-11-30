"""Unit tests for grid extrusion functionality in multiple dimensions and some
mixed-dimensional cases."""

import numpy as np
import pytest
import scipy.sparse as sps
from porepy.utils.array_operations import sparse_array_to_row_col_data
import porepy as pp


def check_cell_map(cell_map, g, nz):
    for ci in range(g.num_cells):
        for li in range(nz):
            if cell_map[ci][li] != ci + li * g.num_cells:
                return False
    return True


def check_face_map(face_map, g, nz):
    for fi in range(g.num_faces):
        for li in range(nz):
            if face_map[fi][li] != fi + li * g.num_faces:
                return False
    return True


@pytest.mark.parametrize(
    "z,x,y",
    [
        (np.array([0, 1]), 0, 0),
        (np.array([0, 1]), 1, 2),
        (np.array([0, 1, 3]), 1, 2),
        (np.array([0, -1]), 0, 0),
    ],
)
def test_GridExtrusion_0D(z, x, y):
    c = np.array([x, y, 0]).reshape((-1, 1))
    g = pp.PointGrid(c)
    gn, cell_map, face_map = pp.grid_extrusion._extrude_0d(g, z)

    # make target coordinates for test
    num_p = z.size
    coord_known = np.vstack((x * np.ones(num_p), y * np.ones(num_p), z))

    assert np.allclose(gn.nodes, coord_known)
    assert check_cell_map(cell_map, g, z.size - 1)
    assert face_map.size == 0


def test_GridExtrusion_0D_ignore_original_z():
    c = np.array([0, 0, 0]).reshape((-1, 1))
    g = pp.PointGrid(c)
    g.cell_centers[2, 0] = 3

    z = np.array([0, 1])
    gn, *_ = pp.grid_extrusion._extrude_0d(g, z)
    num_p = z.size
    # target coordinates
    coord_known = np.vstack((np.zeros(num_p), np.zeros(num_p), z))
    assert np.allclose(gn.nodes, coord_known)


@pytest.fixture
def x_1D_test():
    return np.array([0, 1, 3])


@pytest.fixture
def y_1D_test():
    return np.array([1, 3, 7])


@pytest.fixture
def grid_1D_test(x_1D_test, y_1D_test):
    g = pp.TensorGrid(x_1D_test)
    g.nodes = np.vstack((x_1D_test, y_1D_test, np.zeros(3)))
    return g


@pytest.mark.parametrize(
    "z,target_num_nodes",
    [
        (np.array([0, 1]), 6),
        (np.array([0, 1, 3]), 9),
        (np.array([0, 1, 3]), 9),
    ],
)
def test_GridExtrusion_1D(z, target_num_nodes, request):
    # pytest request to obtain fixture value
    # must not show up in parametrization
    # https://stackoverflow.com/questions/42014484/
    # pytest-using-fixtures-as-arguments-in-parametrize
    x = request.getfixturevalue("x_1D_test")
    y = request.getfixturevalue("y_1D_test")
    g = request.getfixturevalue("grid_1D_test")

    gn, cell_map, face_map = pp.grid_extrusion._extrude_1d(g, z)

    x_tmp = np.array([])
    y_tmp = np.empty_like(x_tmp)
    z_tmp = np.empty_like(x_tmp)
    for z_i in z:
        x_tmp = np.hstack((x_tmp, x))
        y_tmp = np.hstack((y_tmp, y))
        z_tmp = np.hstack((z_tmp, z_i * np.ones(x.size)))
    coord_known = np.vstack((x_tmp, y_tmp, z_tmp))

    assert gn.num_nodes == target_num_nodes
    assert np.allclose(gn.nodes, coord_known)
    assert check_cell_map(cell_map, g, z.size - 1)
    assert check_face_map(face_map, g, z.size - 1)


class TestGridExtrusion2D:
    """Testing grid extrusion in 2D.

    Note:
        (VL) Since all test are quite individual, a parametrization using pytest would
        lead to a hard-to-read input, hence the structure with individual tests is left
        as is.

    """

    def test_single_cell(self):
        # 2d grid is a single triangle grid with nodes ordered in the counter clockwise
        # direction

        fn_2d = sps.csc_matrix(np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]]))

        cf_2d = sps.csc_matrix(np.array([[-1], [1], [-1]]))

        nodes_2d = np.array([[0, 2, 1], [0, 0, 1], [0, 0, 0]])

        g_2d = pp.Grid(2, nodes_2d, fn_2d, cf_2d, "tmp")

        z = np.array([0, 1, 3])
        nz = z.size

        g_3d, cell_map, face_map = pp.grid_extrusion._extrude_2d(g_2d, z)

        assert g_3d.num_nodes == 9
        assert g_3d.num_cells == 2
        assert g_3d.num_faces == 9

        nodes = np.tile(nodes_2d, nz)
        nodes[2, :3] = z[0]
        nodes[2, 3:6] = z[1]
        nodes[2, 6:] = z[2]
        assert np.allclose(g_3d.nodes, nodes)

        fn_3d = np.array(
            [
                [1, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 1, 0, 0, 0],  # Done with first vertical layer
                [0, 0, 0, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 1, 0, 1, 1],
                [0, 0, 0, 1, 0, 1, 1, 0, 1],  # Second vertical layer
                [1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1],
            ]
        ).T
        assert np.allclose(g_3d.face_nodes.toarray(), fn_3d)

        cf_3d = np.array(
            [
                [-1, 0],
                [1, 0],
                [-1, 0],
                [0, -1],
                [0, 1],
                [0, -1],
                [-1, 0],
                [1, -1],
                [0, 1],
            ]
        )
        assert np.allclose(g_3d.cell_faces.toarray(), cf_3d)

        # Geometric quantities
        # Cell center
        cc = np.array([[1, 1], [1 / 3, 1 / 3], [0.5, 2]])
        assert np.allclose(g_3d.cell_centers, cc)

        # Face centers
        fc = np.array(
            [
                [1, 0, 0.5],
                [1.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [1, 0, 2],
                [1.5, 0.5, 2],
                [0.5, 0.5, 2],
                [1, 1 / 3, 0],
                [1, 1 / 3, 1],
                [1, 1 / 3, 3],
            ]
        ).T
        assert np.allclose(g_3d.face_centers, fc)

        # Face areas
        fa = np.array(
            [2, np.sqrt(2), np.sqrt(2), 4, 2 * np.sqrt(2), 2 * np.sqrt(2), 1, 1, 1]
        )
        assert np.allclose(g_3d.face_areas, fa)

        # Normal vectors
        normals = np.array(
            [
                [0, 2, 0],
                [1, 1, 0],
                [1, -1, 0],
                [0, 4, 0],
                [2, 2, 0],
                [2, -2, 0],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
            ]
        ).T
        assert np.allclose(g_3d.face_normals, normals)

        assert check_cell_map(cell_map, g_2d, nz - 1)
        assert check_face_map(face_map, g_2d, nz - 1)

    def test_single_cell_reverted_faces(self):
        # 2d grid is a single triangle grid with nodes ordered in the counter clockwise
        # direction

        fn_2d = sps.csc_matrix(np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]]))

        cf_2d = sps.csc_matrix(np.array([[-1], [1], [-1]]))

        nodes_2d = np.array([[0, 2, 1], [0, 0, 1], [0, 0, 0]])

        g_2d = pp.Grid(2, nodes_2d, fn_2d, cf_2d, "tmp")

        z = np.array([0, 1])
        nz = z.size

        g_3d, cell_map, face_map = pp.grid_extrusion._extrude_2d(g_2d, z)

        assert g_3d.num_nodes == 6
        assert g_3d.num_cells == 1
        assert g_3d.num_faces == 5

        nodes = np.tile(nodes_2d, nz)
        nodes[2, :3] = z[0]
        nodes[2, 3:] = z[1]
        assert np.allclose(g_3d.nodes, nodes)

        fn_3d = np.array(
            [
                [1, 0, 1, 1, 0, 1],
                [0, 1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1],
            ]
        ).T
        assert np.allclose(g_3d.face_nodes.toarray(), fn_3d)

        cf_3d = np.array([[-1], [1], [-1], [-1], [1]])
        assert np.allclose(g_3d.cell_faces.toarray(), cf_3d)

        # Geometric quantities
        # Cell center
        cc = np.array([[1], [1 / 3], [0.5]])
        assert np.allclose(g_3d.cell_centers, cc)

        # Face centers
        fc = np.array(
            [
                [0.5, 0.5, 0.5],
                [1.5, 0.5, 0.5],
                [1, 0, 0.5],
                [1, 1 / 3, 0],
                [1, 1 / 3, 1],
            ]
        ).T
        assert np.allclose(g_3d.face_centers, fc)

        # Face areas
        fa = np.array([np.sqrt(2), np.sqrt(2), 2, 1, 1])
        assert np.allclose(g_3d.face_areas, fa)

        # Normal vectors
        normals = np.array([[1, -1, 0], [1, 1, 0], [0, 2, 0], [0, 0, 1], [0, 0, 1]]).T
        assert np.allclose(g_3d.face_normals, normals)

        assert check_cell_map(cell_map, g_2d, nz - 1)
        assert check_face_map(face_map, g_2d, nz - 1)

    def test_single_cell_negative_z(self):
        # 2d grid is a single triangle grid with nodes ordered in the counter clockwise
        # direction

        fn_2d = sps.csc_matrix(np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]]))

        cf_2d = sps.csc_matrix(np.array([[-1], [1], [-1]]))

        nodes_2d = np.array([[0, 2, 1], [0, 0, 1], [0, 0, 0]])

        g_2d = pp.Grid(2, nodes_2d, fn_2d, cf_2d, "tmp")

        z = np.array([0, -1, -3])
        nz = z.size

        g_3d, cell_map, face_map = pp.grid_extrusion._extrude_2d(g_2d, z)

        assert g_3d.num_nodes == 9
        assert g_3d.num_cells == 2
        assert g_3d.num_faces == 9

        nodes = np.tile(nodes_2d, nz)
        nodes[2, :3] = z[0]
        nodes[2, 3:6] = z[1]
        nodes[2, 6:] = z[2]
        assert np.allclose(g_3d.nodes, nodes)

        fn_3d = np.array(
            [
                [1, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 1, 0, 0, 0],  # Done with first vertical layer
                [0, 0, 0, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 1, 0, 1, 1],
                [0, 0, 0, 1, 0, 1, 1, 0, 1],  # Second vertical layer
                [1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1],
            ]
        ).T
        assert np.allclose(g_3d.face_nodes.toarray(), fn_3d)

        cf_3d = np.array(
            [
                [-1, 0],
                [1, 0],
                [-1, 0],
                [0, -1],
                [0, 1],
                [0, -1],
                [-1, 0],
                [1, -1],
                [0, 1],
            ]
        )
        assert np.allclose(g_3d.cell_faces.toarray(), cf_3d)

        # Geometric quantities
        # Cell center
        cc = np.array([[1, 1], [1 / 3, 1 / 3], [-0.5, -2]])
        assert np.allclose(g_3d.cell_centers, cc)

        # Face centers
        fc = np.array(
            [
                [1, 0, -0.5],
                [1.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
                [1, 0, -2],
                [1.5, 0.5, -2],
                [0.5, 0.5, -2],
                [1, 1 / 3, 0],
                [1, 1 / 3, -1],
                [1, 1 / 3, -3],
            ]
        ).T
        assert np.allclose(g_3d.face_centers, fc)

        # Face areas
        fa = np.array(
            [2, np.sqrt(2), np.sqrt(2), 4, 2 * np.sqrt(2), 2 * np.sqrt(2), 1, 1, 1]
        )
        assert np.allclose(g_3d.face_areas, fa)

        # Normal vectors
        normals = np.array(
            [
                [0, 2, 0],
                [1, 1, 0],
                [1, -1, 0],
                [0, 4, 0],
                [2, 2, 0],
                [2, -2, 0],
                [0, 0, -1],
                [0, 0, -1],
                [0, 0, -1],
            ]
        ).T
        assert np.allclose(g_3d.face_normals, normals)

        assert check_cell_map(cell_map, g_2d, nz - 1)
        assert check_face_map(face_map, g_2d, nz - 1)

    def test_two_cells(self):
        # 2d grid is a single triangle grid with nodes ordered in the counter clockwise
        # direction

        fn_2d = sps.csc_matrix(
            np.array(
                [[1, 0, 0, 1, 0], [1, 1, 0, 0, 1], [0, 1, 1, 0, 0], [0, 0, 1, 1, 1]]
            )
        )

        cf_2d = sps.csc_matrix(np.array([[1, 0], [0, 1], [0, 1], [1, 0], [-1, 1]]))

        nodes_2d = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])

        g_2d = pp.Grid(2, nodes_2d, fn_2d, cf_2d, "tmp")

        z = np.array([0, 1])
        nz = z.size

        g_3d, cell_map, face_map = pp.grid_extrusion._extrude_2d(g_2d, z)

        assert g_3d.num_nodes == 8
        assert g_3d.num_cells == 2
        assert g_3d.num_faces == 9

        nodes = np.tile(nodes_2d, nz)
        nodes[2, :4] = z[0]
        nodes[2, 4:] = z[1]
        assert np.allclose(g_3d.nodes, nodes)

        fn_3d = np.array(
            [
                [1, 1, 0, 0, 1, 1, 0, 0],
                [0, 1, 1, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0, 0, 1, 1],  # Done with first vertical layer
                [1, 0, 0, 1, 1, 0, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [1, 1, 0, 1, 0, 0, 0, 0],  # Second vertical layer
                [0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
            ]
        ).T
        assert np.allclose(g_3d.face_nodes.toarray(), fn_3d)

        cf_3d = np.array(
            [[1, 0], [0, 1], [0, 1], [1, 0], [-1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]]
        )
        assert np.allclose(g_3d.cell_faces.toarray(), cf_3d)

        assert check_cell_map(cell_map, g_2d, nz - 1)
        assert check_face_map(face_map, g_2d, nz - 1)

    def test_two_cells_one_quad(self):
        # 2d grid is a single triangle grid with nodes ordered in the counter clockwise
        # direction

        fn_2d = sps.csc_matrix(
            np.array(
                [
                    [1, 0, 0, 0, 1, 0],
                    [1, 1, 0, 0, 0, 1],
                    [0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1],
                ]
            )
        )

        cf_2d = sps.csc_matrix(
            np.array([[1, 0], [0, -1], [0, 1], [0, 1], [-1, 0], [1, -1]])
        )

        nodes_2d = np.array([[0, 1, 2, 1, 0], [0, 0, 1, 2, 1], [0, 0, 0, 0, 0]])

        g_2d = pp.Grid(2, nodes_2d, fn_2d, cf_2d, "tmp")

        z = np.array([0, 1])
        nz = z.size

        g_3d, cell_map, face_map = pp.grid_extrusion._extrude_2d(g_2d, z)

        assert g_3d.num_nodes == 10
        assert g_3d.num_cells == 2
        assert g_3d.num_faces == 10

        nodes = np.tile(nodes_2d, nz)
        nodes[2, :5] = z[0]
        nodes[2, 5:] = z[1]
        assert np.allclose(g_3d.nodes, nodes)

        fn_ind = np.array(
            [
                0,
                1,
                6,
                5,
                1,
                2,
                7,
                6,
                2,
                3,
                8,
                7,
                3,
                4,
                9,
                8,
                4,
                0,
                5,
                9,
                1,
                4,
                9,
                6,
                0,
                1,
                4,
                1,
                2,
                3,
                4,
                5,
                6,
                9,
                6,
                7,
                8,
                9,
            ]
        )
        fn_ind_ptr = np.array([0, 4, 8, 12, 16, 20, 24, 27, 31, 34, fn_ind.size])

        fn_3d = sps.csc_matrix(
            (np.ones(fn_ind.size), fn_ind, fn_ind_ptr), shape=(10, 10)
        )
        assert np.allclose(g_3d.face_nodes.toarray(), fn_3d.toarray())

        cf_3d = np.array(
            [
                [1, 0],
                [0, -1],
                [0, 1],
                [0, 1],
                [-1, 0],
                [1, -1],
                [-1, 0],
                [0, -1],
                [1, 0],
                [0, 1],
            ]
        )
        assert np.allclose(g_3d.cell_faces.toarray(), cf_3d)

        assert check_cell_map(cell_map, g_2d, nz - 1)
        assert check_face_map(face_map, g_2d, nz - 1)


@pytest.mark.parametrize(
    "g,target_dim",
    [
        (pp.PointGrid(np.zeros((3, 1))), 1),
        (pp.CartGrid(1), 2),
        (pp.CartGrid([1, 1]), 3),
    ],
)
def test_dimension_of_extruded_grid(g, target_dim):
    z = np.arange(2)
    gn, *_ = pp.grid_extrusion.extrude_grid(g, z)
    assert gn.dim == target_dim


def test_3D_grid_cannot_be_extruded():
    g = pp.CartGrid([1, 1, 1])
    z = np.arange(2)
    with pytest.raises(ValueError):
        pp.grid_extrusion.extrude_grid(g, z)


def test_extrusion_with_single_fracture():
    f = np.array([[1, 3], [1, 1]])
    mdg = pp.meshing.cart_grid([f], [4, 2])

    sd_2 = mdg.subdomains(dim=2)[0]
    sd_1 = mdg.subdomains(dim=1)[0]

    z = np.array([0, 1, 2])
    mdg_new, _ = pp.grid_extrusion.extrude_grid_bucket(mdg, z)

    for dim in range(mdg.dim_max()):
        assert len(mdg.subdomains(dim=dim)) == len(mdg_new.subdomains(dim=dim + 1))

    # Get hold of interface grids. We know there is a single interface in each
    # mixed-dimensional grid, so make a loop and abort it.
    for intf in mdg.interfaces():
        break
    for intf_new in mdg_new.interfaces():
        break

    # Check that the old and new grid have
    bound_faces_old = intf.primary_to_mortar_int().tocoo().col
    # We know from the construction of the grid extruded to 3d that the faces
    # on the fracture boundary will be those in the 2d grid stacked on top of each
    # other
    known_bound_faces_new = np.hstack(
        (bound_faces_old, bound_faces_old + sd_2.num_faces)
    )
    bound_faces_new = intf_new.primary_to_mortar_int().tocoo().col
    assert np.allclose(np.sort(known_bound_faces_new), np.sort(bound_faces_new))

    low_cells_old = intf.secondary_to_mortar_int().tocoo().col
    known_cells_new = np.hstack((low_cells_old, low_cells_old + sd_1.num_cells))

    cells_new = intf_new.secondary_to_mortar_int().tocoo().col

    assert np.allclose(np.sort(known_cells_new), np.sort(cells_new))

    # All mortar cells should be associated with a face in primary
    mortar_cells_in_range_from_primary = np.unique(
        intf_new.primary_to_mortar_int().tocoo().row
    )
    assert mortar_cells_in_range_from_primary.size == intf_new.num_cells
    assert np.allclose(
        mortar_cells_in_range_from_primary, np.arange(intf_new.num_cells)
    )

    # All mortar cells should be in the range from secondary
    mortar_cells_in_range_from_secondary = np.unique(
        intf_new.secondary_to_mortar_int().tocoo().row
    )
    assert mortar_cells_in_range_from_secondary.size == intf_new.num_cells
    assert np.allclose(
        np.sort(mortar_cells_in_range_from_secondary),
        np.arange(intf_new.num_cells),
    )

    ## Check that the + and - sides of the extruded mortar grid are correct
    extruded_faces, _, sgn = sparse_array_to_row_col_data(
        intf_new.mortar_to_primary_int() * intf_new.sign_of_mortar_sides()
    )
    # Get the extruded cells next to the mortar grid. Same ordering as extruded_faces
    g3 = mdg_new.subdomains(dim=3)[0]
    extruded_cells = g3.cell_faces[extruded_faces].tocsr().indices

    # coordinates
    cc = g3.cell_centers[:, extruded_cells]
    fc = g3.face_centers[:, extruded_faces]
    # Sign the distances to get a number. Importantly, this will either be positive
    # or negative, depending on which side cc is of the fracture (we know this because
    # of the fracture geometry).
    dist = np.sum(cc - fc)

    # All faces on the same side (identified by the sign of dist) should have the
    # same sign
    pos_dist = dist > 0
    assert np.logical_or(np.all(sgn[pos_dist] > 0), np.all(sgn[pos_dist] < 0))
    neg_dist = dist < 0
    assert np.logical_or(np.all(sgn[neg_dist] > 0), np.all(sgn[neg_dist] < 0))

    g_1_new = mdg_new.subdomains(dim=2)[0]

    # All mortar cells should be associated with two cells in secondary
    secondary_cells_in_range_from_mortar = (
        intf_new.mortar_to_secondary_int().tocoo().row
    )
    assert np.all(np.bincount(secondary_cells_in_range_from_mortar) == 2)
    assert np.unique(secondary_cells_in_range_from_mortar).size == g_1_new.num_cells
    assert np.allclose(
        np.sort(np.unique(secondary_cells_in_range_from_mortar)),
        np.arange(g_1_new.num_cells),
    )


def test_extrusion_with_T_intersection():
    """2d mdg with T-intersection. Mainly ensure that the code runs (it did not
    before #495)
    """
    f = [np.array([[1, 3], [2, 2]]), np.array([[2, 2], [1, 2]])]
    mdg = pp.meshing.cart_grid(f, [4, 3])

    z = np.array([0, 1])
    mdg_new, _ = pp.grid_extrusion.extrude_grid_bucket(mdg, z)

    # Do a simple test on grid geometry; if this fails, there is a more fundamental
    # problem that should be picked up by simpler tests.
    sd = mdg_new.subdomains(dim=2)[1]
    assert np.allclose(sd.cell_centers, np.array([[2], [1.5], [0.5]]))
