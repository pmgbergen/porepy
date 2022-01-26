"""
Unit tests for grid extrusion functionality.
"""

import unittest

import numpy as np
import scipy.sparse as sps

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


class TestGridExtrusion0d(unittest.TestCase):
    def setUp(self, x=0, y=0):
        c = np.array([x, y, 0]).reshape((-1, 1))
        self.g = pp.PointGrid(c)

    def known_array(self, z, x=0, y=0):
        num_p = z.size
        return np.vstack((x * np.ones(num_p), y * np.ones(num_p), z))

    def make_grid_check(self, z, x=0, y=0):
        gn, cell_map, face_map = pp.grid_extrusion._extrude_0d(self.g, z)
        coord_known = self.known_array(z, x, y)
        self.assertTrue(np.allclose(gn.nodes, coord_known))

        self.assertTrue(check_cell_map(cell_map, self.g, z.size - 1))
        self.assertTrue(face_map.size == 0)

    def test_one_layer(self):
        z = np.array([0, 1])
        self.make_grid_check(z)

    def test_not_origin(self):
        # Point grid is away from origin
        self.setUp(x=1, y=2)
        z = np.array([0, 1])
        self.make_grid_check(z, x=1, y=2)

    def test_two_layers(self):
        z = np.array([0, 1, 3])
        self.make_grid_check(z)

    def test_ignore_original_z(self):
        # Assign non-zero z-coordinate of the point grid. This should be ignored
        self.g.cell_centers[2, 0] = 3
        z = np.array([0, 1])
        gn, cell_map, face_map = pp.grid_extrusion._extrude_0d(self.g, z)
        coord_known = self.known_array(z)
        self.assertTrue(np.allclose(gn.nodes, coord_known))

    def test_negative_z(self):
        z = np.array([0, -1])
        self.make_grid_check(z)


class TestGridExtrusion1d(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0, 1, 3])
        self.y = np.array([1, 3, 7])
        self.g = pp.TensorGrid(self.x)
        self.g.nodes = np.vstack((self.x, self.y, np.zeros(3)))

    def known_array(self, z_coord):
        x_tmp = np.array([])
        y_tmp = np.empty_like(x_tmp)
        z_tmp = np.empty_like(x_tmp)
        for z in z_coord:
            x_tmp = np.hstack((x_tmp, self.x))
            y_tmp = np.hstack((y_tmp, self.y))
            z_tmp = np.hstack((z_tmp, z * np.ones(self.x.size)))
        return np.vstack((x_tmp, y_tmp, z_tmp))

    def test_one_layer(self):
        z = np.array([0, 1])
        gn, cell_map, face_map = pp.grid_extrusion._extrude_1d(self.g, z)
        coord_known = self.known_array(z)
        self.assertTrue(np.allclose(gn.nodes, coord_known))
        self.assertTrue(check_cell_map(cell_map, self.g, z.size - 1))
        self.assertTrue(check_face_map(face_map, self.g, z.size - 1))

    def test_two_layers(self):
        z = np.array([0, 1, 3])
        gn, cell_map, face_map = pp.grid_extrusion._extrude_1d(self.g, z)

        self.assertTrue(gn.num_nodes == 9)

        coord_known = self.known_array(z)
        self.assertTrue(np.allclose(gn.nodes, coord_known))
        self.assertTrue(check_cell_map(cell_map, self.g, z.size - 1))
        self.assertTrue(check_face_map(face_map, self.g, z.size - 1))

    def test_negative_z(self):
        z = np.array([0, -1, -3])
        gn, cell_map, face_map = pp.grid_extrusion._extrude_1d(self.g, z)

        self.assertTrue(gn.num_nodes == 9)

        coord_known = self.known_array(z)
        self.assertTrue(np.allclose(gn.nodes, coord_known))
        self.assertTrue(check_cell_map(cell_map, self.g, z.size - 1))
        self.assertTrue(check_face_map(face_map, self.g, z.size - 1))


class TestGridExtrusion2d(unittest.TestCase):
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

        self.assertTrue(g_3d.num_nodes == 9)
        self.assertTrue(g_3d.num_cells == 2)
        self.assertTrue(g_3d.num_faces == 9)

        nodes = np.tile(nodes_2d, nz)
        nodes[2, :3] = z[0]
        nodes[2, 3:6] = z[1]
        nodes[2, 6:] = z[2]
        self.assertTrue(np.allclose(g_3d.nodes, nodes))

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
        self.assertTrue(np.allclose(g_3d.face_nodes.toarray(), fn_3d))

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
        self.assertTrue(np.allclose(g_3d.cell_faces.toarray(), cf_3d))

        # Geometric quantities
        # Cell center
        cc = np.array([[1, 1], [1 / 3, 1 / 3], [0.5, 2]])
        self.assertTrue(np.allclose(g_3d.cell_centers, cc))

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
        self.assertTrue(np.allclose(g_3d.face_centers, fc))

        # Face areas
        fa = np.array(
            [2, np.sqrt(2), np.sqrt(2), 4, 2 * np.sqrt(2), 2 * np.sqrt(2), 1, 1, 1]
        )
        self.assertTrue(np.allclose(g_3d.face_areas, fa))

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
        self.assertTrue(np.allclose(g_3d.face_normals, normals))

        self.assertTrue(check_cell_map(cell_map, g_2d, nz - 1))
        self.assertTrue(check_face_map(face_map, g_2d, nz - 1))

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

        self.assertTrue(g_3d.num_nodes == 6)
        self.assertTrue(g_3d.num_cells == 1)
        self.assertTrue(g_3d.num_faces == 5)

        nodes = np.tile(nodes_2d, nz)
        nodes[2, :3] = z[0]
        nodes[2, 3:] = z[1]
        self.assertTrue(np.allclose(g_3d.nodes, nodes))

        fn_3d = np.array(
            [
                [1, 0, 1, 1, 0, 1],
                [0, 1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1],
            ]
        ).T
        self.assertTrue(np.allclose(g_3d.face_nodes.toarray(), fn_3d))

        cf_3d = np.array([[-1], [1], [-1], [-1], [1]])
        self.assertTrue(np.allclose(g_3d.cell_faces.toarray(), cf_3d))

        # Geometric quantities
        # Cell center
        cc = np.array([[1], [1 / 3], [0.5]])
        self.assertTrue(np.allclose(g_3d.cell_centers, cc))

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
        self.assertTrue(np.allclose(g_3d.face_centers, fc))

        # Face areas
        fa = np.array([np.sqrt(2), np.sqrt(2), 2, 1, 1])
        self.assertTrue(np.allclose(g_3d.face_areas, fa))

        # Normal vectors
        normals = np.array([[1, -1, 0], [1, 1, 0], [0, 2, 0], [0, 0, 1], [0, 0, 1]]).T
        self.assertTrue(np.allclose(g_3d.face_normals, normals))

        self.assertTrue(check_cell_map(cell_map, g_2d, nz - 1))
        self.assertTrue(check_face_map(face_map, g_2d, nz - 1))

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

        self.assertTrue(g_3d.num_nodes == 9)
        self.assertTrue(g_3d.num_cells == 2)
        self.assertTrue(g_3d.num_faces == 9)

        nodes = np.tile(nodes_2d, nz)
        nodes[2, :3] = z[0]
        nodes[2, 3:6] = z[1]
        nodes[2, 6:] = z[2]
        self.assertTrue(np.allclose(g_3d.nodes, nodes))

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
        self.assertTrue(np.allclose(g_3d.face_nodes.toarray(), fn_3d))

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
        self.assertTrue(np.allclose(g_3d.cell_faces.toarray(), cf_3d))

        # Geometric quantities
        # Cell center
        cc = np.array([[1, 1], [1 / 3, 1 / 3], [-0.5, -2]])
        self.assertTrue(np.allclose(g_3d.cell_centers, cc))

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
        self.assertTrue(np.allclose(g_3d.face_centers, fc))

        # Face areas
        fa = np.array(
            [2, np.sqrt(2), np.sqrt(2), 4, 2 * np.sqrt(2), 2 * np.sqrt(2), 1, 1, 1]
        )
        self.assertTrue(np.allclose(g_3d.face_areas, fa))

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
        self.assertTrue(np.allclose(g_3d.face_normals, normals))

        self.assertTrue(check_cell_map(cell_map, g_2d, nz - 1))
        self.assertTrue(check_face_map(face_map, g_2d, nz - 1))

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

        self.assertTrue(g_3d.num_nodes == 8)
        self.assertTrue(g_3d.num_cells == 2)
        self.assertTrue(g_3d.num_faces == 9)

        nodes = np.tile(nodes_2d, nz)
        nodes[2, :4] = z[0]
        nodes[2, 4:] = z[1]
        self.assertTrue(np.allclose(g_3d.nodes, nodes))

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
        self.assertTrue(np.allclose(g_3d.face_nodes.toarray(), fn_3d))

        cf_3d = np.array(
            [[1, 0], [0, 1], [0, 1], [1, 0], [-1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]]
        )
        self.assertTrue(np.allclose(g_3d.cell_faces.toarray(), cf_3d))

        self.assertTrue(check_cell_map(cell_map, g_2d, nz - 1))
        self.assertTrue(check_face_map(face_map, g_2d, nz - 1))

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

        self.assertTrue(g_3d.num_nodes == 10)
        self.assertTrue(g_3d.num_cells == 2)
        self.assertTrue(g_3d.num_faces == 10)

        nodes = np.tile(nodes_2d, nz)
        nodes[2, :5] = z[0]
        nodes[2, 5:] = z[1]
        self.assertTrue(np.allclose(g_3d.nodes, nodes))

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
        self.assertTrue(np.allclose(g_3d.face_nodes.toarray(), fn_3d.toarray()))

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
        self.assertTrue(np.allclose(g_3d.cell_faces.toarray(), cf_3d))

        self.assertTrue(check_cell_map(cell_map, g_2d, nz - 1))
        self.assertTrue(check_face_map(face_map, g_2d, nz - 1))


class TestGridExtrusionNd(unittest.TestCase):
    # Test of the main grid extrusion function.

    def test_0d(self):
        g = pp.PointGrid(np.zeros((3, 1)))
        z = np.arange(2)
        gn, *rest = pp.grid_extrusion.extrude_grid(g, z)
        self.assertTrue(gn.dim == 1)

    def test_1d(self):
        g = pp.CartGrid(1)
        z = np.arange(2)
        gn, *rest = pp.grid_extrusion.extrude_grid(g, z)
        self.assertTrue(gn.dim == 2)

    def test_2d(self):
        g = pp.CartGrid([1, 1])
        z = np.arange(2)
        gn, *rest = pp.grid_extrusion.extrude_grid(g, z)
        self.assertTrue(gn.dim == 3)

    def test_3d(self):
        g = pp.CartGrid([1, 1, 1])
        z = np.arange(2)
        self.assertRaises(ValueError, pp.grid_extrusion.extrude_grid, g, z)


class TestGridBucketExtrusion(unittest.TestCase):
    def test_single_fracture(self):

        f = np.array([[1, 3], [1, 1]])
        gb = pp.meshing.cart_grid([f], [4, 2])

        g_2 = gb.grids_of_dimension(2)[0]
        g_1 = gb.grids_of_dimension(1)[0]

        z = np.array([0, 1, 2])
        gb_new, g_map = pp.grid_extrusion.extrude_grid_bucket(gb, z)

        for dim in range(gb.dim_max()):
            self.assertTrue(
                len(gb.grids_of_dimension(dim))
                == len(gb_new.grids_of_dimension(dim + 1))
            )

        for _, d in gb.edges():
            break
        mg = d["mortar_grid"]
        for _, d in gb_new.edges():
            break
        mg_new = d["mortar_grid"]

        # Check that the old and new grid have
        bound_faces_old = mg.primary_to_mortar_int().tocoo().col
        # We know from the construction of the grid extruded to 3d that the faces
        # on the fracture boundary will be those in the 2d grid stacked on top of each
        # other
        known_bound_faces_new = np.hstack(
            (bound_faces_old, bound_faces_old + g_2.num_faces)
        )
        bound_faces_new = mg_new.primary_to_mortar_int().tocoo().col
        self.assertTrue(
            np.allclose(np.sort(known_bound_faces_new), np.sort(bound_faces_new))
        )

        low_cells_old = mg.secondary_to_mortar_int().tocoo().col
        known_cells_new = np.hstack((low_cells_old, low_cells_old + g_1.num_cells))

        cells_new = mg_new.secondary_to_mortar_int().tocoo().col

        self.assertTrue(np.allclose(np.sort(known_cells_new), np.sort(cells_new)))

        # All mortar cells should be associated with a face in primary
        mortar_cells_in_range_from_primary = np.unique(
            mg_new.primary_to_mortar_int().tocoo().row
        )
        self.assertTrue(mortar_cells_in_range_from_primary.size == mg_new.num_cells)
        self.assertTrue(
            np.allclose(mortar_cells_in_range_from_primary, np.arange(mg_new.num_cells))
        )

        # All mortar cells should be in the range from secondary
        mortar_cells_in_range_from_secondary = np.unique(
            mg_new.secondary_to_mortar_int().tocoo().row
        )
        self.assertTrue(mortar_cells_in_range_from_secondary.size == mg_new.num_cells)
        self.assertTrue(
            np.allclose(
                np.sort(mortar_cells_in_range_from_secondary),
                np.arange(mg_new.num_cells),
            )
        )
        ## Check that the + and - sides of the extruded mortar grid are correct
        extruded_faces, _, sgn = sps.find(
            mg_new.mortar_to_primary_int() * mg_new.sign_of_mortar_sides()
        )
        # Get the extruded cells next to the mortar grid. Same ordering as extruded_faces
        g3 = gb_new.grids_of_dimension(3)[0]
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
        self.assertTrue(
            np.logical_or(np.all(sgn[pos_dist] > 0), np.all(sgn[pos_dist] < 0))
        )
        neg_dist = dist < 0
        self.assertTrue(
            np.logical_or(np.all(sgn[neg_dist] > 0), np.all(sgn[neg_dist] < 0))
        )

        g_1_new = gb_new.grids_of_dimension(2)[0]

        # All mortar cells should be associated with two cells in secondary
        secondary_cells_in_range_from_mortar = (
            mg_new.mortar_to_secondary_int().tocoo().row
        )
        self.assertTrue(np.all(np.bincount(secondary_cells_in_range_from_mortar) == 2))
        self.assertTrue(
            np.unique(secondary_cells_in_range_from_mortar).size == g_1_new.num_cells
        )
        self.assertTrue(
            np.allclose(
                np.sort(np.unique(secondary_cells_in_range_from_mortar)),
                np.arange(g_1_new.num_cells),
            )
        )

    def test_T_intersection(self):
        """ 2d gb with T-intersection. Mainly ensure that the code runs (it did not
        before #495)
        """
        f = [np.array([[1, 3], [2, 2]]), np.array([[2, 2], [1, 2]])]
        gb = pp.meshing.cart_grid(f, [4, 3])

        z = np.array([0, 1])
        gb_new, g_map = pp.grid_extrusion.extrude_grid_bucket(gb, z)

        # Do a simple test on grid geometry; if this fails, there is a more fundamental
        # problem that should be picked up by simpler tests.
        g = gb_new.grids_of_dimension(2)[1]
        self.assertTrue(np.allclose(g.cell_centers, np.array([[2], [1.5], [0.5]])))

if __name__ == "__main__":
    unittest.main()
