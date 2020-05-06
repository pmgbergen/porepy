import numpy as np
import scipy.sparse as sps
import unittest
import warnings

import porepy as pp


class TestCartLeafGrid(unittest.TestCase):
    def test_generation_2_levels(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 2)

        g_t = [pp.CartGrid([2, 2], [1, 1]), pp.CartGrid([4, 4], [1, 1])]

        for i, g in enumerate(lg.level_grids):
            self.assertTrue(np.allclose(g.nodes, g_t[i].nodes))
            self.assertTrue(np.allclose(g.face_nodes.A, g_t[i].face_nodes.A))
            self.assertTrue(np.allclose(g.cell_faces.A, g_t[i].cell_faces.A))

    def test_generation_3_levels(self):
        lg = pp.CartLeafGrid([1, 2], [1, 1], 3)

        g_t = [
            pp.CartGrid([1, 2], [1, 1]),
            pp.CartGrid([2, 4], [1, 1]),
            pp.CartGrid([4, 8], [1, 1]),
        ]

        for i, g in enumerate(lg.level_grids):
            self.assertTrue(np.allclose(g.nodes, g_t[i].nodes))
            self.assertTrue(np.allclose(g.face_nodes.A, g_t[i].face_nodes.A))
            self.assertTrue(np.allclose(g.cell_faces.A, g_t[i].cell_faces.A))

    def test_refinement_full_grid(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 2)
        lg.refine_cells(np.ones(4, dtype=bool))
        g_t = pp.CartGrid([4, 4], [1, 1])

        self.assertTrue(np.allclose(lg.nodes, g_t.nodes))
        self.assertTrue(np.allclose(lg.face_nodes.A, g_t.face_nodes.A))
        self.assertTrue(np.allclose(lg.cell_faces.A, g_t.cell_faces.A))

    def test_refinement_multiple_levels(self):
        lg = pp.CartLeafGrid([1, 2], [1, 1], 3)
        lg.refine_cells(0)
        lg.refine_cells([0, 1])

        nodes = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.25, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [0.0, 0.75, 0.0],
                [0.5, 0.75, 0.0],
                [1.0, 0.75, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.125, 0.0],
                [0.25, 0.125, 0.0],
                [0.5, 0.125, 0.0],
                [0.0, 0.25, 0.0],
                [0.25, 0.25, 0.0],
                [0.5, 0.25, 0.0],
            ]
        ).T

        face_nodes = sps.csc_matrix(
            np.array(
                [
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                ]
            ).T
        )

        cell_faces = sps.csc_matrix(
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
                    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
                    [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
        g_t = pp.Grid(2, nodes, face_nodes, cell_faces, ["Ref"])

        g_t.compute_geometry()
        self._compare_grids(lg, g_t)

    def test_refinement_several_times(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 2)

        lg.refine_cells(0)
        lg.refine_cells(0)
        lg.refine_cells(0)
        lg.refine_cells(0)

        g_t = pp.CartGrid([4, 4], [1, 1])
        g_t.compute_geometry()
        self._compare_grids(lg, g_t)

    def test_recursive_refinement_blocks(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 3)

        lg.refine_cells([0, 1])
        lg.refine_cells([2, 3, 4, 5])

        g0 = pp.CartGrid([2, 1], [1, 0.5])
        g0.nodes[1] += 0.5
        g0.compute_geometry()

        g1 = pp.CartGrid([4, 1], [1, 0.25])
        g1.nodes[1] += 0.25
        g1.compute_geometry()

        g2 = pp.CartGrid([8, 2], [1, 0.25])
        g2.compute_geometry()

        self.assertTrue(np.allclose(lg.cell_centers[:, :2], g0.cell_centers))
        self.assertTrue(np.allclose(lg.cell_centers[:, 2:6], g1.cell_centers))
        self.assertTrue(np.allclose(lg.cell_centers[:, 6:], g2.cell_centers))

    def test_coarse_cell_ref_after_fine_cell_ref(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 3)

        lg.refine_cells(0)  # refine cell 0
        lg.refine_cells(3)  # cell 3 is first cell of level 1
        lg.refine_cells(0)  # Cell 0 is still on level 0

        # This should be equivalent to refinging cell 0, 1 and then cell 2
        lg_ref = pp.CartLeafGrid([2, 2], [1, 1], 3)

        lg_ref.refine_cells([0, 1])
        lg_ref.refine_cells(2)

        self._compare_grids(lg, lg_ref)

    def test_recursive_refinement(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 3)

        lg.refine_cells(0)
        lg.refine_cells(3)

        cell_centers = np.array(
            [
                [0.75, 0.25, 0.0],
                [0.25, 0.75, 0.0],
                [0.75, 0.75, 0.0],
                [0.375, 0.125, 0.0],
                [0.125, 0.375, 0.0],
                [0.375, 0.375, 0.0],
                [0.0625, 0.0625, 0.0],
                [0.1875, 0.0625, 0.0],
                [0.0625, 0.1875, 0.0],
                [0.1875, 0.1875, 0.0],
            ]
        ).T

        face_centers = np.array(
            [
                [1.0, 0.25, 0.0],
                [0.0, 0.75, 0.0],
                [0.5, 0.75, 0.0],
                [1.0, 0.75, 0.0],
                [0.75, 0.0, 0.0],
                [0.75, 0.5, 0.0],
                [0.25, 1.0, 0.0],
                [0.75, 1.0, 0.0],
                [0.5, 0.125, 0.0],
                [0.0, 0.375, 0.0],
                [0.25, 0.375, 0.0],
                [0.5, 0.375, 0.0],
                [0.375, 0.0, 0.0],
                [0.375, 0.25, 0.0],
                [0.125, 0.5, 0.0],
                [0.375, 0.5, 0.0],
                [0.0, 0.0625, 0.0],
                [0.125, 0.0625, 0.0],
                [0.25, 0.0625, 0.0],
                [0.0, 0.1875, 0.0],
                [0.125, 0.1875, 0.0],
                [0.25, 0.1875, 0.0],
                [0.0625, 0.0, 0.0],
                [0.1875, 0.0, 0.0],
                [0.0625, 0.125, 0.0],
                [0.1875, 0.125, 0.0],
                [0.0625, 0.25, 0.0],
                [0.1875, 0.25, 0.0],
            ]
        ).T

        nodes = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.25, 0.0],
                [0.0, 0.5, 0.0],
                [0.25, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 0.0],
                [0.125, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.0, 0.125, 0.0],
                [0.125, 0.125, 0.0],
                [0.25, 0.125, 0.0],
                [0.0, 0.25, 0.0],
                [0.125, 0.25, 0.0],
                [0.25, 0.25, 0.0],
            ]
        ).T

        self.assertTrue(np.allclose(lg.nodes, nodes))
        self.assertTrue(np.allclose(lg.face_centers, face_centers))
        self.assertTrue(np.allclose(lg.cell_centers, cell_centers))

    def test_refinement_singel_cell(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 2)
        lg.refine_cells(0)

        nodes = np.array(
            [
                [1, 1, 0, 0.5, 1, 0, 0.25, 0.5, 0, 0.25, 0.5, 0.0, 0.25, 0.5],
                [0, 0.5, 1, 1, 1, 0, 0, 0, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        face_nodes = np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            ]
        ).T
        cell_faces = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
                [0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.assertTrue(np.allclose(lg.nodes, nodes))
        self.assertTrue(np.allclose(lg.face_nodes.A, face_nodes))
        self.assertTrue(np.allclose(lg.cell_faces.A, cell_faces))

    def test_max_one_level_ref(self):
        """
        Refine CartGrid([2, 2], [1, 1]) to:
         _______ _______
        |       |       |
        |       |       |
        |       |       |
        |_______|___ ___|
        |   |   |   |   |
        |___|___|___|___|
        |   |_|_|   |   |
        |___|_|_|___|___|
        """
        lg = pp.CartLeafGrid([2, 2], [1, 1], 3)

        lg.refine_cells(0)
        lg.refine_cells(4) # Should refine cell 0 as well

        lg_ref = pp.CartLeafGrid([2, 2], [1, 1], 3)
        lg_ref.refine_cells([0, 1])
        lg_ref.refine_cells(3)

        self._compare_grids(lg, lg_ref)

    def test_max_one_level_ref_rec(self):
        """
        Refine CartGrid([2, 2], [1, 1]) to:

         _______ _______
        |       |       |
        |       |       |
        |       |       |
        |___ ___|___ ___|
        |   |   |       |
        |___|___|       |
        |   |   |       |
        |___|___|___ ___|
        |   |_|_|   |   |
        |___|_|_|___|___|
        |   |   |   |   |
        |___|___|___|___|
        """
        lg = pp.CartLeafGrid([1, 2], [1, 1], 4)

        lg.refine_cells(0)
        lg.refine_cells(1)
        lg.refine_cells(7) # should refine cell 0, 1, and 2 as well

        lg_ref = pp.CartLeafGrid([1, 2], [1, 1], 4)
        lg_ref.refine_cells([0, 1])
        lg_ref.refine_cells([0, 1, 2])
        lg_ref.refine_cells(10)
        pp.plot_grid(lg_ref)
        pp.plot_grid(lg)
        self._compare_grids(lg, lg_ref)
        

    def _compare_grids(self, g0, g1):
        self.assertTrue(np.allclose(g0.nodes, g1.nodes))
        self.assertTrue(np.allclose(g0.face_nodes.A, g1.face_nodes.A))
        self.assertTrue(np.allclose(g0.cell_faces.A, g1.cell_faces.A))

        self.assertTrue(np.allclose(g0.face_areas, g1.face_areas))
        self.assertTrue(np.allclose(g0.face_centers, g1.face_centers))
        self.assertTrue(np.allclose(g0.face_normals, g1.face_normals))
        self.assertTrue(np.allclose(g0.cell_volumes, g1.cell_volumes))
        self.assertTrue(np.allclose(g0.cell_centers, g1.cell_centers))
