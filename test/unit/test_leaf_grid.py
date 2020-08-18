import numpy as np
import scipy.sparse as sps
import unittest
import warnings

import porepy as pp


class TestCartLeafGrid_2d(unittest.TestCase):
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
        old_to_new, _ = lg.refine_cells(np.ones(4, dtype=bool))
        g_t = pp.CartGrid([4, 4], [1, 1])

        proj_ref = np.array(
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ]
        )
        self.assertTrue(np.allclose(lg.nodes, g_t.nodes))
        self.assertTrue(np.allclose(lg.face_nodes.A, g_t.face_nodes.A))
        self.assertTrue(np.allclose(lg.cell_faces.A, g_t.cell_faces.A))
        self.assertTrue(np.allclose(old_to_new.A, proj_ref))

    def test_refinement_multiple_levels(self):
        lg = pp.CartLeafGrid([1, 2], [1, 1], 3)
        old2new, _ = lg.refine_cells(0)
        old2new = lg.refine_cells([0, 1])[0] * old2new
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
        proj_known = np.array(
            [[1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]]
        ).T
        g_t.compute_geometry()
        self._compare_grids(lg, g_t)
        self.assertTrue(np.allclose(old2new.A, proj_known))

    def test_refinement_several_times(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 2)
        old2new, _ = lg.refine_cells(0)
        old2new = lg.refine_cells(0)[0] * old2new
        old2new = lg.refine_cells(0)[0] * old2new
        old2new = lg.refine_cells(0)[0] * old2new

        g_t = pp.CartGrid([4, 4], [1, 1])
        g_t.compute_geometry()
        proj_known = np.array(
            [
                [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            ]
        ).T

        self._compare_grids(lg, g_t)
        self.assertTrue(np.allclose(old2new.A, proj_known))

    def test_recursive_refinement_blocks(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 3)

        old2new, _ = lg.refine_cells([0, 1])
        old2new = lg.refine_cells([2, 3, 4, 5])[0] * old2new

        g0 = pp.CartGrid([2, 1], [1, 0.5])
        g0.nodes[1] += 0.5
        g0.compute_geometry()

        g1 = pp.CartGrid([4, 1], [1, 0.25])
        g1.nodes[1] += 0.25
        g1.compute_geometry()

        g2 = pp.CartGrid([8, 2], [1, 0.25])
        g2.compute_geometry()

        proj_known = np.array(
            [
                [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ).T

        self.assertTrue(np.allclose(lg.cell_centers[:, :2], g0.cell_centers))
        self.assertTrue(np.allclose(lg.cell_centers[:, 2:6], g1.cell_centers))
        self.assertTrue(np.allclose(lg.cell_centers[:, 6:], g2.cell_centers))
        self.assertTrue(np.allclose(old2new.A, proj_known))

    def test_coarse_cell_ref_after_fine_cell_ref(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 3)

        old2new, _ = lg.refine_cells(0)  # refine cell 0
        old2new = lg.refine_cells(3)[0] * old2new  # cell 3 is first cell of level 1
        old2new = lg.refine_cells(0)[0] * old2new  # Cell 0 is still on level 0

        # This should be equivalent to refinging cell 0, 1 and then cell 2
        lg_ref = pp.CartLeafGrid([2, 2], [1, 1], 3)

        old2new_ref = lg_ref.refine_cells([0, 1])[0]
        old2new_ref = lg_ref.refine_cells(2)[0] * old2new_ref

        self._compare_grids(lg, lg_ref)
        self.assertTrue(np.allclose(old2new.A, old2new_ref.A))

    def test_recursive_refinement(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 3)

        old2new, _ = lg.refine_cells(0)
        old2new = lg.refine_cells(3)[0] * old2new

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

        proj_ref = np.array(
            [
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            ]
        ).T

        self.assertTrue(np.allclose(lg.nodes, nodes))
        self.assertTrue(np.allclose(lg.face_centers, face_centers))
        self.assertTrue(np.allclose(lg.cell_centers, cell_centers))
        self.assertTrue(np.allclose(old2new.A, proj_ref))

    def test_refinement_singel_cell(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 2)
        old2new, _ = lg.refine_cells(0)

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

        proj_ref = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ]
        )

        self.assertTrue(np.allclose(lg.nodes, nodes))
        self.assertTrue(np.allclose(lg.face_nodes.A, face_nodes))
        self.assertTrue(np.allclose(lg.cell_faces.A, cell_faces))
        self.assertTrue(np.allclose(old2new.A, proj_ref))

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

        old2new, old2new_f = lg.refine_cells(0)
        old2new_tmp, old2new_f_tmp = lg.refine_cells(4)  # Should refine cell 0 as well
        old2new = old2new_tmp * old2new
        old2new_f = old2new_f_tmp * old2new_f

        lg_ref = pp.CartLeafGrid([2, 2], [1, 1], 3)
        old2new_ref, old2new_f_ref = lg_ref.refine_cells([0, 1])
        old2new_ref_tmp, old2new_f_ref_tmp = lg_ref.refine_cells(3)
        old2new_ref = old2new_ref_tmp * old2new_ref
        old2new_f_ref = old2new_f_ref_tmp * old2new_f_ref

        self._compare_grids(lg, lg_ref)
        self.assertTrue(np.allclose(old2new.A, old2new_ref.A))
        self.assertTrue(np.allclose(old2new_f.A, old2new_f_ref.A))

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

        old2new, _ = lg.refine_cells(0)
        old2new = lg.refine_cells(1)[0] * old2new
        old2new = (
            lg.refine_cells(7)[0] * old2new
        )  # should refine cell 0, 1, and 2 as well

        lg_ref = pp.CartLeafGrid([1, 2], [1, 1], 4)
        old2new_ref = lg_ref.refine_cells([0, 1])[0]
        old2new_ref = lg_ref.refine_cells([0, 1, 2])[0] * old2new_ref
        old2new_ref = lg_ref.refine_cells(10)[0] * old2new_ref
        self._compare_grids(lg, lg_ref)
        self.assertTrue(np.allclose(old2new.A, old2new_ref.A))

    def test_coarsening_full_grid(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 2)
        old2new, _ = lg.refine_cells(np.ones(4, dtype=bool))
        old2new = lg.coarsen_cells(np.ones(16, dtype=bool))[0] * old2new

        g_t = pp.CartGrid([2, 2], [1, 1])
        g_t.compute_geometry()

        proj_ref = np.eye(4)
        self._compare_grids(lg, g_t)
        self.assertTrue(np.allclose(old2new.A, proj_ref))

    def test_coarsening_one_cell(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 2)
        old2new, old2new_f = lg.refine_cells(np.ones(4, dtype=bool))
        old2new_tmp, old2new_f_tmp = lg.coarsen_cells([0, 1, 4, 5])
        old2new = old2new_tmp * old2new
        old2new_f = old2new_f_tmp * old2new_f

        lg_ref = pp.CartLeafGrid([2, 2], [1, 1], 2)
        lg_ref.refine_cells([1, 2, 3])
        proj_ref = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            ]
        ).T
        proj_f_ref = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        self._compare_grids(lg, lg_ref)
        self.assertTrue(np.allclose(old2new.A, proj_ref))
        self.assertTrue(np.allclose(old2new_f.A, proj_f_ref))

    def test_coarsening_multiple_cells(self):
        """ Refine cell 1 and 2. Refine and coarsen back cell 0 and 3"""
        lg = pp.CartLeafGrid([2, 2], [1, 1], 2)
        old2new, _ = lg.refine_cells(np.ones(4, dtype=bool))
        old2new = lg.coarsen_cells([0, 1, 4, 5, 10, 11, 14, 15])[0] * old2new

        lg_ref = pp.CartLeafGrid([2, 2], [1, 1], 2)
        lg_ref.refine_cells([1, 2])
        proj_ref = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ).T

        self._compare_grids(lg, lg_ref)
        self.assertTrue(np.allclose(old2new.A, proj_ref))

    def test_coarsening_multiple_levels(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 3)
        old2new, _ = lg.refine_cells([0, 1])
        old2new = lg.refine_cells(2)[0] * old2new

        old2new = lg.coarsen_cells([3, 4, 7, 8, 9, 10, 11, 12])[0] * old2new

        lg_ref = pp.CartLeafGrid([2, 2], [1, 1], 2)
        lg_ref.refine_cells(0)
        proj_ref = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )

        self._compare_grids(lg, lg_ref)
        self.assertTrue(np.allclose(old2new.A, proj_ref))

    def test_coarsening_no_prop(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 3)
        lg.refine_cells([0, 1, 2, 3])
        lg.refine_cells(1)
        old2new, _ = lg.coarsen_cells([1, 2, 5, 6])

        lg_ref = pp.CartLeafGrid([2, 2], [1, 1], 3)
        lg_ref.refine_cells([0, 1, 2, 3])
        lg_ref.refine_cells(1)

        proj_ref = np.eye(19)

        self._compare_grids(lg, lg_ref)
        self.assertTrue(np.allclose(old2new.A, proj_ref))

    def test_face_ref_periodic_boundary(self):
        """ refine one cell on the left periodic boundary, check that
            the face on the righ boundary is also refined
        """
        lg = pp.CartLeafGrid([2, 2], [1, 1], 2)

        lg.per_map = np.array([[6, 7], [10, 11]])
        lg.level_grids[0].per_map = np.array([[6, 7], [10, 11]])
        lg.level_grids[1].per_map = np.array([[20, 21, 22, 23], [36, 37, 38, 39]])
        lg.level_grids[1].face_centers[:, lg.level_grids[1].per_map.ravel()]
        lg.refine_cells(0)

        nodes = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.25, 0.0],
                [0.25, 0.25, 0.0],
                [0.5, 0.25, 0.0],
                [0.0, 0.5, 0.0],
                [0.25, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.25, 1.0, 0.0],
                [0.5, 1.0, 0.0],
            ]
        ).T

        face_nodes = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        cell_faces = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
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
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        per_ref = np.array([[4, 13, 14], [6, 19, 20]])

        self.assertTrue(np.allclose(lg.nodes, nodes))
        self.assertTrue(np.allclose(lg.face_nodes.A, face_nodes))
        self.assertTrue(np.allclose(lg.cell_faces.A, cell_faces))
        self.assertTrue(np.allclose(lg.per_map, per_ref))

    def test_face_ref_propagation_boundary(self):
        """ Test that there is maximum one level of refinement, also over
            the periodic boundary
        """
        lg = pp.CartLeafGrid([2, 2], [1, 1], 3)
        lg.per_map = np.array([[6, 7], [10, 11]])
        lg.level_grids[0].per_map = np.array([[6, 7], [10, 11]])
        lg.level_grids[1].per_map = np.array([[20, 21, 22, 23], [36, 37, 38, 39]])
        lg.level_grids[2].per_map = np.array(
            [[72, 73, 74, 75, 76, 77, 78, 79], [136, 137, 138, 139, 140, 141, 142, 143]]
        )

        lg.refine_cells(0)
        lg.refine_cells(3)

        lg_ref = pp.CartLeafGrid([2, 2], [1, 1], 3)
        lg_ref.refine_cells(0)
        lg_ref.refine_cells([1, 3])

        per_ref = np.array([[2, 15, 28, 29], [4, 21, 34, 35]])

        self.assertTrue(np.allclose(lg.cell_volumes, lg_ref.cell_volumes))
        self.assertTrue(np.allclose(lg.cell_centers, lg_ref.cell_centers))
        self.assertTrue(np.allclose(lg.per_map, per_ref))

    def _compare_grids(self, g0, g1):
        self.assertTrue(np.allclose(g0.nodes, g1.nodes))
        self.assertTrue(np.allclose(g0.face_nodes.A, g1.face_nodes.A))
        self.assertTrue(np.allclose(g0.cell_faces.A, g1.cell_faces.A))

        self.assertTrue(np.allclose(g0.face_areas, g1.face_areas))
        self.assertTrue(np.allclose(g0.face_centers, g1.face_centers))
        self.assertTrue(np.allclose(g0.face_normals, g1.face_normals))
        self.assertTrue(np.allclose(g0.cell_volumes, g1.cell_volumes))
        self.assertTrue(np.allclose(g0.cell_centers, g1.cell_centers))


class TestCartLeafGrid_1d(unittest.TestCase):
    def test_generation_2_levels(self):
        lg = pp.CartLeafGrid(2, 1, 2)

        g_t = [pp.CartGrid(2, 1), pp.CartGrid(4, 1)]

        for i, g in enumerate(lg.level_grids):
            self.assertTrue(np.allclose(g.nodes, g_t[i].nodes))
            self.assertTrue(np.allclose(g.face_nodes.A, g_t[i].face_nodes.A))
            self.assertTrue(np.allclose(g.cell_faces.A, g_t[i].cell_faces.A))

    def test_generation_3_levels(self):
        lg = pp.CartLeafGrid(1, 1, 3)

        g_t = [
            pp.CartGrid(1, 1),
            pp.CartGrid(2, 1),
            pp.CartGrid(4, 1),
        ]

        for i, g in enumerate(lg.level_grids):
            self.assertTrue(np.allclose(g.nodes, g_t[i].nodes))
            self.assertTrue(np.allclose(g.face_nodes.A, g_t[i].face_nodes.A))
            self.assertTrue(np.allclose(g.cell_faces.A, g_t[i].cell_faces.A))

    def test_refinement_full_grid(self):
        lg = pp.CartLeafGrid(2, 1, 2)
        old_to_new, _ = lg.refine_cells(np.ones(2, dtype=bool))
        g_t = pp.CartGrid(4, 1)

        proj_ref = np.array([[1, 0], [1, 0], [0, 1], [0, 1],])
        self.assertTrue(np.allclose(lg.nodes, g_t.nodes))
        self.assertTrue(np.allclose(lg.face_nodes.A, g_t.face_nodes.A))
        self.assertTrue(np.allclose(lg.cell_faces.A, g_t.cell_faces.A))
        self.assertTrue(np.allclose(old_to_new.A, proj_ref))

    def test_refinement_multiple_levels(self):
        lg = pp.CartLeafGrid(1, 1, 3)
        old2new, _ = lg.refine_cells(0)
        old2new = lg.refine_cells(1)[0] * old2new
        nodes = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.75, 0.0, 0.0], [1.0, 0.0, 0.0],]
        ).T

        face_nodes = sps.csc_matrix(
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],])
        )

        cell_faces = sps.csc_matrix(
            np.array(
                [[-1.0, 0.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0],]
            )
        )
        g_t = pp.Grid(1, nodes, face_nodes, cell_faces, ["Ref"])
        proj_known = np.array([[1, 1, 1]]).T
        g_t.compute_geometry()
        self._compare_grids(lg, g_t)
        self.assertTrue(np.allclose(old2new.A, proj_known))

    def test_coarse_cell_ref_after_fine_cell_ref(self):
        lg = pp.CartLeafGrid(2, 1, 3)

        old2new, _ = lg.refine_cells(0)  # refine cell 0
        old2new = lg.refine_cells(1)[0] * old2new  # cell 2 is first cell of level 1
        old2new = lg.refine_cells(0)[0] * old2new  # Cell 0 is still on level 0

        # This should be equivalent to refinging cell 0, 1 and then cell 0
        lg_ref = pp.CartLeafGrid(2, 1, 3)

        old2new_ref, _ = lg_ref.refine_cells([0, 1])
        old2new_ref = lg_ref.refine_cells(0)[0] * old2new_ref

        self._compare_grids(lg, lg_ref)
        self.assertTrue(np.allclose(old2new.A, old2new_ref.A))

    def test_max_one_level_ref_rec(self):
        """
        Refine CartGrid(2, 1) to:

        |---|-|-|---|---|
        """
        lg = pp.CartLeafGrid(2, 1, 4)

        old2new, old2new_f = lg.refine_cells(0)
        old2new_tmp, old2new_f_tmp = lg.refine_cells(2)  # should refine cell 0 as well
        old2new = old2new_tmp * old2new
        old2new_f = old2new_f_tmp * old2new_f

        lg_ref = pp.CartLeafGrid(2, 1, 4)
        old2new_ref, old2new_f_ref = lg_ref.refine_cells([0, 1])
        old2new_ref_tmp, old2new_f_ref_tmp = lg_ref.refine_cells(1)
        old2new_ref = old2new_ref_tmp * old2new_ref
        old2new_f_ref = old2new_f_ref_tmp * old2new_f_ref

        self._compare_grids(lg, lg_ref)
        self.assertTrue(np.allclose(old2new.A, old2new_ref.A))
        self.assertTrue(np.allclose(old2new_f.A, old2new_f_ref.A))

    def test_coarsening_full_grid(self):
        lg = pp.CartLeafGrid(2, 1, 2)
        old2new, old2new_f = lg.refine_cells(np.ones(2, dtype=bool))
        old2new_tmp, old2new_f_tmp = lg.coarsen_cells(np.ones(4, dtype=bool))
        old2new = old2new_tmp * old2new
        old2new_f = old2new_f_tmp * old2new_f

        g_t = pp.CartGrid(2, 1)
        g_t.compute_geometry()

        proj_ref = np.eye(2)
        proj_f_ref = np.eye(3)

        self._compare_grids(lg, g_t)
        self.assertTrue(np.allclose(old2new.A, proj_ref))
        self.assertTrue(np.allclose(old2new_f.A, proj_f_ref))

    def test_coarsening_one_cell(self):
        lg = pp.CartLeafGrid(2, 1, 2)
        old2new, _ = lg.refine_cells(np.ones(2, dtype=bool))
        old2new = lg.coarsen_cells([0, 1])[0] * old2new

        lg_ref = pp.CartLeafGrid(2, 1, 2)
        lg_ref.refine_cells(1)
        proj_ref = np.array([[1, 0, 0], [0, 1, 1],]).T

        self._compare_grids(lg, lg_ref)
        self.assertTrue(np.allclose(old2new.A, proj_ref))

    def _compare_grids(self, g0, g1):
        self.assertTrue(np.allclose(g0.nodes, g1.nodes))
        self.assertTrue(np.allclose(g0.face_nodes.A, g1.face_nodes.A))
        self.assertTrue(np.allclose(g0.cell_faces.A, g1.cell_faces.A))

        self.assertTrue(np.allclose(g0.face_areas, g1.face_areas))
        self.assertTrue(np.allclose(g0.face_centers, g1.face_centers))
        self.assertTrue(np.allclose(g0.face_normals, g1.face_normals))
        self.assertTrue(np.allclose(g0.cell_volumes, g1.cell_volumes))
        self.assertTrue(np.allclose(g0.cell_centers, g1.cell_centers))
