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

        g_t = [pp.CartGrid([1, 2], [1, 1]),
               pp.CartGrid([2, 4], [1, 1]),
               pp.CartGrid([4, 8], [1, 1])]

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

    def test_refinement_several_times(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 2)

        lg.refine_cells(0)
        lg.refine_cells(0)
        lg.refine_cells(0)
        lg.refine_cells(0)

        g_t = pp.CartGrid([4, 4], [1, 1])

        self.assertTrue(np.allclose(lg.nodes, g_t.nodes))
        self.assertTrue(np.allclose(lg.face_nodes.A, g_t.face_nodes.A))
        self.assertTrue(np.allclose(lg.cell_faces.A, g_t.cell_faces.A))

        
    def test_refinement_singel_cell(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 2)
        lg.refine_cells(0)

        nodes = np.array([
            [1, 1, 0, 0.5, 1, 0, 0.25, 0.5 ,0, 0.25, 0.5,0.  , 0.25, 0.5 ],
            [0, 0.5 , 1, 1, 1, 0, 0, 0, 0.25, 0.25, 0.25, 0.5 , 0.5 , 0.5 ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        face_nodes = np.array([
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
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        ]).T
        cell_faces = np.array([
            [ 1.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0., -1.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  1., -1.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  1.,  0.,  0.,  0.,  0.],
            [-1.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 1.,  0., -1.,  0.,  0.,  0.,  0.],
            [ 0.,  1.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  1.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0., -1.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  1., -1.,  0.,  0.],
            [-1.,  0.,  0.,  0.,  1.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  1., -1.],
            [-1.,  0.,  0.,  0.,  0.,  0.,  1.],
            [ 0.,  0.,  0., -1.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0., -1.,  0.,  0.],
            [ 0.,  0.,  0.,  1.,  0., -1.,  0.],
            [ 0.,  0.,  0.,  0.,  1.,  0., -1.],
            [ 0., -1.,  0.,  0.,  0.,  1.,  0.],
            [ 0., -1.,  0.,  0.,  0.,  0.,  1.]
        ])

        self.assertTrue(np.allclose(lg.nodes, nodes))
        self.assertTrue(np.allclose(lg.face_nodes.A, face_nodes))
        self.assertTrue(np.allclose(lg.cell_faces.A, cell_faces))
