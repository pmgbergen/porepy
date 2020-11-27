import unittest

import numpy as np

import porepy as pp


class FvSubGrid(unittest.TestCase):
    def test_cart_2d(self):
        g = pp.CartGrid([2, 1], [1, 1])
        g.compute_geometry()
        eta = np.pi
        gsub = pp.FvSubGrid(g, eta)
        self.compare_grids(g, gsub, eta)

    def test_simplex_2d(self):
        pts = np.random.rand(2, 6)
        g = pp.TriangleGrid(pts)
        g.compute_geometry()
        eta = np.pi
        gsub = pp.FvSubGrid(g, eta)
        self.compare_grids(g, gsub, eta)

    def test_cart32d(self):
        g = pp.CartGrid([2, 1, 3], [1, 1, 1])
        g.compute_geometry()
        eta = np.pi
        gsub = pp.FvSubGrid(g, eta)
        self.compare_grids(g, gsub, eta)

    def test_simplex_3d(self):
        pts = np.random.rand(3, 6)
        g = pp.TetrahedralGrid(pts)
        g.compute_geometry()
        eta = np.pi
        gsub = pp.FvSubGrid(g, eta)
        self.compare_grids(g, gsub, eta)

    def compare_grids(self, g, gsub, eta):
        s_t = pp.numerics.fv.fvutils.SubcellTopology(g)

        continuity_point = g.face_centers[:, s_t.fno_unique] + eta * (
            g.nodes[:, s_t.nno_unique] - g.face_centers[:, s_t.fno_unique]
        )
        self.assertTrue(np.allclose(gsub.nodes, g.nodes))
        self.assertTrue(np.allclose(gsub.face_centers, continuity_point))
        self.assertTrue(np.allclose(gsub.face_nodes.indices, s_t.nno_unique))
        self.assertTrue(np.allclose(gsub.cell_nodes().indices, g.cell_nodes().indices))


if __name__ == "__main__":
    unittest.main()
