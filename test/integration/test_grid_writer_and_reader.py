import unittest

import numpy as np
import scipy.sparse as sps
import os

import porepy as pp


class TestWriterReader(unittest.TestCase):
    """
    We test the grid writer and reader by dumping the grid
    to file, then read it, and compare it to the original grid.
    """
    def test_2d_cartesian_grid(self):
        """
        Test 2d cartesian grid
        """
        g = pp.CartGrid([2, 2], [1, 1])
        g.compute_geometry()

        pp.io.grid_writer.dump_grid_to_file(g, "test_grid_temp.txt")
        g_inn = pp.io.grid_reader.read_grid("test_grid_temp.txt")
        os.remove("test_grid_temp.txt")

        self.assertTrue(self._grids_are_equal(g, g_inn))

    def test_2d_cartesian_grid_cell_facetag(self):
        """
        Test 2d cartesian grid
        """
        g = pp.CartGrid([2, 2], [1, 1])
        g.compute_geometry()
        g.cell_facetag = np.random.randint(0, 4, g.cell_faces.indices.size)

        pp.io.grid_writer.dump_grid_to_file(g, "test_grid_temp.txt")
        g_inn = pp.io.grid_reader.read_grid("test_grid_temp.txt")
        os.remove("test_grid_temp.txt")

        self.assertTrue(self._grids_are_equal(g, g_inn))


    def test_3d_cartesian_grid(self):
        """
        Test 2d cartesian grid
        """
        g = pp.CartGrid([2, 2, 3], [1, 1, 2])
        g.compute_geometry()

        pp.io.grid_writer.dump_grid_to_file(g, "test_grid_temp.txt")
        g_inn = pp.io.grid_reader.read_grid("test_grid_temp.txt")
        os.remove("test_grid_temp.txt")

        self.assertTrue(self._grids_are_equal(g, g_inn))

    def test_3d_cartesian_grid_cell_facetag(self):
        """
        Test 2d cartesian grid
        """
        g = pp.CartGrid([2, 2, 3], [1, 1, 2])
        g.compute_geometry()
        g.cell_facetag = np.random.randint(0, 6, g.cell_faces.indices.size)

        pp.io.grid_writer.dump_grid_to_file(g, "test_grid_temp.txt")
        g_inn = pp.io.grid_reader.read_grid("test_grid_temp.txt")
        os.remove("test_grid_temp.txt")

        self.assertTrue(self._grids_are_equal(g, g_inn))

    def test_2d_simplex_grid(self):
        """
        Test 2d cartesian grid
        """
        g = pp.StructuredTriangleGrid([2, 3], [1, 1])
        g.compute_geometry()

        pp.io.grid_writer.dump_grid_to_file(g, "test_grid_temp.txt")
        g_inn = pp.io.grid_reader.read_grid("test_grid_temp.txt")
        os.remove("test_grid_temp.txt")

        self.assertTrue(self._grids_are_equal(g, g_inn))

    def test_3d_simplex_grid(self):
        """
        Test 2d cartesian grid
        """
        g = pp.StructuredTetrahedralGrid([3, 3, 1], [5, 1, 1])
        g.compute_geometry()

        pp.io.grid_writer.dump_grid_to_file(g, "test_grid_temp.txt")
        g_inn = pp.io.grid_reader.read_grid("test_grid_temp.txt")
        os.remove("test_grid_temp.txt")

        self.assertTrue(self._grids_are_equal(g, g_inn))

    def _grids_are_equal(self, g0, g1):
        if not np.allclose(g0.nodes, g1.nodes):
            return False
        if not np.allclose(g0.face_nodes.indices, g1.face_nodes.indices):
            return False
        if not np.allclose(g0.face_nodes.indptr, g1.face_nodes.indptr):
            return False
        if not np.allclose(g0.face_nodes.data, g1.face_nodes.data):
            return False
        if not np.allclose(g0.face_centers, g1.face_centers):
            return False
        if not np.allclose(g0.face_areas, g1.face_areas):
            return False
        if not np.allclose(g0.face_normals, g1.face_normals):
            return False
        if not np.allclose(g0.cell_faces.indices, g1.cell_faces.indices):
            return False
        if not np.allclose(g0.cell_faces.indptr, g1.cell_faces.indptr):
            return False
        if not np.allclose(g0.cell_faces.data, g1.cell_faces.data):
            return False
        if not np.allclose(g0.cell_volumes, g1.cell_volumes):
            return False
        if hasattr(g0, "cell_facetag"):
            if not np.allclose(g0.cell_facetag, g1.cell_facetag):
                return False
        return True

if __name__ == "__main__":
    unittest.main()
