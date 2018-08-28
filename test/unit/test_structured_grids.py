from __future__ import division
import numpy as np
import unittest

from porepy.grids import structured, simplex
from porepy.utils import setmembership


def set_tol():
    return 1e-8


class TestCartGrideGeometry2DUnpert(unittest.TestCase):
    def setUp(self):
        self.tol = set_tol()
        nc = np.array([2, 3])
        self.g = structured.CartGrid(nc)
        self.g.compute_geometry()

    def test_node_coord(self):
        xn = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        yn = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        assert np.isclose(xn, self.g.nodes[0]).all()
        assert np.isclose(yn, self.g.nodes[1]).all()

    def test_faces_areas(self):
        areas = 1 * np.ones(self.g.num_faces)
        assert np.isclose(areas, self.g.face_areas).all()

    def test_cell_volumes(self):
        volumes = 1 * np.ones(self.g.num_cells)
        assert np.isclose(volumes, self.g.cell_volumes).all()

    if __name__ == "__main__":
        unittest.main()


class TestCartGridGeometry2DPert(unittest.TestCase):
    def setUp(self):
        self.tol = set_tol()
        nc = np.array([2, 2])
        g = structured.CartGrid(nc)
        g.nodes[0, 4] = 1.5
        g.compute_geometry()
        self.g = g

    def test_node_coord(self):
        xn = np.array([0, 1, 2, 0, 1.5, 2, 0, 1, 2])
        yn = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        assert np.isclose(xn, self.g.nodes[0]).all()
        assert np.isclose(yn, self.g.nodes[1]).all()

    def test_faces_areas(self):
        areas = np.array(
            [1, np.sqrt(1.25), 1, 1, np.sqrt(1.25), 1, 1, 1, 1.5, 0.5, 1, 1]
        )
        assert np.isclose(areas, self.g.face_areas).all()

    def test_cell_volumes(self):
        volumes = np.array([1.25, 0.75, 1.25, 0.75])
        assert np.isclose(volumes, self.g.cell_volumes).all()

    def test_face_normals(self):
        nx = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        ny = np.array([0, -0.5, 0, 0, 0.5, 0, 1, 1, 1.5, 0.5, 1, 1])
        fn = self.g.face_normals
        assert np.isclose(nx, fn[0]).all()
        assert np.isclose(ny, fn[1]).all()

    if __name__ == "__main__":
        unittest.main()


class TestCartGridGeometryUnpert3D(unittest.TestCase):
    def setUp(self):
        nc = 2 * np.ones(3, dtype=np.int)
        g = structured.CartGrid(nc)
        g.compute_geometry()
        self.g = g

    def test_node_coord(self):
        x = np.array(
            [
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,  # z == 0
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,  # z == 1
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
            ]
        )  # z == 2
        y = np.array(
            [
                0,
                0,
                0,
                1,
                1,
                1,
                2,
                2,
                2,
                0,
                0,
                0,
                1,
                1,
                1,
                2,
                2,
                2,
                0,
                0,
                0,
                1,
                1,
                1,
                2,
                2,
                2,
            ]
        )
        z = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
            ]
        )

        assert np.isclose(x, self.g.nodes[0]).all()
        assert np.isclose(y, self.g.nodes[1]).all()
        assert np.isclose(z, self.g.nodes[2]).all()

    def test_faces_areas(self):
        areas = 1 * np.ones(self.g.num_faces)
        assert np.isclose(areas, self.g.face_areas).all()

    def test_face_normals(self):
        face_per_dim = 3 * 2 * 2
        ones = np.ones(face_per_dim)
        zeros = np.zeros(face_per_dim)

        nx = np.hstack((ones, zeros, zeros))
        ny = np.hstack((zeros, ones, zeros))
        nz = np.hstack((zeros, zeros, ones))

        fn = self.g.face_normals
        assert np.isclose(nx, fn[0]).all()
        assert np.isclose(ny, fn[1]).all()
        assert np.isclose(nz, fn[2]).all()

    def test_cell_volumes(self):
        volumes = 1 * np.ones(self.g.num_cells)
        assert np.isclose(volumes, self.g.cell_volumes).all()

    def test_cell_centers(self):
        cx = np.array([0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
        cy = np.array([0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5])
        cz = np.array([0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5])

        cc = self.g.cell_centers

        assert np.isclose(cx, cc[0]).all()
        assert np.isclose(cy, cc[1]).all()
        assert np.isclose(cz, cc[2]).all()

    if __name__ == "__main__":
        unittest.main()


class TestCartGridGeometry1CellPert3D(unittest.TestCase):
    def setUp(self):
        nc = np.ones(3, dtype=np.int)
        g = structured.CartGrid(nc)
        g.nodes[:, -1] = [2, 2, 2]
        g.compute_geometry()
        self.g = g

    def node_coord(self):
        x = np.array([0, 1, 0, 1, 0, 1, 0, 2])
        y = np.array([0, 0, 1, 1, 0, 0, 1, 2])
        z = np.array([0, 0, 0, 0, 1, 1, 1, 2])

        assert np.isclose(x, self.g.nodes[0]).all()
        assert np.isclose(y, self.g.nodes[1]).all()
        assert np.isclose(z, self.g.nodes[2]).all()

    def test_faces_areas(self):
        a1 = 1
        a2 = 2.159875808805010  # Face area computed another place
        areas = np.array([a1, a2, a1, a2, a1, a2])
        assert np.isclose(areas, self.g.face_areas).all()

    def test_face_normals(self):
        nx = np.array([1, 2, 0, -0.5, 0, -0.5])
        ny = np.array([0, -0.5, 1, 2, 0, -0.5])
        nz = np.array([0, -0.5, 0, -0.5, 1, 2])

        fn = self.g.face_normals
        assert np.isclose(nx, fn[0]).all()
        assert np.isclose(ny, fn[1]).all()
        assert np.isclose(nz, fn[2]).all()

    def test_face_centers(self):
        f1 = 1.294658198738520
        f2 = 0.839316397477041

        fx = np.array([0, f1, 0.5, f2, 0.5, f2])
        fy = np.array([0.5, f2, 0, f1, 0.5, f2])
        fz = np.array([0.5, f2, 0.5, f2, 0, f1])

        fc = self.g.face_centers

        assert np.isclose(fx, fc[0]).all()
        assert np.isclose(fy, fc[1]).all()
        assert np.isclose(fz, fc[2]).all()

    def test_cell_volumes(self):
        assert np.isclose(self.g.cell_volumes, 1.75)

    def test_cell_centers(self):
        # Center coordinate imported from other places
        cx = 0.717261904761905
        cy = cx
        cz = cx
        cc = self.g.cell_centers
        assert np.isclose(cx, cc[0]).all()
        assert np.isclose(cy, cc[1]).all()
        assert np.isclose(cz, cc[2]).all()

    if __name__ == "__main__":
        unittest.main()


class TestStructuredTriangleGridGeometry(unittest.TestCase):
    """ Create simplest possible configuration, test sanity of geometric
    quantities.
    """

    def setUp(self):
        g = simplex.StructuredTriangleGrid([1, 1])
        g.compute_geometry()
        self.g = g

    def test_node_coords(self):
        nodes = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
        assert np.allclose(self.g.nodes, nodes)

    def test_face_centers(self):
        fc = np.array([[0.5, 0, 0.5, 1, 0.5], [0, 0.5, 0.5, 0.5, 1], [0, 0, 0, 0, 0]])
        assert np.allclose(self.g.face_centers, fc)

    def test_cell_centers(self):
        cc = np.array([[2 / 3, 1 / 3], [1 / 3, 2 / 3], [0, 0]])
        assert np.allclose(self.g.cell_centers, cc)

    def test_cell_volumes(self):
        cv = np.array([0.5, 0.5])
        assert np.allclose(self.g.cell_volumes, cv)

    def test_face_areas(self):
        fa = np.array([1, 1, np.sqrt(2), 1, 1])
        assert np.allclose(self.g.face_areas, fa)

    def test_face_normals(self):
        fn = np.array([[0, -1, -1, 1, 0], [-1, 0, 1, 0, 1], [0, 0, 0, 0, 0]])
        assert np.allclose(self.g.face_normals, fn)

    if __name__ == "__main__":
        unittest.main()


class TestStructuredTetrahedralGrid(unittest.TestCase):
    def setUp(self):
        g = simplex.StructuredTetrahedralGrid([1, 1, 1])
        g.compute_geometry()
        self.g = g

        # The ordering of faces may differ depending on the test system (presumably version of scipy or similar). Below are hard-coded combination of face-nodes, and the corresponding faces and face_areas.
        self.fn = np.array(
            [
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 5],
                [1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 3, 4, 5, 5, 6, 5, 6],
                [2, 4, 4, 3, 4, 6, 5, 6, 5, 6, 6, 6, 6, 6, 7, 7, 6, 7],
            ]
        )
        self.face_areas = np.array(
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.8660254,
                0.70710678,
                0.5,
                0.70710678,
                0.5,
                0.70710678,
                0.70710678,
                0.5,
                0.5,
                0.8660254,
                0.5,
                0.5,
                0.5,
                0.5,
            ]
        )
        self.face_center = np.array(
            [
                [
                    0.33333333,
                    0.33333333,
                    0.,
                    0.66666667,
                    0.33333333,
                    0.33333333,
                    1.,
                    0.66666667,
                    0.66666667,
                    0.33333333,
                    0.66666667,
                    0.33333333,
                    0.,
                    0.66666667,
                    1.,
                    0.66666667,
                    0.33333333,
                    0.66666667,
                ],
                [
                    0.33333333,
                    0.,
                    0.33333333,
                    0.66666667,
                    0.33333333,
                    0.66666667,
                    0.33333333,
                    0.66666667,
                    0.,
                    0.33333333,
                    0.33333333,
                    1.,
                    0.66666667,
                    0.66666667,
                    0.66666667,
                    1.,
                    0.33333333,
                    0.66666667,
                ],
                [
                    0.,
                    0.33333333,
                    0.33333333,
                    0.,
                    0.33333333,
                    0.33333333,
                    0.33333333,
                    0.33333333,
                    0.66666667,
                    0.66666667,
                    0.66666667,
                    0.33333333,
                    0.66666667,
                    0.66666667,
                    0.66666667,
                    0.66666667,
                    1.,
                    1.,
                ],
            ]
        )

    def test_face_centers_areas(self):
        face_nodes = self.g.face_nodes.indices.reshape((3, self.g.num_faces), order="F")
        ismem, ind_map = setmembership.ismember_rows(self.fn, face_nodes)
        assert np.all(ismem)

        assert np.allclose(self.face_areas, self.g.face_areas[ind_map])
        assert np.allclose(self.face_center, self.g.face_centers[:, ind_map])

    def test_node_coords(self):
        nodes = np.array(
            [
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 0, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ]
        )
        assert np.allclose(self.g.nodes, nodes)

    def test_cell_centers(self):
        cc = np.array(
            [
                [1 / 4, 1 / 4, 1 / 4],
                [1 / 4, 1 / 2, 1 / 2],
                [1 / 2, 1 / 4, 3 / 4],
                [1 / 2, 3 / 4, 1 / 4],
                [3 / 4, 1 / 2, 1 / 2],
                [3 / 4, 3 / 4, 3 / 4],
            ]
        ).T
        assert np.allclose(self.g.cell_centers, cc)

    def test_cell_volumes(self):
        cv = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
        assert np.allclose(self.g.cell_volumes, cv)

    if __name__ == "__main__":
        unittest.main()


class TestStructuredSimplexGridCoverage(unittest.TestCase):
    """ Verify that the tessalation covers the whole domain.
    """

    def test_coverage_triangles(self):
        domain = np.array([2, 3])
        grid_size = np.array([3, 5])
        g = simplex.StructuredTriangleGrid(grid_size, domain)
        g.compute_geometry()
        assert np.abs(domain.prod() - np.sum(g.cell_volumes)) < 1e-10

    def test_coverage_tets(self):
        domain = np.array([2, 3, 0.7])
        grid_size = np.array([3, 5, 2])
        g = simplex.StructuredTetrahedralGrid(grid_size, domain)
        g.compute_geometry()
        assert np.abs(domain.prod() - np.sum(g.cell_volumes)) < 1e-10

    if __name__ == "__main__":
        unittest.main()
