
import numpy as np
import unittest

from core.grids import structured


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

    if __name__ == '__main__':
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
        areas = np.array([1, np.sqrt(1.25), 1, 1, np.sqrt(1.25), 1, 1, 1,
                          1.5, 0.5, 1, 1])
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

    if __name__ == '__main__':
        unittest.main()


class TestCartGridGeometryUnpert3D(unittest.TestCase):

    def setUp(self):
        nc = 2 * np.ones(3)
        g = structured.CartGrid(nc)
        g.compute_geometry()
        self.g = g

    def test_node_coord(self):
        x = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2,   # z == 0
                      0, 1, 2, 0, 1, 2, 0, 1, 2,   # z == 1
                      0, 1, 2, 0, 1, 2, 0, 1, 2])  # z == 2
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2,
                      0, 0, 0, 1, 1, 1, 2, 2, 2,
                      0, 0, 0, 1, 1, 1, 2, 2, 2,
                      ])
        z = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 1, 1, 1, 1,
                      2, 2, 2, 2, 2, 2, 2, 2, 2])

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

    if __name__ == '__main__':
        unittest.main()


class TestCartGridGeometry1CellPert3D(unittest.TestCase):

    def setUp(self):
        nc = np.ones(3)
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
        print(cx)
        print(cc[:])
        assert np.isclose(cx, cc[0]).all()
        assert np.isclose(cy, cc[1]).all()
        assert np.isclose(cz, cc[2]).all()

    if __name__ == '__main__':
        unittest.main()
