import numpy as np
import scipy.sparse as sps
import unittest

import porepy as pp


class RobinBoundTest(unittest.TestCase):
    def test_dir_rob(self):
        nx = 2
        ny = 2
        g = pp.CartGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()
        c = pp.FourthOrderTensor(2, np.ones(g.num_cells), np.ones(g.num_cells))
        robin_weight = np.pi

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        dir_ind = np.ravel(np.argwhere(left + bot + top))
        neu_ind = np.ravel(np.argwhere([]))
        rob_ind = np.ravel(np.argwhere(right))

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryCondition(g, bnd_ind, names)

        def u_ex(x):
            return np.vstack((x[0], 0 * x[1]))

        def T_ex(faces):
            if np.size(faces) == 0:
                return np.atleast_2d(np.array([]))
            sigma = np.array([[3, 0], [0, 1]])
            T_r = [np.dot(sigma, g.face_normals[:2, f]) for f in faces]
            return np.vstack(T_r).T

        u_bound = np.zeros((2, g.num_faces))

        sgn_n = pp.numerics.fracture_deformation.sign_of_faces(g, neu_ind)
        sgn_r = pp.numerics.fracture_deformation.sign_of_faces(g, rob_ind)

        u_bound[:, dir_ind] = u_ex(g.face_centers[:, dir_ind])
        u_bound[:, neu_ind] = T_ex(neu_ind) * sgn_n
        u_bound[:, rob_ind] = (
            T_ex(rob_ind) * sgn_r
            + robin_weight * u_ex(g.face_centers[:, rob_ind]) * g.face_areas[rob_ind]
        )
        u, T = self.solve_mpsa(g, c, robin_weight, bnd, u_bound)

        assert np.allclose(u, u_ex(g.cell_centers).ravel("F"))
        assert np.allclose(T, T_ex(np.arange(g.num_faces)).ravel("F"))

    def test_dir_neu_rob(self):
        nx = 2
        ny = 3
        g = pp.CartGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()
        c = pp.FourthOrderTensor(2, np.ones(g.num_cells), np.ones(g.num_cells))
        robin_weight = np.pi

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        dir_ind = np.ravel(np.argwhere(left))
        neu_ind = np.ravel(np.argwhere(top))
        rob_ind = np.ravel(np.argwhere(right + bot))

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryCondition(g, bnd_ind, names)

        def u_ex(x):
            return np.vstack((x[0], 0 * x[1]))

        def T_ex(faces):
            if np.size(faces) == 0:
                return np.atleast_2d(np.array([]))
            sigma = np.array([[3, 0], [0, 1]])
            T_r = [np.dot(sigma, g.face_normals[:2, f]) for f in faces]
            return np.vstack(T_r).T

        u_bound = np.zeros((2, g.num_faces))

        sgn_n = pp.numerics.fracture_deformation.sign_of_faces(g, neu_ind)
        sgn_r = pp.numerics.fracture_deformation.sign_of_faces(g, rob_ind)

        u_bound[:, dir_ind] = u_ex(g.face_centers[:, dir_ind])
        u_bound[:, neu_ind] = T_ex(neu_ind) * sgn_n
        u_bound[:, rob_ind] = (
            T_ex(rob_ind) * sgn_r
            + robin_weight * u_ex(g.face_centers[:, rob_ind]) * g.face_areas[rob_ind]
        )
        u, T = self.solve_mpsa(g, c, robin_weight, bnd, u_bound)

        assert np.allclose(u, u_ex(g.cell_centers).ravel("F"))
        assert np.allclose(T, T_ex(np.arange(g.num_faces)).ravel("F"))

    def test_structured_triang(self):
        nx = 1
        ny = 1
        g = pp.StructuredTriangleGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()
        c = pp.FourthOrderTensor(2, np.ones(g.num_cells), np.ones(g.num_cells))
        robin_weight = np.pi

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        dir_ind = np.ravel(np.argwhere(left))
        neu_ind = np.ravel(np.argwhere(top))
        rob_ind = np.ravel(np.argwhere(right + bot))

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryCondition(g, bnd_ind, names)

        def u_ex(x):
            return np.vstack((x[1], x[0]))

        def T_ex(faces):
            if np.size(faces) == 0:
                return np.atleast_2d(np.array([]))
            sigma = np.array([[0, 2], [2, 0]])
            T_r = [np.dot(sigma, g.face_normals[:2, f]) for f in faces]
            return np.vstack(T_r).T

        u_bound = np.zeros((2, g.num_faces))

        sgn_n = pp.numerics.fracture_deformation.sign_of_faces(g, neu_ind)
        sgn_r = pp.numerics.fracture_deformation.sign_of_faces(g, rob_ind)

        u_bound[:, dir_ind] = u_ex(g.face_centers[:, dir_ind])
        u_bound[:, neu_ind] = T_ex(neu_ind) * sgn_n
        u_bound[:, rob_ind] = (
            T_ex(rob_ind) * sgn_r
            + robin_weight * u_ex(g.face_centers[:, rob_ind]) * g.face_areas[rob_ind]
        )
        u, T = self.solve_mpsa(g, c, robin_weight, bnd, u_bound)

        assert np.allclose(u, u_ex(g.cell_centers).ravel("F"))
        assert np.allclose(T, T_ex(np.arange(g.num_faces)).ravel("F"))

    def test_unstruct_triang(self):
        corners = np.array([[0, 0, 1, 1], [0, 1, 1, 0]])
        points = np.random.rand(2, 2)
        points = np.hstack((corners, points))
        g = pp.TriangleGrid(points)
        g.compute_geometry()
        c = pp.FourthOrderTensor(2, np.ones(g.num_cells), np.ones(g.num_cells))
        robin_weight = np.pi

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        dir_ind = np.ravel(np.argwhere(left))
        neu_ind = np.ravel(np.argwhere(top))
        rob_ind = np.ravel(np.argwhere(right + bot))

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryCondition(g, bnd_ind, names)

        def u_ex(x):
            return np.vstack((x[1], x[0]))

        def T_ex(faces):
            if np.size(faces) == 0:
                return np.atleast_2d(np.array([]))
            sigma = np.array([[0, 2], [2, 0]])
            T_r = [np.dot(sigma, g.face_normals[:2, f]) for f in faces]
            return np.vstack(T_r).T

        u_bound = np.zeros((2, g.num_faces))

        sgn_n = pp.numerics.fracture_deformation.sign_of_faces(g, neu_ind)
        sgn_r = pp.numerics.fracture_deformation.sign_of_faces(g, rob_ind)

        u_bound[:, dir_ind] = u_ex(g.face_centers[:, dir_ind])
        u_bound[:, neu_ind] = T_ex(neu_ind) * sgn_n
        u_bound[:, rob_ind] = (
            T_ex(rob_ind) * sgn_r
            + robin_weight * u_ex(g.face_centers[:, rob_ind]) * g.face_areas[rob_ind]
        )
        u, T = self.solve_mpsa(g, c, robin_weight, bnd, u_bound)

        assert np.allclose(u, u_ex(g.cell_centers).ravel("F"))
        assert np.allclose(T, T_ex(np.arange(g.num_faces)).ravel("F"))

    def test_unstruct_tetrahedron(self):
        box = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}

        g = pp.meshing.simplex_grid([], box, mesh_size_min=3, mesh_size_frac=3)
        g = g.grids_of_dimension(3)[0]
        g = pp.CartGrid([4, 4, 4], [1, 1, 1])
        g.compute_geometry()
        c = pp.FourthOrderTensor(3, np.ones(g.num_cells), np.ones(g.num_cells))
        robin_weight = 1.0

        bot = g.face_centers[2] < 1e-10
        top = g.face_centers[2] > 1 - 1e-10
        west = g.face_centers[0] < 1e-10
        east = g.face_centers[0] > 1 - 1e-10
        north = g.face_centers[1] > 1 - 1e-10
        south = g.face_centers[1] < 1e-10

        dir_ind = np.ravel(np.argwhere(west + top))
        neu_ind = np.ravel(np.argwhere(bot))
        rob_ind = np.ravel(np.argwhere(east + north + south))

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryCondition(g, bnd_ind, names)

        def u_ex(x):
            return np.vstack((x[1], x[0], 0 * x[2]))

        def T_ex(faces):
            if np.size(faces) == 0:
                return np.atleast_2d(np.array([]))
            sigma = np.array([[0, 2, 0], [2, 0, 0], [0, 0, 0]])
            T_r = [np.dot(sigma, g.face_normals[:, f]) for f in faces]
            return np.vstack(T_r).T

        u_bound = np.zeros((3, g.num_faces))

        sgn_n = pp.numerics.fracture_deformation.sign_of_faces(g, neu_ind)
        sgn_r = pp.numerics.fracture_deformation.sign_of_faces(g, rob_ind)

        u_bound[:, dir_ind] = u_ex(g.face_centers[:, dir_ind])
        u_bound[:, neu_ind] = T_ex(neu_ind) * sgn_n
        u_bound[:, rob_ind] = (
            T_ex(rob_ind) * sgn_r
            + robin_weight * u_ex(g.face_centers[:, rob_ind]) * g.face_areas[rob_ind]
        )
        u, T = self.solve_mpsa(g, c, robin_weight, bnd, u_bound)

        assert np.allclose(u, u_ex(g.cell_centers).ravel("F"))
        assert np.allclose(T, T_ex(np.arange(g.num_faces)).ravel("F"))

    def solve_mpsa(self, g, c, robin_weight, bnd, u_bound):
        stress, bound_stress = pp.numerics.fv.mpsa._mpsa_local(
            g, c, bnd, robin_weight=robin_weight
        )
        div = pp.fvutils.vector_divergence(g)
        a = div * stress
        b = -div * bound_stress * u_bound.ravel("F")

        u = np.linalg.solve(a.A, b)
        T = stress * u + bound_stress * u_bound.ravel("F")
        return u, T
