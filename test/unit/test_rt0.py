""" Module contains unit tests for RT0, and a separate set of tests for RT0 with
gravitational forces.
"""
import unittest

import numpy as np
import scipy.sparse as sps

import porepy as pp


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#
    def _matrix(self, g, perm, bc):
        solver = pp.RT0(keyword="flow")

        data = pp.initialize_default_data(
            g, {}, "flow", {"second_order_tensor": perm, "bc": bc}
        )
        solver.discretize(g, data)
        M = solver.assemble_matrix(g, data).todense()
        P = data[pp.DISCRETIZATION_MATRICES]["flow"][solver.vector_proj_key].todense()
        return M, P

    def test_rt0_1d_iso(self):
        g = pp.CartGrid(3, 1)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx, kyy=1, kzz=1)
        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])

        M, P = self._matrix(g, perm, bc)
        M_known = np.matrix(
            [
                [0.11111111, 0.05555556, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.05555556, 0.22222222, 0.05555556, 0.0, -1.0, 1.0, 0.0],
                [0.0, 0.05555556, 0.22222222, 0.05555556, 0.0, -1.0, 1.0],
                [0.0, 0.0, 0.05555556, 0.11111111, 0.0, 0.0, -1.0],
                [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
            ]
        )

        self.assertTrue(np.allclose(M, M.T))
        self.assertTrue(np.allclose(M, M_known))

        P_known = np.matrix(
            [
                [5.0000e-01, 5.0000e-01, 0.0000e00, 0.0000e00],
                [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                [-5.5511e-17, -5.5511e-17, 0.0000e00, 0.0000e00],
                [0.0000e00, 5.0000e-01, 5.0000e-01, 0.0000e00],
                [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                [0.0000e00, -5.5511e-17, -5.5511e-17, 0.0000e00],
                [0.0000e00, 0.0000e00, 5.0000e-01, 5.0000e-01],
                [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                [0.0000e00, 0.0000e00, -5.5511e-17, -5.5511e-17],
            ]
        )

        self.assertTrue(np.allclose(P, P_known))

    # ------------------------------------------------------------------------------#

    def test_rt0_1d_ani(self):
        g = pp.CartGrid(3, 1)
        g.compute_geometry()

        kxx = 1.0 / (np.sin(g.cell_centers[0, :]) + 1)
        perm = pp.SecondOrderTensor(kxx, kyy=1, kzz=1)
        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        M, P = self._matrix(g, perm, bc)

        M_known = np.matrix(
            [
                [0.12954401, 0.06477201, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.06477201, 0.29392463, 0.08219031, 0.0, -1.0, 1.0, 0.0],
                [0.0, 0.08219031, 0.3577336, 0.09667649, 0.0, -1.0, 1.0],
                [0.0, 0.0, 0.09667649, 0.19335298, 0.0, 0.0, -1.0],
                [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
            ]
        )

        self.assertTrue(np.allclose(M, M.T))
        self.assertTrue(np.allclose(M, M_known))

        P_known = np.matrix(
            [
                [5.0000e-01, 5.0000e-01, 0.0000e00, 0.0000e00],
                [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                [-5.5511e-17, -5.5511e-17, 0.0000e00, 0.0000e00],
                [0.0000e00, 5.0000e-01, 5.0000e-01, 0.0000e00],
                [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                [0.0000e00, -5.5511e-17, -5.5511e-17, 0.0000e00],
                [0.0000e00, 0.0000e00, 5.0000e-01, 5.0000e-01],
                [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                [0.0000e00, 0.0000e00, -5.5511e-17, -5.5511e-17],
            ]
        )

        self.assertTrue(np.allclose(P, P_known))

    # ------------------------------------------------------------------------------#

    def test_rt0_2d_triangle(self):

        dim = 2
        nodes = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

        indices = [0, 1, 1, 2, 2, 0]
        indptr = [0, 2, 4, 6]
        face_nodes = sps.csc_matrix(([True] * 6, indices, indptr))
        cell_faces = sps.csc_matrix([[1], [1], [1]])
        name = "test"

        g = pp.Grid(dim, nodes, face_nodes, cell_faces, name)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        M, P = self._matrix(g, perm, bc)

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [0.33333333, 0.0, -0.16666667, -1.0],
                [0.0, 0.16666667, 0.0, -1.0],
                [-0.16666667, 0.0, 0.33333333, -1.0],
                [-1.0, -1.0, -1.0, 0.0],
            ]
        )

        self.assertTrue(np.allclose(M, M.T))
        self.assertTrue(np.allclose(M, M_known))

        P_known = np.matrix(
            [
                [0.33333333, 0.33333333, -0.66666667],
                [-0.66666667, 0.33333333, 0.33333333],
                [0.0, 0.0, 0.0],
            ]
        )

        self.assertTrue(np.allclose(P, P_known))

    # ------------------------------------------------------------------------------#

    def test_rt0_2d_iso_simplex(self):
        g = pp.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        M, P = self._matrix(g, perm, bc)

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [0.33333333, 0.0, 0.0, -0.16666667, 0.0, -1.0, 0.0],
                [0.0, 0.33333333, 0.0, 0.0, -0.16666667, 0.0, -1.0],
                [0.0, 0.0, 0.33333333, 0.0, 0.0, -1.0, 1.0],
                [-0.16666667, 0.0, 0.0, 0.33333333, 0.0, -1.0, 0.0],
                [0.0, -0.16666667, 0.0, 0.0, 0.33333333, 0.0, -1.0],
                [-1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            ]
        )

        self.assertTrue(np.allclose(M, M.T))
        self.assertTrue(np.allclose(M, M_known))

        P_known = np.matrix(
            [
                [-0.33333333, 0.0, -0.33333333, 0.66666667, 0.0],
                [-0.66666667, 0.0, 0.33333333, 0.33333333, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -0.66666667, -0.33333333, 0.0, 0.33333333],
                [0.0, -0.33333333, 0.33333333, 0.0, 0.66666667],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        self.assertTrue(np.allclose(P, P_known))

    # ------------------------------------------------------------------------------#

    def test_rt0_2d_iso_simplex_mixed_bc(self):
        g = pp.StructuredTriangleGrid([2, 2], [1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)

        tol = 1e-6
        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bf_centers = g.face_centers[:, bf]
        left = bf_centers[0, :] < 0 + tol
        right = bf_centers[0, :] > 1 - tol

        labels = np.array(["neu"] * bf.size)
        labels[left] = "rob"
        labels[right] = "dir"

        bc = pp.BoundaryCondition(g, bf, labels)
        bc.robin_weight[bf[left]] = 2

        bc_val = np.zeros(g.num_faces)
        bc_val[bf[left]] = 3

        solver = pp.RT0(keyword="flow")

        specified_parameters = {
            "bc": bc,
            "second_order_tensor": perm,
            "bc_values": bc_val,
        }
        data = pp.initialize_default_data(g, {}, "flow", specified_parameters)
        solver.discretize(g, data)
        M, rhs = solver.assemble_matrix_rhs(g, data)
        up = sps.linalg.spsolve(M, rhs)

        p = solver.extract_pressure(g, up, data)
        u = solver.extract_flux(g, up, data)
        P0u = solver.project_flux(g, u, data)

        p_ex = 1 - g.cell_centers[0, :]
        P0u_ex = np.vstack(
            (np.ones(g.num_cells), np.zeros(g.num_cells), np.zeros(g.num_cells))
        )

        self.assertTrue(np.allclose(p, p_ex))
        self.assertTrue(np.allclose(P0u, P0u_ex))

        # Matrix computed with an already validated code
        M_known = (
            1.0
            / 6.0
            * np.matrix(
                [
                    [
                        6,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        8,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -6,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        2,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -6,
                        6,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        6,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        -1,
                        0,
                        0,
                        0,
                        4,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -6,
                        0,
                        0,
                        6,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        2,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -6,
                        6,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        -1,
                        0,
                        0,
                        2,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -6,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        -1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        4,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -6,
                        0,
                        0,
                        6,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        8,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -6,
                        0,
                        0,
                    ],
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
                        2,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -6,
                        6,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        4,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -6,
                        0,
                        0,
                        6,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        4,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        -6,
                        0,
                        0,
                        6,
                    ],
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
                        0,
                        0,
                        0,
                        2,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -6,
                        6,
                    ],
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
                        0,
                        1,
                        0,
                        0,
                        2,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -6,
                        0,
                    ],
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
                        0,
                        0,
                        0,
                        0,
                        0,
                        6,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
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
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        6,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        -6,
                        0,
                        -6,
                        0,
                        -6,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        -6,
                        6,
                        0,
                        0,
                        0,
                        0,
                        -6,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        -6,
                        0,
                        -6,
                        -6,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        6,
                        6,
                        0,
                        0,
                        0,
                        0,
                        -6,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        6,
                        0,
                        -6,
                        0,
                        -6,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -6,
                        6,
                        0,
                        0,
                        0,
                        0,
                        -6,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
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
                        0,
                        6,
                        0,
                        -6,
                        -6,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
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
                        0,
                        0,
                        6,
                        6,
                        0,
                        0,
                        -6,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                ]
            )
        )

        self.assertTrue(np.allclose(M.todense(), M_known))

    # ------------------------------------------------------------------------------#

    def test_rt0_2d_ani_simplex(self):
        g = pp.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        al = np.square(g.cell_centers[1, :]) + np.square(g.cell_centers[0, :]) + 1
        kxx = (np.square(g.cell_centers[0, :]) + 1) / al
        kyy = (np.square(g.cell_centers[1, :]) + 1) / al
        kxy = np.multiply(g.cell_centers[0, :], g.cell_centers[1, :]) / al

        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kxy=kxy, kzz=1)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        M, P = self._matrix(g, perm, bc)

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [0.39814815, 0.0, -0.0462963, -0.15740741, 0.0, -1.0, 0.0],
                [0.0, 0.39814815, 0.0462963, 0.0, -0.15740741, 0.0, -1.0],
                [-0.0462963, 0.0462963, 0.46296296, 0.00925926, -0.00925926, -1.0, 1.0],
                [-0.15740741, 0.0, 0.00925926, 0.34259259, 0.0, -1.0, 0.0],
                [0.0, -0.15740741, -0.00925926, 0.0, 0.34259259, 0.0, -1.0],
                [-1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            ]
        )

        self.assertTrue(np.allclose(M, M.T))
        self.assertTrue(np.allclose(M, M_known))

        P_known = np.matrix(
            [
                [-0.33333333, 0.0, -0.33333333, 0.66666667, 0.0],
                [-0.66666667, 0.0, 0.33333333, 0.33333333, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -0.66666667, -0.33333333, 0.0, 0.33333333],
                [0.0, -0.33333333, 0.33333333, 0.0, 0.66666667],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        self.assertTrue(np.allclose(P, P_known))

    # ------------------------------------------------------------------------------#

    def test_rt0_tetra(self):

        dim = 3
        nodes = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        indices = [0, 2, 1, 1, 2, 3, 2, 0, 3, 3, 0, 1]
        indptr = [0, 3, 6, 9, 12]
        face_nodes = sps.csc_matrix(([True] * 12, indices, indptr))
        cell_faces = sps.csc_matrix([[1], [1], [1], [1]])
        name = "test"

        g = pp.Grid(dim, nodes, face_nodes, cell_faces, name)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=kxx)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        M, P = self._matrix(g, perm, bc)
        M_known = np.matrix(
            [
                [0.53333333, 0.03333333, -0.13333333, -0.13333333, -1.0],
                [0.03333333, 0.2, 0.03333333, 0.03333333, -1.0],
                [-0.13333333, 0.03333333, 0.53333333, -0.13333333, -1.0],
                [-0.13333333, 0.03333333, -0.13333333, 0.53333333, -1.0],
                [-1.0, -1.0, -1.0, -1.0, 0.0],
            ]
        )

        self.assertTrue(np.allclose(M, M.T))
        self.assertTrue(np.allclose(M, M_known))

        P_known = np.matrix(
            [[0.5, 0.5, -1.5, 0.5], [0.5, 0.5, 0.5, -1.5], [-1.5, 0.5, 0.5, 0.5]]
        )

        self.assertTrue(np.allclose(P, P_known))

    # ------------------------------------------------------------------------------#

    def test_rt0_3d(self):

        g = pp.StructuredTetrahedralGrid([1, 1, 1], [1, 1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=kxx)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        M, P = self._matrix(g, perm, bc)
        M_known, P_known = matrix_for_test_rt0_3d()

        self.assertTrue(np.allclose(M, M.T))
        self.assertTrue(np.allclose(M, M_known))
        self.assertTrue(np.allclose(P, P_known))

    # ------------------------------------------------------------------------------#

    def test_dual_rt0_1d_iso_line(self):
        g = pp.CartGrid(3, 1)
        R = pp.map_geometry.rotation_matrix(np.pi / 6.0, [0, 0, 1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx, kyy=1, kzz=1)
        perm.rotate(R)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        M, P = self._matrix(g, perm, bc)

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [0.11111111, 0.05555556, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.05555556, 0.22222222, 0.05555556, 0.0, -1.0, 1.0, 0.0],
                [0.0, 0.05555556, 0.22222222, 0.05555556, 0.0, -1.0, 1.0],
                [0.0, 0.0, 0.05555556, 0.11111111, 0.0, 0.0, -1.0],
                [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
            ]
        )

        self.assertTrue(np.allclose(M, M.T))
        self.assertTrue(np.allclose(M, M_known))

        P_known = np.matrix(
            [
                [4.33012702e-01, 4.33012702e-01, 0.00000000e00, 0.00000000e00],
                [2.50000000e-01, 2.50000000e-01, 0.00000000e00, 0.00000000e00],
                [5.55111512e-17, 5.55111512e-17, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 4.33012702e-01, 4.33012702e-01, 0.00000000e00],
                [0.00000000e00, 2.50000000e-01, 2.50000000e-01, 0.00000000e00],
                [0.00000000e00, 5.55111512e-17, 5.55111512e-17, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 4.33012702e-01, 4.33012702e-01],
                [0.00000000e00, 0.00000000e00, 2.50000000e-01, 2.50000000e-01],
                [0.00000000e00, 0.00000000e00, 5.55111512e-17, 5.55111512e-17],
            ]
        )

        self.assertTrue(np.allclose(P, P_known))

    # ------------------------------------------------------------------------------#

    def test_rt0_2d_iso_simplex_surf(self):
        g = pp.StructuredTriangleGrid([1, 1], [1, 1])
        R = pp.map_geometry.rotation_matrix(-np.pi / 4.0, [1, 1, -1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
        perm.rotate(R)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        M, P = self._matrix(g, perm, bc)

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [0.33333333, 0.0, 0.0, -0.16666667, 0.0, -1.0, 0.0],
                [0.0, 0.33333333, 0.0, 0.0, -0.16666667, 0.0, -1.0],
                [0.0, 0.0, 0.33333333, 0.0, 0.0, -1.0, 1.0],
                [-0.16666667, 0.0, 0.0, 0.33333333, 0.0, -1.0, 0.0],
                [0.0, -0.16666667, 0.0, 0.0, 0.33333333, 0.0, -1.0],
                [-1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            ]
        )

        self.assertTrue(np.allclose(M, M.T))
        self.assertTrue(np.allclose(M, M_known))

        P_known = np.matrix(
            [
                [-0.06116781, 0.0, -0.37178502, 0.43295283, 0.0],
                [-0.70511836, 0.0, 0.0996195, 0.60549886, 0.0],
                [0.23371384, 0.0, -0.27216553, 0.03845169, 0.0],
                [0.0, -0.43295283, -0.37178502, 0.0, 0.06116781],
                [0.0, -0.60549886, 0.0996195, 0.0, 0.70511836],
                [0.0, -0.03845169, -0.27216553, 0.0, -0.23371384],
            ]
        )

        self.assertTrue(np.allclose(P, P_known))

    # ------------------------------------------------------------------------------#

    def test_rt0_2d_ani_simplex_surf(self):
        g = pp.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        al = np.square(g.cell_centers[1, :]) + np.square(g.cell_centers[0, :]) + 1
        kxx = (np.square(g.cell_centers[0, :]) + 1) / al
        kyy = (np.square(g.cell_centers[1, :]) + 1) / al
        kxy = np.multiply(g.cell_centers[0, :], g.cell_centers[1, :]) / al

        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kxy=kxy, kzz=1)

        R = pp.map_geometry.rotation_matrix(np.pi / 3.0, [1, 1, 0])
        perm.rotate(R)
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        M, P = self._matrix(g, perm, bc)

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [0.39814815, 0.0, -0.0462963, -0.15740741, 0.0, -1.0, 0.0],
                [0.0, 0.39814815, 0.0462963, 0.0, -0.15740741, 0.0, -1.0],
                [-0.0462963, 0.0462963, 0.46296296, 0.00925926, -0.00925926, -1.0, 1.0],
                [-0.15740741, 0.0, 0.00925926, 0.34259259, 0.0, -1.0, 0.0],
                [0.0, -0.15740741, -0.00925926, 0.0, 0.34259259, 0.0, -1.0],
                [-1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            ]
        )

        self.assertTrue(np.allclose(M, M.T))
        self.assertTrue(np.allclose(M, M_known))

        P_known = np.matrix(
            [
                [-0.41666667, 0.0, -0.16666667, 0.58333333, 0.0],
                [-0.58333333, 0.0, 0.16666667, 0.41666667, 0.0],
                [-0.20412415, 0.0, 0.40824829, -0.20412415, 0.0],
                [0.0, -0.58333333, -0.16666667, 0.0, 0.41666667],
                [0.0, -0.41666667, 0.16666667, 0.0, 0.58333333],
                [0.0, 0.20412415, 0.40824829, 0.0, 0.20412415],
            ]
        )

        self.assertTrue(np.allclose(P, P_known))

    # ------------------------------------------------------------------------------#

    def test_convergence_rt0_2d_iso_simplex_exact(self):

        p_ex = lambda pt: 2 * pt[0, :] - 3 * pt[1, :] - 9
        u_ex = np.array([-2, 3, 0])

        for i in np.arange(5):
            g = pp.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            g.compute_geometry()

            kxx = np.ones(g.num_cells)
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
            bf = g.get_boundary_faces()
            bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
            bc_val = np.zeros(g.num_faces)
            bc_val[bf] = p_ex(g.face_centers[:, bf])

            solver = pp.RT0(keyword="flow")

            specified_parameters = {
                "bc": bc,
                "bc_values": bc_val,
                "second_order_tensor": perm,
            }
            data = pp.initialize_default_data(g, {}, "flow", specified_parameters)

            solver.discretize(g, data)
            M, rhs = solver.assemble_matrix_rhs(g, data)
            up = sps.linalg.spsolve(M, rhs)
            p = solver.extract_pressure(g, up, data)
            err = np.sum(np.abs(p - p_ex(g.cell_centers)))

            self.assertTrue(np.isclose(err, 0))

            P = data[pp.DISCRETIZATION_MATRICES]["flow"][solver.vector_proj_key]
            u = solver.extract_flux(g, up, data)
            P0u = solver.project_flux(g, u, data)
            err = np.sum(
                np.abs(P0u - np.tile(u_ex, g.num_cells).reshape((3, -1), order="F"))
            )

            self.assertTrue(np.isclose(err, 0))

    # ------------------------------------------------------------------------------#

    def test_convergence_rt0_2d_iso_simplex(self):

        a = 8 * np.pi ** 2
        rhs_ex = lambda pt: np.multiply(
            np.sin(2 * np.pi * pt[0, :]), np.sin(2 * np.pi * pt[1, :])
        )
        p_ex = lambda pt: rhs_ex(pt) / a
        u_ex_0 = (
            lambda pt: np.multiply(
                -np.cos(2 * np.pi * pt[0, :]), np.sin(2 * np.pi * pt[1, :])
            )
            * 2
            * np.pi
            / a
        )
        u_ex_1 = (
            lambda pt: np.multiply(
                -np.sin(2 * np.pi * pt[0, :]), np.cos(2 * np.pi * pt[1, :])
            )
            * 2
            * np.pi
            / a
        )

        p_errs_known = np.array(
            [
                0.00128247705764,
                0.000770088925769,
                0.00050939369071,
                0.000360006145403,
                0.000267209318912,
            ]
        )

        u_errs_known = np.array(
            [
                0.024425617686195795,
                0.016806807988931593,
                0.012859109258624914,
                0.01044523811171081,
                0.008811844361691244,
            ]
        )

        for i, p_err_known, u_err_known in zip(
            np.arange(5), p_errs_known, u_errs_known
        ):
            g = pp.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            g.compute_geometry()

            kxx = np.ones(g.num_cells)
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
            bf = g.get_boundary_faces()
            bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
            bc_val = np.zeros(g.num_faces)
            bc_val[bf] = p_ex(g.face_centers[:, bf])
            # Minus sign to move to rhs
            source = np.multiply(g.cell_volumes, rhs_ex(g.cell_centers))

            solver = pp.RT0(keyword="flow")
            solver_rhs = pp.DualScalarSource(keyword="flow")

            specified_parameters = {
                "bc": bc,
                "bc_values": bc_val,
                "second_order_tensor": perm,
                "source": source,
            }
            data = pp.initialize_default_data(g, {}, "flow", specified_parameters)

            solver.discretize(g, data)
            solver_rhs.discretize(g, data)

            M, rhs_bc = solver.assemble_matrix_rhs(g, data)
            _, rhs = solver_rhs.assemble_matrix_rhs(g, data)

            up = sps.linalg.spsolve(M, rhs_bc + rhs)
            p = solver.extract_pressure(g, up, data)
            err = np.sqrt(
                np.sum(
                    np.multiply(g.cell_volumes, np.power(p - p_ex(g.cell_centers), 2))
                )
            )
            self.assertTrue(np.isclose(err, p_err_known))

            P = data[pp.DISCRETIZATION_MATRICES]["flow"][solver.vector_proj_key]
            u = solver.extract_flux(g, up, data)
            P0u = solver.project_flux(g, u, data)
            uu_ex_0 = u_ex_0(g.cell_centers)
            uu_ex_1 = u_ex_1(g.cell_centers)
            uu_ex_2 = np.zeros(g.num_cells)
            uu_ex = np.vstack((uu_ex_0, uu_ex_1, uu_ex_2))
            err = np.sqrt(
                np.sum(
                    np.multiply(
                        g.cell_volumes, np.sum(np.power(P0u - uu_ex, 2), axis=0)
                    )
                )
            )
            self.assertTrue(np.isclose(err, u_err_known))

    # ------------------------------------------------------------------------------#

    def test_convergence_rt0_2d_ani_simplex(self):

        rhs_ex = lambda pt: 14
        p_ex = (
            lambda pt: 2 * np.power(pt[0, :], 2)
            - 6 * np.power(pt[1, :], 2)
            + np.multiply(pt[0, :], pt[1, :])
        )
        u_ex_0 = lambda pt: -9 * pt[0, :] + 10 * pt[1, :]
        u_ex_1 = lambda pt: -6 * pt[0, :] + 23 * pt[1, :]

        p_errs_known = np.array(
            [
                0.014848639601,
                0.00928479234915,
                0.00625096095775,
                0.00446722560521,
                0.00334170283883,
            ]
        )
        u_errs_known = np.array(
            [
                1.7264059760345325,
                1.34164231163404,
                1.0925566034251666,
                0.9198698104736417,
                0.7936243780450762,
            ]
        )

        for i, p_err_known, u_err_known in zip(
            np.arange(5), p_errs_known, u_errs_known
        ):
            g = pp.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            g.compute_geometry()

            kxx = 2 * np.ones(g.num_cells)
            kxy = np.ones(g.num_cells)
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kxy=kxy, kzz=1)
            bf = g.get_boundary_faces()
            bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
            bc_val = np.zeros(g.num_faces)
            bc_val[bf] = p_ex(g.face_centers[:, bf])
            # Minus sign to move to rhs
            source = np.multiply(g.cell_volumes, rhs_ex(g.cell_centers))

            solver = pp.RT0(keyword="flow")
            solver_rhs = pp.DualScalarSource(keyword="flow")

            specified_parameters = {
                "bc": bc,
                "bc_values": bc_val,
                "second_order_tensor": perm,
                "source": source,
            }
            data = pp.initialize_default_data(g, {}, "flow", specified_parameters)

            solver.discretize(g, data)
            solver_rhs.discretize(g, data)
            M, rhs_bc = solver.assemble_matrix_rhs(g, data)
            _, rhs = solver_rhs.assemble_matrix_rhs(g, data)

            up = sps.linalg.spsolve(M, rhs_bc + rhs)
            p = solver.extract_pressure(g, up, data)
            err = np.sqrt(
                np.sum(
                    np.multiply(g.cell_volumes, np.power(p - p_ex(g.cell_centers), 2))
                )
            )
            self.assertTrue(np.isclose(err, p_err_known))

            P = data[pp.DISCRETIZATION_MATRICES]["flow"][solver.vector_proj_key]
            u = solver.extract_flux(g, up, data)
            P0u = solver.project_flux(g, u, data)
            uu_ex_0 = u_ex_0(g.cell_centers)
            uu_ex_1 = u_ex_1(g.cell_centers)
            uu_ex_2 = np.zeros(g.num_cells)
            uu_ex = np.vstack((uu_ex_0, uu_ex_1, uu_ex_2))
            err = np.sqrt(
                np.sum(
                    np.multiply(
                        g.cell_volumes, np.sum(np.power(P0u - uu_ex, 2), axis=0)
                    )
                )
            )
            self.assertTrue(np.isclose(err, u_err_known))


class TestRT0Gravity(unittest.TestCase):

    # ------------------------------------------------------------------------------#
    def _matrix(self, g, perm, bc, vect):
        solver = pp.RT0(keyword="flow")

        data = pp.initialize_default_data(
            g,
            {},
            "flow",
            {"second_order_tensor": perm, "bc": bc, "vector_source": vect},
        )
        solver.discretize(g, data)
        return solver.assemble_matrix_rhs(g, data)[1]

    # ------------------------------------------------------------------------------#

    def test_rt0_1d(self):
        g = pp.CartGrid(3, 1)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx, kyy=1, kzz=1)
        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        vect = np.vstack(
            (g.cell_volumes, np.zeros(g.num_cells), np.zeros(g.num_cells))
        ).ravel(order="F")

        b = self._matrix(g, perm, bc, vect)
        b_known = np.array(
            [0.16666667, 0.33333333, 0.33333333, 0.16666667, 0.0, 0.0, 0.0]
        )

        self.assertTrue(np.allclose(b, b_known))

    # ------------------------------------------------------------------------------#

    def test_rt0_2d_triangle(self):

        dim = 2
        nodes = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

        indices = [0, 1, 1, 2, 2, 0]
        indptr = [0, 2, 4, 6]
        face_nodes = sps.csc_matrix(([True] * 6, indices, indptr))
        cell_faces = sps.csc_matrix([[1], [1], [1]])
        name = "test"

        g = pp.Grid(dim, nodes, face_nodes, cell_faces, name)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        vect = np.vstack(
            (g.cell_volumes, 3 * g.cell_volumes, np.zeros(g.num_cells))
        ).ravel(order="F")

        b = self._matrix(g, perm, bc, vect)
        b_known = np.array([-0.83333333, 0.66666667, 0.16666667, 0.0])

        self.assertTrue(np.allclose(b, b_known))

    # ------------------------------------------------------------------------------#

    def test_rt0_2d_simplex(self):
        g = pp.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        vect = np.vstack(
            (2 * g.cell_volumes, 3 * g.cell_volumes, np.zeros(g.num_cells))
        ).ravel(order="F")
        b = self._matrix(g, perm, bc, vect)

        b_known = np.array(
            [-1.33333333, -1.16666667, 0.33333333, 1.16666667, 1.33333333, 0.0, 0.0]
        )

        self.assertTrue(np.allclose(b, b_known))

    # ------------------------------------------------------------------------------#

    def test_rt0_tetra(self):

        dim = 3
        nodes = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        indices = [0, 2, 1, 1, 2, 3, 2, 0, 3, 3, 0, 1]
        indptr = [0, 3, 6, 9, 12]
        face_nodes = sps.csc_matrix(([True] * 12, indices, indptr))
        cell_faces = sps.csc_matrix([[1], [1], [1], [1]])
        name = "test"

        g = pp.Grid(dim, nodes, face_nodes, cell_faces, name)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=kxx)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        vect = np.vstack(
            (2 * g.cell_volumes, 3 * g.cell_volumes, np.zeros(g.num_cells))
        ).ravel(order="F")

        b = self._matrix(g, perm, bc, vect)
        b_known = np.array([0.41666667, 0.41666667, -0.25, -0.58333333, 0.0])

        self.assertTrue(np.allclose(b, b_known))

    # ------------------------------------------------------------------------------#

    def test_convergence_rt0_2d_iso_simplex_exact(self):

        p_ex = lambda pt: 2 * pt[0, :] - 3 * pt[1, :] - 9
        u_ex = np.array([-1, 4, 0])

        for i in np.arange(5):
            g = pp.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            g.compute_geometry()

            kxx = np.ones(g.num_cells)
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
            bf = g.get_boundary_faces()
            bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
            bc_val = np.zeros(g.num_faces)
            bc_val[bf] = p_ex(g.face_centers[:, bf])
            vect = np.vstack(
                (g.cell_volumes, g.cell_volumes, np.zeros(g.num_cells))
            ).ravel(order="F")

            solver = pp.RT0(keyword="flow")

            specified_parameters = {
                "bc": bc,
                "bc_values": bc_val,
                "second_order_tensor": perm,
                "vector_source": vect,
            }
            data = pp.initialize_default_data(g, {}, "flow", specified_parameters)

            solver.discretize(g, data)
            M, rhs = solver.assemble_matrix_rhs(g, data)
            up = sps.linalg.spsolve(M, rhs)
            p = solver.extract_pressure(g, up, data)
            err = np.sum(np.abs(p - p_ex(g.cell_centers)))

            self.assertTrue(np.isclose(err, 0))

            _ = data[pp.DISCRETIZATION_MATRICES]["flow"][solver.vector_proj_key]
            u = solver.extract_flux(g, up, data)
            P0u = solver.project_flux(g, u, data)
            err = np.sum(
                np.abs(P0u - np.tile(u_ex, g.num_cells).reshape((3, -1), order="F"))
            )

            self.assertTrue(np.isclose(err, 0))

    # ------------------------------------------------------------------------------#

    def test_rt0_3d(self):

        g = pp.StructuredTetrahedralGrid([1, 1, 1], [1, 1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=kxx)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        vect = np.vstack(
            (7 * g.cell_volumes, 4 * g.cell_volumes, 3 * g.cell_volumes)
        ).ravel(order="F")

        b = self._matrix(g, perm, bc, vect)
        b_known = np.array(
            [
                -0.16666667,
                -0.16666667,
                1.16666667,
                0.08333333,
                1.75,
                2.0,
                0.58333333,
                1.5,
                0.08333333,
                -1.5,
                -2.0,
                -0.08333333,
                -0.58333333,
                -1.75,
                -1.16666667,
                0.16666667,
                -0.08333333,
                0.16666667,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        self.assertTrue(np.allclose(b, b_known))

    # ------------------------------------------------------------------------------#

    def test_dual_rt0_1d_line(self):
        g = pp.CartGrid(3, 1)
        R = pp.map_geometry.rotation_matrix(np.pi / 6.0, [0, 0, 1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx, kyy=1, kzz=1)
        perm.rotate(R)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        vect = np.vstack(
            (g.cell_volumes, 0 * g.cell_volumes, 0 * g.cell_volumes)
        ).ravel(order="F")

        b = self._matrix(g, perm, bc, vect)
        b_known = np.array(
            [0.14433757, 0.28867513, 0.28867513, 0.14433757, 0.0, 0.0, 0.0]
        )

        self.assertTrue(np.allclose(b, b_known))

    # ------------------------------------------------------------------------------#

    def test_rt0_2d_simplex_surf(self):
        g = pp.StructuredTriangleGrid([1, 1], [1, 1])
        R = pp.map_geometry.rotation_matrix(-np.pi / 4.0, [1, 1, -1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
        perm.rotate(R)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        vect = np.vstack(
            (g.cell_volumes, 2 * g.cell_volumes, 0 * g.cell_volumes)
        ).ravel(order="F")

        b = self._matrix(g, perm, bc, vect)
        b_known = np.array(
            [-0.73570226, -0.82197528, -0.17254603, 0.82197528, 0.73570226, 0.0, 0.0]
        )

        self.assertTrue(np.allclose(b, b_known))

    # ------------------------------------------------------------------------------#

    def test_convergence_rt0_2d_iso_simplex(self):

        a = 8 * np.pi ** 2
        rhs_ex = lambda pt: np.multiply(
            np.sin(2 * np.pi * pt[0, :]), np.sin(2 * np.pi * pt[1, :])
        )
        p_ex = lambda pt: rhs_ex(pt) / a
        u_ex_0 = (
            lambda pt: np.multiply(
                -np.cos(2 * np.pi * pt[0, :]), np.sin(2 * np.pi * pt[1, :])
            )
            * 2
            * np.pi
            / a
            + 1
        )
        u_ex_1 = (
            lambda pt: np.multiply(
                -np.sin(2 * np.pi * pt[0, :]), np.cos(2 * np.pi * pt[1, :])
            )
            * 2
            * np.pi
            / a
        )

        p_errs_known = np.array(
            [
                0.00128247705764,
                0.000770088925769,
                0.00050939369071,
                0.000360006145403,
                0.000267209318912,
            ]
        )

        u_errs_known = np.array(
            [
                0.0244256176861958,
                0.016806807988931607,
                0.012859109258624908,
                0.010445238111710818,
                0.008811844361691242,
            ]
        )

        for i, p_err_known, u_err_known in zip(
            np.arange(5), p_errs_known, u_errs_known
        ):
            g = pp.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            g.compute_geometry()

            kxx = np.ones(g.num_cells)
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
            bf = g.get_boundary_faces()
            bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
            bc_val = np.zeros(g.num_faces)
            bc_val[bf] = p_ex(g.face_centers[:, bf])
            # Minus sign to move to rhs
            source = np.multiply(g.cell_volumes, rhs_ex(g.cell_centers))
            vect = np.vstack(
                (g.cell_volumes, np.zeros(g.num_cells), np.zeros(g.num_cells))
            ).ravel(order="F")

            solver = pp.RT0(keyword="flow")
            solver_rhs = pp.DualScalarSource(keyword="flow")

            specified_parameters = {
                "bc": bc,
                "bc_values": bc_val,
                "second_order_tensor": perm,
                "source": source,
                "vector_source": vect,
            }
            data = pp.initialize_default_data(g, {}, "flow", specified_parameters)

            solver.discretize(g, data)
            solver_rhs.discretize(g, data)

            M, rhs_bc = solver.assemble_matrix_rhs(g, data)
            _, rhs = solver_rhs.assemble_matrix_rhs(g, data)

            up = sps.linalg.spsolve(M, rhs_bc + rhs)
            p = solver.extract_pressure(g, up, data)
            err = np.sqrt(
                np.sum(
                    np.multiply(g.cell_volumes, np.power(p - p_ex(g.cell_centers), 2))
                )
            )
            self.assertTrue(np.isclose(err, p_err_known))

            _ = data[pp.DISCRETIZATION_MATRICES]["flow"][solver.vector_proj_key]
            u = solver.extract_flux(g, up, data)
            P0u = solver.project_flux(g, u, data)
            uu_ex_0 = u_ex_0(g.cell_centers)
            uu_ex_1 = u_ex_1(g.cell_centers)
            uu_ex_2 = np.zeros(g.num_cells)
            uu_ex = np.vstack((uu_ex_0, uu_ex_1, uu_ex_2))
            err = np.sqrt(
                np.sum(
                    np.multiply(
                        g.cell_volumes, np.sum(np.power(P0u - uu_ex, 2), axis=0)
                    )
                )
            )
            self.assertTrue(np.isclose(err, u_err_known))

    # ------------------------------------------------------------------------------#

    def test_convergence_rt0_2d_ani_simplex(self):

        rhs_ex = lambda pt: 14
        p_ex = (
            lambda pt: 2 * np.power(pt[0, :], 2)
            - 6 * np.power(pt[1, :], 2)
            + np.multiply(pt[0, :], pt[1, :])
        )
        u_ex_0 = lambda pt: -9 * pt[0, :] + 10 * pt[1, :] + 4
        u_ex_1 = lambda pt: -6 * pt[0, :] + 23 * pt[1, :] + 5

        p_errs_known = np.array(
            [
                0.014848639601,
                0.00928479234915,
                0.00625096095775,
                0.00446722560521,
                0.00334170283883,
            ]
        )
        u_errs_known = np.array(
            [
                1.7264059760345314,
                1.34164231163404,
                1.0925566034251668,
                0.9198698104736419,
                0.7936243780450764,
            ]
        )

        for i, p_err_known, u_err_known in zip(
            np.arange(5), p_errs_known, u_errs_known
        ):
            g = pp.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            g.compute_geometry()

            kxx = 2 * np.ones(g.num_cells)
            kxy = np.ones(g.num_cells)
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kxy=kxy, kzz=1)
            bf = g.get_boundary_faces()
            bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
            bc_val = np.zeros(g.num_faces)
            bc_val[bf] = p_ex(g.face_centers[:, bf])
            # Minus sign to move to rhs
            source = np.multiply(g.cell_volumes, rhs_ex(g.cell_centers))
            vect = np.vstack(
                (g.cell_volumes, 2 * g.cell_volumes, np.zeros(g.num_cells))
            ).ravel(order="F")

            solver = pp.RT0(keyword="flow")
            solver_rhs = pp.DualScalarSource(keyword="flow")

            specified_parameters = {
                "bc": bc,
                "bc_values": bc_val,
                "second_order_tensor": perm,
                "source": source,
                "vector_source": vect,
            }
            data = pp.initialize_default_data(g, {}, "flow", specified_parameters)

            solver.discretize(g, data)
            solver_rhs.discretize(g, data)
            M, rhs_bc = solver.assemble_matrix_rhs(g, data)
            _, rhs = solver_rhs.assemble_matrix_rhs(g, data)

            up = sps.linalg.spsolve(M, rhs_bc + rhs)
            p = solver.extract_pressure(g, up, data)
            err = np.sqrt(
                np.sum(
                    np.multiply(g.cell_volumes, np.power(p - p_ex(g.cell_centers), 2))
                )
            )
            self.assertTrue(np.isclose(err, p_err_known))

            _ = data[pp.DISCRETIZATION_MATRICES]["flow"][solver.vector_proj_key]
            u = solver.extract_flux(g, up, data)
            P0u = solver.project_flux(g, u, data)
            uu_ex_0 = u_ex_0(g.cell_centers)
            uu_ex_1 = u_ex_1(g.cell_centers)
            uu_ex_2 = np.zeros(g.num_cells)
            uu_ex = np.vstack((uu_ex_0, uu_ex_1, uu_ex_2))
            err = np.sqrt(
                np.sum(
                    np.multiply(
                        g.cell_volumes, np.sum(np.power(P0u - uu_ex, 2), axis=0)
                    )
                )
            )
            self.assertTrue(np.isclose(err, u_err_known))


def matrix_for_test_rt0_3d():
    return (
        np.matrix(
            [
                [
                    0.53333333,
                    0.13333333,
                    -0.13333333,
                    0.0,
                    -0.03333333,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.13333333,
                    0.53333333,
                    0.13333333,
                    0.0,
                    0.03333333,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    -0.13333333,
                    0.13333333,
                    0.53333333,
                    0.0,
                    -0.03333333,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.66666667,
                    0.0,
                    0.16666667,
                    0.0,
                    0.16666667,
                    0.0,
                    0.0,
                    0.0,
                    0.33333333,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                ],
                [
                    -0.03333333,
                    0.03333333,
                    -0.03333333,
                    0.0,
                    0.66666667,
                    -0.13333333,
                    0.0,
                    0.0,
                    0.0,
                    0.13333333,
                    0.0,
                    0.0,
                    0.36666667,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.16666667,
                    -0.13333333,
                    0.8,
                    0.0,
                    0.0,
                    0.0,
                    0.2,
                    0.0,
                    0.16666667,
                    -0.03333333,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.8,
                    0.03333333,
                    0.0,
                    0.0,
                    -0.03333333,
                    0.0,
                    0.0,
                    0.36666667,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.16666667,
                    0.0,
                    0.0,
                    0.03333333,
                    0.8,
                    0.0,
                    0.0,
                    0.2,
                    0.16666667,
                    0.0,
                    0.13333333,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.66666667,
                    0.16666667,
                    0.16666667,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.33333333,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.13333333,
                    0.2,
                    0.0,
                    0.0,
                    0.16666667,
                    0.8,
                    0.0,
                    0.0,
                    0.03333333,
                    0.0,
                    0.0,
                    0.0,
                    0.16666667,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.03333333,
                    0.2,
                    0.16666667,
                    0.0,
                    0.8,
                    0.0,
                    0.0,
                    -0.13333333,
                    0.0,
                    0.0,
                    0.16666667,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    -1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.33333333,
                    0.0,
                    0.16666667,
                    0.0,
                    0.16666667,
                    0.0,
                    0.0,
                    0.0,
                    0.66666667,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.36666667,
                    -0.03333333,
                    0.0,
                    0.0,
                    0.0,
                    0.03333333,
                    0.0,
                    0.0,
                    0.8,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.36666667,
                    0.13333333,
                    0.0,
                    0.0,
                    -0.13333333,
                    0.0,
                    0.0,
                    0.66666667,
                    -0.03333333,
                    0.03333333,
                    0.0,
                    -0.03333333,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.03333333,
                    0.53333333,
                    0.13333333,
                    0.0,
                    -0.13333333,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.03333333,
                    0.13333333,
                    0.53333333,
                    0.0,
                    0.13333333,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.33333333,
                    0.16666667,
                    0.16666667,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.66666667,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.03333333,
                    -0.13333333,
                    0.13333333,
                    0.0,
                    0.53333333,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                [
                    1.0,
                    -1.0,
                    1.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    1.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    1.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    1.0,
                    -1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        ),
        np.matrix(
            [
                [
                    -0.5,
                    0.5,
                    1.5,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    -0.5,
                    -1.5,
                    -0.5,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.5,
                    0.5,
                    -0.5,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    -0.5,
                    0.0,
                    0.0,
                    -1.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.5,
                    0.5,
                    -0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                    -0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.5,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.5,
                    0.0,
                    0.5,
                    0.0,
                    -0.5,
                    0.0,
                    0.0,
                    0.0,
                    -1.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    -1.5,
                    0.0,
                    -0.5,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    -0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.5,
                    0.5,
                    0.0,
                    0.0,
                    -0.5,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.5,
                    -1.5,
                    -0.5,
                    0.0,
                    0.5,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.5,
                    0.5,
                    1.5,
                    0.0,
                    0.5,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.5,
                    0.5,
                    -0.5,
                    0.0,
                    -1.5,
                ],
            ]
        ),
    )


if __name__ == "__main__":
    unittest.main()
