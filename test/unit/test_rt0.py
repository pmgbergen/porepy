import numpy as np
import scipy.sparse as sps
import unittest

import porepy as pp
from porepy import cg

# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#

    def test_rt0_1d_iso(self):
        g = pp.structured.CartGrid(3, 1)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(3, kxx, kyy=1, kzz=1)
        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])

        solver = pp.RT0(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()

        M_known = np.matrix(
            [
                [0.11111111, 0.05555556, 0., 0., 1., 0., 0.],
                [0.05555556, 0.22222222, 0.05555556, 0., -1., 1., 0.],
                [0., 0.05555556, 0.22222222, 0.05555556, 0., -1., 1.],
                [0., 0., 0.05555556, 0.11111111, 0., 0., -1.],
                [1., -1., 0., 0., 0., 0., 0.],
                [0., 1., -1., 0., 0., 0., 0.],
                [0., 0., 1., -1., 0., 0., 0.],
            ]
        )

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

    # ------------------------------------------------------------------------------#

    def test_rt0_1d_ani(self):
        g = pp.structured.CartGrid(3, 1)
        g.compute_geometry()

        kxx = 1. / (np.sin(g.cell_centers[0, :]) + 1)
        perm = pp.SecondOrderTensor(3, kxx, kyy=1, kzz=1)
        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        solver = pp.RT0(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()

        M_known = np.matrix(
            [
                [0.12954401, 0.06477201, 0., 0., 1., 0., 0.],
                [0.06477201, 0.29392463, 0.08219031, 0., -1., 1., 0.],
                [0., 0.08219031, 0.3577336, 0.09667649, 0., -1., 1.],
                [0., 0., 0.09667649, 0.19335298, 0., 0., -1.],
                [1., -1., 0., 0., 0., 0., 0.],
                [0., 1., -1., 0., 0., 0., 0.],
                [0., 0., 1., -1., 0., 0., 0.],
            ]
        )

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

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
        perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kxx, kzz=1)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        solver = pp.RT0(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [0.33333333, 0., -0.16666667, -1.],
                [0., 0.16666667, 0., -1.],
                [-0.16666667, 0., 0.33333333, -1.],
                [-1., -1., -1., 0.],
            ]
        )

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

    # ------------------------------------------------------------------------------#

    def test_rt0_2d_iso_simplex(self):
        g = pp.simplex.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kxx, kzz=1)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        solver = pp.RT0(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [0.33333333, 0., 0., -0.16666667, 0., -1., 0.],
                [0., 0.33333333, 0., 0., -0.16666667, 0., -1.],
                [0., 0., 0.33333333, 0., 0., -1., 1.],
                [-0.16666667, 0., 0., 0.33333333, 0., -1., 0.],
                [0., -0.16666667, 0., 0., 0.33333333, 0., -1.],
                [-1., 0., -1., -1., 0., 0., 0.],
                [0., -1., 1., 0., -1., 0., 0.],
            ]
        )

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

    # ------------------------------------------------------------------------------#

    def test_rt0_2d_ani_simplex(self):
        g = pp.simplex.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        al = np.square(g.cell_centers[1, :]) + np.square(g.cell_centers[0, :]) + 1
        kxx = (np.square(g.cell_centers[0, :]) + 1) / al
        kyy = (np.square(g.cell_centers[1, :]) + 1) / al
        kxy = np.multiply(g.cell_centers[0, :], g.cell_centers[1, :]) / al

        perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kyy, kxy=kxy, kzz=1)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        solver = pp.RT0(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [0.39814815, 0., -0.0462963, -0.15740741, 0., -1., 0.],
                [0., 0.39814815, 0.0462963, 0., -0.15740741, 0., -1.],
                [-0.0462963, 0.0462963, 0.46296296, 0.00925926, -0.00925926, -1., 1.],
                [-0.15740741, 0., 0.00925926, 0.34259259, 0., -1., 0.],
                [0., -0.15740741, -0.00925926, 0., 0.34259259, 0., -1.],
                [-1., 0., -1., -1., 0., 0., 0.],
                [0., -1., 1., 0., -1., 0., 0.],
            ]
        )

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

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
        perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kxx, kzz=kxx)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        solver = pp.RT0(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()
        M_known = np.matrix(
            [
                [0.53333333, 0.03333333, -0.13333333, -0.13333333, -1.],
                [0.03333333, 0.2, 0.03333333, 0.03333333, -1.],
                [-0.13333333, 0.03333333, 0.53333333, -0.13333333, -1.],
                [-0.13333333, 0.03333333, -0.13333333, 0.53333333, -1.],
                [-1., -1., -1., -1., 0.],
            ]
        )

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

    # ------------------------------------------------------------------------------#

    def test_rt0_3d(self):

        g = pp.simplex.StructuredTetrahedralGrid([1, 1, 1], [1, 1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kxx, kzz=kxx)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        solver = pp.RT0(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()
        M_known = matrix_for_test_rt0_3d()

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

    # ------------------------------------------------------------------------------#

    def test_dual_rt0_1d_iso_line(self):
        g = pp.structured.CartGrid(3, 1)
        R = cg.rot(np.pi / 6., [0, 0, 1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(3, kxx, kyy=1, kzz=1)
        perm.rotate(R)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        solver = pp.RT0(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [0.11111111, 0.05555556, 0., 0., 1., 0., 0.],
                [0.05555556, 0.22222222, 0.05555556, 0., -1., 1., 0.],
                [0., 0.05555556, 0.22222222, 0.05555556, 0., -1., 1.],
                [0., 0., 0.05555556, 0.11111111, 0., 0., -1.],
                [1., -1., 0., 0., 0., 0., 0.],
                [0., 1., -1., 0., 0., 0., 0.],
                [0., 0., 1., -1., 0., 0., 0.],
            ]
        )

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

    # ------------------------------------------------------------------------------#

    def test_rt0_2d_iso_simplex_surf(self):
        g = pp.simplex.StructuredTriangleGrid([1, 1], [1, 1])
        R = cg.rot(-np.pi / 4., [1, 1, -1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kxx, kzz=1)
        perm.rotate(R)

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        solver = pp.RT0(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [0.33333333, 0., 0., -0.16666667, 0., -1., 0.],
                [0., 0.33333333, 0., 0., -0.16666667, 0., -1.],
                [0., 0., 0.33333333, 0., 0., -1., 1.],
                [-0.16666667, 0., 0., 0.33333333, 0., -1., 0.],
                [0., -0.16666667, 0., 0., 0.33333333, 0., -1.],
                [-1., 0., -1., -1., 0., 0., 0.],
                [0., -1., 1., 0., -1., 0., 0.],
            ]
        )

        assert np.allclose(M, M.T)
        # We test only the mass-Hdiv part
        assert np.allclose(M, M_known)

    # ------------------------------------------------------------------------------#

    def test_rt0_2d_ani_simplex_surf(self):
        g = pp.simplex.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        al = np.square(g.cell_centers[1, :]) + np.square(g.cell_centers[0, :]) + 1
        kxx = (np.square(g.cell_centers[0, :]) + 1) / al
        kyy = (np.square(g.cell_centers[1, :]) + 1) / al
        kxy = np.multiply(g.cell_centers[0, :], g.cell_centers[1, :]) / al

        perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kyy, kxy=kxy, kzz=1)

        R = cg.rot(np.pi / 3., [1, 1, 0])
        perm.rotate(R)
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        bf = g.get_boundary_faces()
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        solver = pp.RT0(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [0.39814815, 0., -0.0462963, -0.15740741, 0., -1., 0.],
                [0., 0.39814815, 0.0462963, 0., -0.15740741, 0., -1.],
                [-0.0462963, 0.0462963, 0.46296296, 0.00925926, -0.00925926, -1., 1.],
                [-0.15740741, 0., 0.00925926, 0.34259259, 0., -1., 0.],
                [0., -0.15740741, -0.00925926, 0., 0.34259259, 0., -1.],
                [-1., 0., -1., -1., 0., 0., 0.],
                [0., -1., 1., 0., -1., 0., 0.],
            ]
        )

        assert np.allclose(M, M.T)
        # We test only the mass-Hdiv part
        assert np.allclose(M, M_known)

    # ------------------------------------------------------------------------------#

    def test_convergence_rt0_2d_iso_simplex_exact(self):

        p_ex = lambda pt: 2 * pt[0, :] - 3 * pt[1, :] - 9

        for i in np.arange(5):
            g = pp.simplex.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            g.compute_geometry()

            kxx = np.ones(g.num_cells)
            perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kxx, kzz=1)
            bf = g.get_boundary_faces()
            bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
            bc_val = np.zeros(g.num_faces)
            bc_val[bf] = p_ex(g.face_centers[:, bf])

            solver = pp.RT0(physics="flow")

            param = pp.Parameters(g)
            param.set_tensor(solver, perm)
            param.set_bc(solver, bc)
            param.set_bc_val(solver, bc_val)
            M, rhs = solver.matrix_rhs(g, {"param": param})

            up = sps.linalg.spsolve(M, rhs)
            p = solver.extract_p(g, up)
            err = np.sum(np.abs(p - p_ex(g.cell_centers)))

            assert np.isclose(err, 0)

    # ------------------------------------------------------------------------------#

    def test_convergence_rt0_2d_iso_simplex(self):

        a = 8 * np.pi ** 2
        rhs_ex = lambda pt: np.multiply(
            np.sin(2 * np.pi * pt[0, :]), np.sin(2 * np.pi * pt[1, :])
        )
        p_ex = lambda pt: rhs_ex(pt) / a

        errs_known = np.array(
            [
                0.00128247705764,
                0.000770088925769,
                0.00050939369071,
                0.000360006145403,
                0.000267209318912,
            ]
        )

        for i, err_known in zip(np.arange(5), errs_known):
            g = pp.simplex.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            g.compute_geometry()

            kxx = np.ones(g.num_cells)
            perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kxx, kzz=1)
            bf = g.get_boundary_faces()
            bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
            bc_val = np.zeros(g.num_faces)
            bc_val[bf] = p_ex(g.face_centers[:, bf])
            source = np.multiply(g.cell_volumes, rhs_ex(g.cell_centers))

            solver = pp.RT0(physics="flow")
            solver_rhs = pp.DualSource(physics="flow")

            param = pp.Parameters(g)
            param.set_tensor("flow", perm)
            param.set_bc("flow", bc)
            param.set_bc_val("flow", bc_val)
            param.set_source("flow", source)

            M, rhs_bc = solver.matrix_rhs(g, {"param": param})
            _, rhs = solver_rhs.matrix_rhs(g, {"param": param})

            up = sps.linalg.spsolve(M, rhs_bc + rhs)
            p = solver.extract_p(g, up)
            err = np.sqrt(
                np.sum(
                    np.multiply(g.cell_volumes, np.power(p - p_ex(g.cell_centers), 2))
                )
            )
            assert np.isclose(err, err_known)

    # ------------------------------------------------------------------------------#

    def test_convergence_rt0_2d_ani_simplex(self):

        rhs_ex = lambda pt: 14
        p_ex = (
            lambda pt: 2 * np.power(pt[0, :], 2)
            - 6 * np.power(pt[1, :], 2)
            + np.multiply(pt[0, :], pt[1, :])
        )

        errs_known = np.array(
            [
                0.014848639601,
                0.00928479234915,
                0.00625096095775,
                0.00446722560521,
                0.00334170283883,
            ]
        )

        for i, err_known in zip(np.arange(5), errs_known):
            g = pp.simplex.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            g.compute_geometry()

            kxx = 2 * np.ones(g.num_cells)
            kxy = np.ones(g.num_cells)
            perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kxx, kxy=kxy, kzz=1)
            bf = g.get_boundary_faces()
            bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
            bc_val = np.zeros(g.num_faces)
            bc_val[bf] = p_ex(g.face_centers[:, bf])
            source = np.multiply(g.cell_volumes, rhs_ex(g.cell_centers))

            solver = pp.RT0(physics="flow")
            solver_rhs = pp.DualSource(physics="flow")

            param = pp.Parameters(g)
            param.set_tensor("flow", perm)
            param.set_bc("flow", bc)
            param.set_bc_val("flow", bc_val)
            param.set_source("flow", source)

            M, rhs_bc = solver.matrix_rhs(g, {"param": param})
            _, rhs = solver_rhs.matrix_rhs(g, {"param": param})

            up = sps.linalg.spsolve(M, rhs_bc + rhs)
            p = solver.extract_p(g, up)
            err = np.sqrt(
                np.sum(
                    np.multiply(g.cell_volumes, np.power(p - p_ex(g.cell_centers), 2))
                )
            )
            assert np.isclose(err, err_known)


# ------------------------------------------------------------------------------#


def matrix_for_test_rt0_3d():
    return np.matrix(
        [
            [
                0.53333333,
                0.13333333,
                -0.13333333,
                0.,
                -0.03333333,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                1.,
                0.,
                0.,
                0.,
                0.,
                0.,
            ],
            [
                0.13333333,
                0.53333333,
                0.13333333,
                0.,
                0.03333333,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -1.,
                0.,
                0.,
                0.,
                0.,
                0.,
            ],
            [
                -0.13333333,
                0.13333333,
                0.53333333,
                0.,
                -0.03333333,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                1.,
                0.,
                0.,
                0.,
                0.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                0.66666667,
                0.,
                0.16666667,
                0.,
                0.16666667,
                0.,
                0.,
                0.,
                0.33333333,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -1.,
                0.,
                0.,
            ],
            [
                -0.03333333,
                0.03333333,
                -0.03333333,
                0.,
                0.66666667,
                -0.13333333,
                0.,
                0.,
                0.,
                0.13333333,
                0.,
                0.,
                0.36666667,
                0.,
                0.,
                0.,
                0.,
                0.,
                -1.,
                1.,
                0.,
                0.,
                0.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                0.16666667,
                -0.13333333,
                0.8,
                0.,
                0.,
                0.,
                0.2,
                0.,
                0.16666667,
                -0.03333333,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -1.,
                0.,
                1.,
                0.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.8,
                0.03333333,
                0.,
                0.,
                -0.03333333,
                0.,
                0.,
                0.36666667,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -1.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                0.16666667,
                0.,
                0.,
                0.03333333,
                0.8,
                0.,
                0.,
                0.2,
                0.16666667,
                0.,
                0.13333333,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -1.,
                1.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.66666667,
                0.16666667,
                0.16666667,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.33333333,
                0.,
                0.,
                0.,
                1.,
                0.,
                0.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                0.,
                0.13333333,
                0.2,
                0.,
                0.,
                0.16666667,
                0.8,
                0.,
                0.,
                0.03333333,
                0.,
                0.,
                0.,
                0.16666667,
                0.,
                0.,
                1.,
                -1.,
                0.,
                0.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -0.03333333,
                0.2,
                0.16666667,
                0.,
                0.8,
                0.,
                0.,
                -0.13333333,
                0.,
                0.,
                0.16666667,
                0.,
                0.,
                0.,
                1.,
                0.,
                -1.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                0.33333333,
                0.,
                0.16666667,
                0.,
                0.16666667,
                0.,
                0.,
                0.,
                0.66666667,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                1.,
                0.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                0.,
                0.36666667,
                -0.03333333,
                0.,
                0.,
                0.,
                0.03333333,
                0.,
                0.,
                0.8,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -1.,
                0.,
                0.,
                0.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.36666667,
                0.13333333,
                0.,
                0.,
                -0.13333333,
                0.,
                0.,
                0.66666667,
                -0.03333333,
                0.03333333,
                0.,
                -0.03333333,
                0.,
                0.,
                0.,
                0.,
                1.,
                -1.,
            ],
            [
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -0.03333333,
                0.53333333,
                0.13333333,
                0.,
                -0.13333333,
                0.,
                0.,
                0.,
                0.,
                0.,
                1.,
            ],
            [
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.03333333,
                0.13333333,
                0.53333333,
                0.,
                0.13333333,
                0.,
                0.,
                0.,
                0.,
                0.,
                -1.,
            ],
            [
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.33333333,
                0.16666667,
                0.16666667,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.66666667,
                0.,
                0.,
                0.,
                -1.,
                0.,
                0.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -0.03333333,
                -0.13333333,
                0.13333333,
                0.,
                0.53333333,
                0.,
                0.,
                0.,
                0.,
                0.,
                1.,
            ],
            [
                1.,
                -1.,
                1.,
                0.,
                -1.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                0.,
                1.,
                -1.,
                0.,
                0.,
                0.,
                1.,
                0.,
                0.,
                -1.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                1.,
                -1.,
                1.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -1.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                -1.,
                0.,
                1.,
                0.,
                -1.,
                0.,
                0.,
                0.,
                1.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -1.,
                1.,
                0.,
                0.,
                -1.,
                0.,
                0.,
                1.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -1.,
                1.,
                -1.,
                0.,
                1.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
            ],
        ]
    )


# ------------------------------------------------------------------------------#
