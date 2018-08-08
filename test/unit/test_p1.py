import numpy as np
import scipy.sparse as sps
import unittest

import porepy as pp
from porepy import cg

# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#

    def test_p1_1d(self):
        g = pp.structured.CartGrid(1, 1)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(3, kxx, kyy=1, kzz=1)
        bn = g.get_boundary_nodes()
        bc = pp.BoundaryConditionNode(g, bn, bn.size * ["neu"])

        solver = pp.P1(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()

        M_known = np.matrix([[1., -1.], [-1., 1.]])

        assert np.allclose(M, M_known)

        solver = pp.P1MassMatrix(physics="flow")
        M = solver.matrix(g, {"param": param}).todense()

        M_known = np.matrix([[2., 1.], [1., 2.]]) / 6.

        assert np.allclose(M, M_known)

    # ------------------------------------------------------------------------------#

    def test_p1_1d_iso(self):
        g = pp.structured.CartGrid(3, 1)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(3, kxx, kyy=1, kzz=1)
        bn = g.get_boundary_nodes()
        bc = pp.BoundaryConditionNode(g, bn, ["dir", "neu"])

        solver = pp.P1(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()

        M_known = np.matrix(
            [
                [1., 0., 0., 0.],
                [-3., 6., -3., 0.],
                [0., -3., 6., -3.],
                [0., 0., -3., 3.],
            ]
        )

        assert np.allclose(M, M_known)

        solver = pp.P1MassMatrix(physics="flow")
        M = solver.matrix(g, {"param": param}).todense()

        M_known = (
            np.matrix(
                [[0., 0., 0., 0.], [1., 4., 1., 0.], [0., 1., 4., 1.], [0., 0., 1., 2.]]
            )
            / 18.
        )

        assert np.allclose(M, M_known)

    # ------------------------------------------------------------------------------#

    def test_p1_1d_ani(self):
        g = pp.structured.CartGrid(3, 1)
        g.compute_geometry()

        kxx = np.sin(g.cell_centers[0, :]) + 1
        perm = pp.SecondOrderTensor(3, kxx, kyy=1, kzz=1)
        bn = g.get_boundary_nodes()
        bc = pp.BoundaryConditionNode(g, bn, bn.size * ["neu"])
        solver = pp.P1(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()

        M_known = np.matrix(
            [
                [3.4976884, -3.4976884, 0., 0.],
                [-3.4976884, 7.93596501, -4.43827662, 0.],
                [0., -4.43827662, 9.65880718, -5.22053056],
                [0., 0., -5.22053056, 5.22053056],
            ]
        )

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

    # ------------------------------------------------------------------------------#

    def test_p1_2d_iso_simplex(self):
        g = pp.simplex.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kxx, kzz=1)

        bn = g.get_boundary_nodes()
        bc = pp.BoundaryConditionNode(g, bn, bn.size * ["neu"])
        solver = pp.P1(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [1., -0.5, -0.5, 0.],
                [-0.5, 1., 0., -0.5],
                [-0.5, 0., 1., -0.5],
                [0., -0.5, -0.5, 1.],
            ]
        )

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

        solver = pp.P1MassMatrix(physics="flow")
        M = solver.matrix(g, {"param": param}).todense()

        M_known = (
            np.matrix(
                [
                    [1., 0.25, 0.25, 0.5],
                    [0.25, 0.5, 0., 0.25],
                    [0.25, 0., 0.5, 0.25],
                    [0.5, 0.25, 0.25, 1.],
                ]
            )
            / 6.
        )

        assert np.allclose(M, M_known)

    # ------------------------------------------------------------------------------#

    def test_p1_2d_ani_simplex(self):
        g = pp.simplex.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        kxx = np.square(g.cell_centers[1, :]) + 1
        kyy = np.square(g.cell_centers[0, :]) + 1
        kxy = -np.multiply(g.cell_centers[0, :], g.cell_centers[1, :])

        perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kyy, kxy=kxy, kzz=1)

        bn = g.get_boundary_nodes()
        bc = pp.BoundaryConditionNode(g, bn, bn.size * ["neu"])
        solver = pp.P1(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [1.11111111, -0.66666667, -0.66666667, 0.22222222],
                [-0.66666667, 1.5, 0., -0.83333333],
                [-0.66666667, 0., 1.5, -0.83333333],
                [0.22222222, -0.83333333, -0.83333333, 1.44444444],
            ]
        )

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

    # ------------------------------------------------------------------------------#

    def test_p1_3d(self):

        g = pp.simplex.StructuredTetrahedralGrid([1, 1, 1], [1, 1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kxx, kzz=kxx)

        bn = g.get_boundary_nodes()
        bc = pp.BoundaryConditionNode(g, bn, bn.size * ["neu"])
        solver = pp.P1(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()

        M_known = matrix_for_test_p1_3d()

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

        solver = pp.P1MassMatrix(physics="flow")
        M = solver.matrix(g, {"param": param}).todense()

        M_known = (
            np.matrix(
                [
                    [1., 0.5, 0.5, 0., 0.5, 0., 0., 0.],
                    [0.5, 5., 1.5, 1., 1.5, 1., 2., 0.],
                    [0.5, 1.5, 3., 0.5, 1., 0., 1., 0.],
                    [0., 1., 0.5, 3., 0., 1., 1.5, 0.5],
                    [0.5, 1.5, 1., 0., 3., 0.5, 1., 0.],
                    [0., 1., 0., 1., 0.5, 3., 1.5, 0.5],
                    [0., 2., 1., 1.5, 1., 1.5, 5., 0.5],
                    [0., 0., 0., 0.5, 0., 0.5, 0.5, 1.],
                ]
            )
            / 60.
        )

        assert np.allclose(M, M_known)

    # ------------------------------------------------------------------------------#

    def test_p1_1d_iso_line(self):
        g = pp.structured.CartGrid(3, 1)
        R = cg.rot(np.pi / 6., [0, 0, 1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(3, kxx, kyy=1, kzz=1)
        perm.rotate(R)

        bn = g.get_boundary_nodes()
        bc = pp.BoundaryConditionNode(g, bn, ["dir", "neu"])
        solver = pp.P1(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [1., 0., 0., 0.],
                [-3., 6., -3., 0.],
                [0., -3., 6., -3.],
                [0., 0., -3., 3.],
            ]
        )

        assert np.allclose(M, M_known)

        solver = pp.P1MassMatrix(physics="flow")
        M = solver.matrix(g, {"param": param}).todense()

        M_known = (
            np.matrix(
                [[0., 0., 0., 0.], [1., 4., 1., 0.], [0., 1., 4., 1.], [0., 0., 1., 2.]]
            )
            / 18.
        )

        assert np.allclose(M, M_known)

    # ------------------------------------------------------------------------------#

    def test_p1_2d_iso_simplex_surf(self):
        g = pp.simplex.StructuredTriangleGrid([1, 1], [1, 1])
        R = cg.rot(-np.pi / 4., [1, 1, -1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kxx, kzz=1)
        perm.rotate(R)

        bn = g.get_boundary_nodes()
        bc = pp.BoundaryConditionNode(g, bn, bn.size * ["neu"])
        solver = pp.P1(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [1., -0.5, -0.5, 0.],
                [-0.5, 1., 0., -0.5],
                [-0.5, 0., 1., -0.5],
                [0., -0.5, -0.5, 1.],
            ]
        )

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

        solver = pp.P1MassMatrix(physics="flow")
        M = solver.matrix(g, {"param": param}).todense()

        M_known = (
            np.matrix(
                [
                    [1., 0.25, 0.25, 0.5],
                    [0.25, 0.5, 0., 0.25],
                    [0.25, 0., 0.5, 0.25],
                    [0.5, 0.25, 0.25, 1.],
                ]
            )
            / 6.
        )

        assert np.allclose(M, M_known)

    # ------------------------------------------------------------------------------#

    def test_p1_2d_ani_simplex_surf(self):
        g = pp.simplex.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        kxx = np.square(g.cell_centers[1, :]) + 1
        kyy = np.square(g.cell_centers[0, :]) + 1
        kxy = -np.multiply(g.cell_centers[0, :], g.cell_centers[1, :])
        perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kyy, kxy=kxy, kzz=1)

        R = cg.rot(np.pi / 3., [1, 1, 0])
        perm.rotate(R)
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        bn = g.get_boundary_nodes()
        bc = pp.BoundaryConditionNode(g, bn, bn.size * ["neu"])
        solver = pp.P1(physics="flow")

        param = pp.Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {"param": param}).todense()

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [1.11111111, -0.66666667, -0.66666667, 0.22222222],
                [-0.66666667, 1.5, 0., -0.83333333],
                [-0.66666667, 0., 1.5, -0.83333333],
                [0.22222222, -0.83333333, -0.83333333, 1.44444444],
            ]
        )

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

    # ------------------------------------------------------------------------------#

    def test_p1_convergence_1d_exact(self):

        p_ex = lambda pt: 2 * pt[0, :] - 3

        for i in np.arange(5):
            g = pp.structured.CartGrid(3 + i, 1)
            g.compute_geometry()

            kxx = np.ones(g.num_cells)
            perm = pp.SecondOrderTensor(3, kxx, kyy=1, kzz=1)
            bn = g.get_boundary_nodes()
            bc = pp.BoundaryConditionNode(g, bn, bn.size * ["dir"])
            bc_val = np.zeros(g.num_nodes)
            bc_val[bn] = p_ex(g.nodes[:, bn])

            solver = pp.P1(physics="flow")

            param = pp.Parameters(g)
            param.set_tensor(solver, perm)
            param.set_bc(solver, bc)
            param.set_bc_val(solver, bc_val)
            M, rhs = solver.matrix_rhs(g, {"param": param})

            p = sps.linalg.spsolve(M, rhs)
            err = np.sum(np.abs(p - p_ex(g.nodes)))

            assert np.isclose(err, 0)

    # ------------------------------------------------------------------------------#

    def test_p1_convergence_1d_not_exact(self):

        p_ex = lambda pt: np.sin(2 * np.pi * pt[0, :])
        source_ex = lambda pt: 4 * np.pi ** 2 * np.sin(2 * np.pi * pt[0, :])

        known_errors = [
            0.0739720694066,
            0.0285777089832,
            0.00791150843359,
            0.00202828006648,
            0.00051026002257,
            0.000127765008718,
            3.19537621983e-05,
        ]

        for i, known_error in zip(np.arange(7), known_errors):
            g = pp.structured.CartGrid(4 * 2 ** i, 1)
            g.compute_geometry()

            kxx = np.ones(g.num_cells)
            perm = pp.SecondOrderTensor(3, kxx, kyy=1, kzz=1)
            bn = g.get_boundary_nodes()
            bc = pp.BoundaryConditionNode(g, bn, bn.size * ["dir"])
            source_val = source_ex(g.nodes)

            solver = pp.P1(physics="flow")
            integral = pp.P1Source(physics="flow")

            param = pp.Parameters(g)
            param.set_tensor(solver, perm)
            param.set_bc(solver, bc)
            param.set_source(solver, source_val)
            M, _ = solver.matrix_rhs(g, {"param": param})
            _, rhs = integral.matrix_rhs(g, {"param": param})

            p = sps.linalg.spsolve(M, rhs)

            mass = pp.P1MassMatrix(physics="flow")
            M = mass.matrix(g, {"param": param})

            error = np.sum(M.dot(np.abs(p - p_ex(g.nodes))))
            assert np.isclose(error, known_error)

    # ------------------------------------------------------------------------------#

    def test_p1_convergence_2d_exact(self):

        p_ex = lambda pt: 2 * pt[0, :] - 3 * pt[1, :] - 9

        for i in np.arange(5):
            g = pp.simplex.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            g.compute_geometry()

            kxx = np.ones(g.num_cells)
            perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kxx, kzz=1)
            bn = g.get_boundary_nodes()
            bc = pp.BoundaryConditionNode(g, bn, bn.size * ["dir"])
            bc_val = np.zeros(g.num_nodes)
            bc_val[bn] = p_ex(g.nodes[:, bn])

            solver = pp.P1(physics="flow")

            param = pp.Parameters(g)
            param.set_tensor(solver, perm)
            param.set_bc(solver, bc)
            param.set_bc_val(solver, bc_val)
            M, rhs = solver.matrix_rhs(g, {"param": param})

            p = sps.linalg.spsolve(M, rhs)
            err = np.sum(np.abs(p - p_ex(g.nodes)))
            assert np.isclose(err, 0)


# ------------------------------------------------------------------------------#


def matrix_for_test_p1_3d():
    return np.matrix(
        [
            [0.5, -0.16666667, -0.16666667, 0., -0.16666667, 0., 0., 0.],
            [
                -0.16666667,
                1.16666667,
                -0.16666667,
                -0.5,
                -0.16666667,
                -0.5,
                0.33333333,
                0.,
            ],
            [
                -0.16666667,
                -0.16666667,
                0.83333333,
                -0.16666667,
                0.16666667,
                0.,
                -0.5,
                0.,
            ],
            [
                0.,
                -0.5,
                -0.16666667,
                0.83333333,
                0.,
                0.16666667,
                -0.16666667,
                -0.16666667,
            ],
            [
                -0.16666667,
                -0.16666667,
                0.16666667,
                0.,
                0.83333333,
                -0.16666667,
                -0.5,
                0.,
            ],
            [
                0.,
                -0.5,
                0.,
                0.16666667,
                -0.16666667,
                0.83333333,
                -0.16666667,
                -0.16666667,
            ],
            [
                0.,
                0.33333333,
                -0.5,
                -0.16666667,
                -0.5,
                -0.16666667,
                1.16666667,
                -0.16666667,
            ],
            [0., 0., 0., -0.16666667, 0., -0.16666667, -0.16666667, 0.5],
        ]
    )


# ------------------------------------------------------------------------------#
