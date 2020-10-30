import numpy as np
import scipy.sparse as sps
import unittest

import porepy as pp


class TestMVEMGravity(unittest.TestCase):

    # ------------------------------------------------------------------------------#
    def _matrix(self, g, perm, bc, vect):
        solver = pp.MVEM(keyword="flow")

        data = pp.initialize_default_data(
            g,
            {},
            "flow",
            {"second_order_tensor": perm, "bc": bc, "vector_source": vect},
        )
        solver.discretize(g, data)
        return solver.assemble_matrix_rhs(g, data)[1]

    # ------------------------------------------------------------------------------#

    def test_mvem_1d(self):
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

    def test_mvem_2d_triangle(self):

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

    def test_mvem_2d_simplex(self):
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

    def test_mvem_tetra(self):

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

    def test_convergence_mvem_2d_iso_simplex_exact(self):

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

            solver = pp.MVEM(keyword="flow")

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

    def test_mvem_3d(self):

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

    def test_dual_mvem_1d_line(self):
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

    def test_mvem_2d_simplex_surf(self):
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

    def test_convergence_mvem_2d_iso_simplex(self):

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
                0.007347293666843033,
                0.004057878042430692,
                0.002576479539795832,
                0.0017817307824819935,
                0.0013057660031758425,
            ]
        )

        u_errs_known = np.array(
            [
                0.024425617686195774,
                0.016806807988931565,
                0.012859109258624922,
                0.010445238111710832,
                0.00881184436169123,
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

            solver = pp.MVEM(keyword="flow")
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

    def test_convergence_mvem_2d_ani_simplex(self):

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
                0.2411784823808065,
                0.13572349427526526,
                0.08688469978140642,
                0.060345813825004285,
                0.044340156291519606,
            ]
        )
        u_errs_known = np.array(
            [
                1.7264059760345327,
                1.3416423116340397,
                1.0925566034251672,
                0.9198698104736416,
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

            solver = pp.MVEM(keyword="flow")
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


if __name__ == "__main__":
    unittest.main()
