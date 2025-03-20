"""Module contains two sets of unit tests for lowest-order Raviart-Thomas (RT0).
- The first is dedicated for discrete operators;
- The second is dedicated for the right-hand side (Gravitational forces).
"""

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.applications.test_utils.arrays import compare_arrays
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data

"""Test collections for RT0 discretization"""


class TestRaviartThomasDiscretization:
    def _create_subdomain_grid(self, dimension):
        """Generates a mono-dimensional grid."""
        if dimension == 1:
            sd = pp.CartGrid(3, 1)
            sd.compute_geometry()
        elif dimension == 2:
            sd = pp.StructuredTriangleGrid([1, 1], [1, 1])
            sd.compute_geometry()
        else:
            sd = pp.StructuredTetrahedralGrid([1, 1, 1], [1, 1, 1])
            sd.compute_geometry()
        return sd

    def _create_single_cell_grid(self, dimension):
        """Generates a single cell grid"""
        if dimension == 1:
            raise NotImplementedError
        elif dimension == 2:
            nodes = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

            indices = [0, 1, 1, 2, 2, 0]
            indptr = [0, 2, 4, 6]
            face_nodes = sps.csc_matrix(([True] * 6, indices, indptr))
            cell_faces = sps.csc_matrix([[1], [1], [1]])
            name = "test"

            sd = pp.Grid(dimension, nodes, face_nodes, cell_faces, name)
            sd.compute_geometry()
        else:
            nodes = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            indices = [0, 2, 1, 1, 2, 3, 2, 0, 3, 3, 0, 1]
            indptr = [0, 3, 6, 9, 12]
            face_nodes = sps.csc_matrix(([True] * 12, indices, indptr))
            cell_faces = sps.csc_matrix([[1], [1], [1], [1]])
            name = "test"

            sd = pp.Grid(dimension, nodes, face_nodes, cell_faces, name)
            sd.compute_geometry()
        return sd

    def _matrix(self, sd, perm, bc):
        """Compute stiffness and projector operators for a given subdomain"""
        solver = pp.RT0(keyword="flow")
        data = pp.initialize_default_data(
            sd, {}, "flow", {"second_order_tensor": perm, "bc": bc}
        )
        solver.discretize(sd, data)
        M = solver.assemble_matrix(sd, data).todense()
        P = data[pp.DISCRETIZATION_MATRICES]["flow"][solver.vector_proj_key].todense()
        return M, P

    def _assertion(self, M, P, M_ref, P_ref):
        # Asserts the operator M is symmetric
        assert np.allclose(M, M.T)
        # Asserts the operator M match the reference state
        assert np.allclose(M, M_ref)
        # Asserts the operator P match the reference state
        assert np.allclose(P, P_ref)

    """Test one-dimensional for isotropic material data."""

    def test_1d_isotropic_permeability(self):
        sd = self._create_subdomain_grid(dimension=1)

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx, kyy=1, kzz=1)
        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])

        M, P = self._matrix(sd, perm, bc)
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

        self._assertion(M, P, M_known, P_known)

    """Test one-dimensional for anisotropic material data."""

    def test_1d_anisotropic_permeability(self):
        sd = self._create_subdomain_grid(dimension=1)

        kxx = 1.0 / (np.sin(sd.cell_centers[0, :]) + 1)
        perm = pp.SecondOrderTensor(kxx, kyy=1, kzz=1)
        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        M, P = self._matrix(sd, perm, bc)

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

        self._assertion(M, P, M_known, P_known)

    """Test two-dimensional for isotropic material data."""

    def test_2d_isotropic_permeability(self):
        sd = self._create_subdomain_grid(dimension=2)

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)

        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        M, P = self._matrix(sd, perm, bc)

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [0.33333333, 0.0, 0.0, -0.16666667, 0.0, -1.0, 0.0],
                [0.0, 0.33333333, 0.0, 0.0, -0.16666667, 0.0, 1.0],
                [0.0, 0.0, 0.33333333, 0.0, 0.0, 1.0, -1.0],
                [-0.16666667, 0.0, 0.0, 0.33333333, 0.0, -1.0, 0.0],
                [0.0, -0.16666667, 0.0, 0.0, 0.33333333, 0.0, 1.0],
                [-1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )

        P_known = np.matrix(
            [
                [-0.33333333, 0.0, 0.33333333, 0.66666667, 0.0],
                [-0.66666667, 0.0, -0.33333333, 0.33333333, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.66666667, 0.33333333, 0.0, -0.33333333],
                [0.0, 0.33333333, -0.33333333, 0.0, -0.66666667],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        self._assertion(M, P, M_known, P_known)

    """Test two-dimensional for anisotropic material data."""

    def test_2d_anisotropic_permeability(self):
        sd = self._create_subdomain_grid(dimension=2)

        al = np.square(sd.cell_centers[1, :]) + np.square(sd.cell_centers[0, :]) + 1
        kxx = (np.square(sd.cell_centers[0, :]) + 1) / al
        kyy = (np.square(sd.cell_centers[1, :]) + 1) / al
        kxy = np.multiply(sd.cell_centers[0, :], sd.cell_centers[1, :]) / al

        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kxy=kxy, kzz=1)

        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        M, P = self._matrix(sd, perm, bc)

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [0.39814815, 0.0, 0.0462963, -0.15740741, 0.0, -1.0, 0.0],
                [0.0, 0.39814815, 0.0462963, 0.0, -0.15740741, 0.0, 1.0],
                [0.0462963, 0.0462963, 0.46296296, -0.00925926, -0.00925926, 1.0, -1.0],
                [-0.15740741, 0.0, -0.00925926, 0.34259259, 0.0, -1.0, 0.0],
                [0.0, -0.15740741, -0.00925926, 0.0, 0.34259259, 0.0, 1.0],
                [-1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )

        P_known = np.matrix(
            [
                [-0.33333333, 0.0, 0.33333333, 0.66666667, 0.0],
                [-0.66666667, 0.0, -0.33333333, 0.33333333, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.66666667, 0.33333333, 0.0, -0.33333333],
                [0.0, 0.33333333, -0.33333333, 0.0, -0.66666667],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        self._assertion(M, P, M_known, P_known)

    """Test for single 2d-cell with isotropic material data."""

    def test_single_triangle_discretization(self):
        sd = self._create_single_cell_grid(dimension=2)

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)

        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        M, P = self._matrix(sd, perm, bc)

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [0.33333333, 0.0, -0.16666667, -1.0],
                [0.0, 0.16666667, 0.0, -1.0],
                [-0.16666667, 0.0, 0.33333333, -1.0],
                [-1.0, -1.0, -1.0, 0.0],
            ]
        )

        P_known = np.matrix(
            [
                [0.33333333, 0.33333333, -0.66666667],
                [-0.66666667, 0.33333333, 0.33333333],
                [0.0, 0.0, 0.0],
            ]
        )

        self._assertion(M, P, M_known, P_known)

    """Test for single 3d-cell with isotropic material data."""

    def test_single_tetrahedron_discretization(self):
        sd = self._create_single_cell_grid(dimension=3)

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=kxx)

        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        M, P = self._matrix(sd, perm, bc)
        M_known = np.matrix(
            [
                [0.53333333, 0.03333333, -0.13333333, -0.13333333, -1.0],
                [0.03333333, 0.2, 0.03333333, 0.03333333, -1.0],
                [-0.13333333, 0.03333333, 0.53333333, -0.13333333, -1.0],
                [-0.13333333, 0.03333333, -0.13333333, 0.53333333, -1.0],
                [-1.0, -1.0, -1.0, -1.0, 0.0],
            ]
        )

        P_known = np.matrix(
            [[0.5, 0.5, -1.5, 0.5], [0.5, 0.5, 0.5, -1.5], [-1.5, 0.5, 0.5, 0.5]]
        )

        self._assertion(M, P, M_known, P_known)

    """Test for single 3d-cell with isotropic material data."""

    def test_2d_isotropic_permeability_mixed_bc(self):
        sd = pp.StructuredTriangleGrid([2, 2], [1, 1])
        sd.compute_geometry()

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)

        tol = 1e-6
        bf = sd.tags["domain_boundary_faces"].nonzero()[0]
        bf_centers = sd.face_centers[:, bf]
        left = bf_centers[0, :] < 0 + tol
        risdht = bf_centers[0, :] > 1 - tol

        labels = np.array(["neu"] * bf.size)
        labels[left] = "rob"
        labels[risdht] = "dir"

        bc = pp.BoundaryCondition(sd, bf, labels)
        bc.robin_weight[bf[left]] = 2

        bc_val = np.zeros(sd.num_faces)
        bc_val[bf[left]] = 3

        solver = pp.RT0(keyword="flow")

        specified_parameters = {
            "bc": bc,
            "second_order_tensor": perm,
            "bc_values": bc_val,
        }
        data = pp.initialize_default_data(sd, {}, "flow", specified_parameters)
        solver.discretize(sd, data)
        M, rhs = solver.assemble_matrix_rhs(sd, data)
        up = sps.linalg.spsolve(M, rhs)

        p = solver.extract_pressure(sd, up, data)
        u = solver.extract_flux(sd, up, data)
        P0u = solver.project_flux(sd, u, data)

        p_ex = 1 - sd.cell_centers[0, :]
        P0u_ex = np.vstack(
            (np.ones(sd.num_cells), np.zeros(sd.num_cells), np.zeros(sd.num_cells))
        )

        assert np.allclose(p, p_ex)
        assert np.allclose(P0u, P0u_ex)

        r, c, d = sparse_array_to_row_col_data(M)

        r_known = np.array(
            [
                0,
                1,
                1,
                1,
                2,
                2,
                2,
                3,
                4,
                4,
                4,
                4,
                4,
                5,
                5,
                5,
                6,
                6,
                6,
                7,
                7,
                7,
                7,
                7,
                8,
                8,
                8,
                9,
                9,
                9,
                10,
                10,
                10,
                10,
                10,
                11,
                11,
                11,
                11,
                11,
                12,
                12,
                12,
                13,
                13,
                13,
                14,
                15,
                16,
                16,
                16,
                17,
                17,
                17,
                18,
                18,
                18,
                19,
                19,
                19,
                20,
                20,
                20,
                21,
                21,
                21,
                22,
                22,
                22,
                23,
                23,
                23,
            ],
            dtype=np.int32,
        )
        c_known = np.array(
            [
                0,
                1,
                7,
                17,
                2,
                16,
                17,
                3,
                0,
                4,
                10,
                16,
                19,
                5,
                18,
                19,
                3,
                6,
                18,
                1,
                7,
                11,
                17,
                20,
                8,
                14,
                21,
                9,
                20,
                21,
                4,
                10,
                13,
                19,
                22,
                7,
                11,
                15,
                20,
                23,
                12,
                22,
                23,
                10,
                13,
                22,
                14,
                15,
                0,
                2,
                4,
                1,
                2,
                7,
                3,
                5,
                6,
                4,
                5,
                10,
                7,
                9,
                11,
                8,
                9,
                14,
                10,
                12,
                13,
                11,
                12,
                15,
            ],
            dtype=np.int32,
        )
        d_known = np.array(
            [
                1.0,
                1.33333333,
                -0.16666667,
                1.0,
                0.33333333,
                1.0,
                -1.0,
                1.0,
                -0.16666667,
                0.66666667,
                -0.16666667,
                -1.0,
                1.0,
                0.33333333,
                1.0,
                -1.0,
                -0.16666667,
                0.33333333,
                -1.0,
                -0.16666667,
                0.66666667,
                -0.16666667,
                1.0,
                -1.0,
                1.33333333,
                -0.16666667,
                1.0,
                0.33333333,
                1.0,
                -1.0,
                -0.16666667,
                0.66666667,
                -0.16666667,
                1.0,
                -1.0,
                -0.16666667,
                0.66666667,
                -0.16666667,
                -1.0,
                1.0,
                0.33333333,
                1.0,
                -1.0,
                -0.16666667,
                0.33333333,
                -1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
            ]
        )

        assert np.allclose(r, r_known)
        assert np.allclose(c, c_known)
        assert np.allclose(d, d_known)

    """Test three-dimensional for anisotropic material data."""

    def test_3d_isotropic_permeability(self):
        sd = self._create_subdomain_grid(dimension=3)

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=kxx)

        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        M, P = self._matrix(sd, perm, bc)
        M_known, P_known = matrix_for_test_rt0_3d()

        self._assertion(M, P, M_known, P_known)

    """Test one-dimensional R1-R3 for isotropic material data."""

    def test_1d_R1_R3_isotropic_permeability(self):
        sd = self._create_subdomain_grid(dimension=1)
        R = pp.map_geometry.rotation_matrix(np.pi / 6.0, [0, 0, 1])
        sd.nodes = np.dot(R, sd.nodes)
        sd.compute_geometry()

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx, kyy=1, kzz=1)
        perm.rotate(R)

        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        M, P = self._matrix(sd, perm, bc)

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

        self._assertion(M, P, M_known, P_known)

    """Test two-dimensional R2-R3 for isotropic material data."""

    def test_2d_R2_R3_isotropic_permeability(self):
        sd = self._create_subdomain_grid(dimension=2)
        R = pp.map_geometry.rotation_matrix(-np.pi / 4.0, [1, 1, -1])
        sd.nodes = np.dot(R, sd.nodes)
        sd.compute_geometry()

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
        perm.rotate(R)

        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        M, P = self._matrix(sd, perm, bc)

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [0.33333333, 0.0, 0.0, -0.16666667, 0.0, -1.0, 0.0],
                [0.0, 0.33333333, 0.0, 0.0, -0.16666667, 0.0, 1.0],
                [0.0, 0.0, 0.33333333, 0.0, 0.0, 1.0, -1.0],
                [-0.16666667, 0.0, 0.0, 0.33333333, 0.0, -1.0, 0.0],
                [0.0, -0.16666667, 0.0, 0.0, 0.33333333, 0.0, 1.0],
                [-1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )

        P_known = np.matrix(
            [
                [-0.06116781, 0.0, 0.37178502, 0.43295283, 0.0],
                [-0.70511836, 0.0, -0.0996195, 0.60549886, 0.0],
                [0.23371384, 0.0, 0.27216553, 0.03845169, 0.0],
                [0.0, 0.43295283, 0.37178502, 0.0, -0.06116781],
                [0.0, 0.60549886, -0.0996195, 0.0, -0.70511836],
                [0.0, 0.03845169, 0.27216553, 0.0, 0.23371384],
            ]
        )

        self._assertion(M, P, M_known, P_known)

    """Test one-dimensional R2-R3 for aniisotropic material data."""

    def test_2d_R2_R3_anisotropic_permeability(self):
        sd = self._create_subdomain_grid(dimension=2)

        al = np.square(sd.cell_centers[1, :]) + np.square(sd.cell_centers[0, :]) + 1
        kxx = (np.square(sd.cell_centers[0, :]) + 1) / al
        kyy = (np.square(sd.cell_centers[1, :]) + 1) / al
        kxy = np.multiply(sd.cell_centers[0, :], sd.cell_centers[1, :]) / al

        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kxy=kxy, kzz=1)

        R = pp.map_geometry.rotation_matrix(np.pi / 3.0, [1, 1, 0])
        perm.rotate(R)
        sd.nodes = np.dot(R, sd.nodes)
        sd.compute_geometry()

        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        M, P = self._matrix(sd, perm, bc)

        # Matrix computed with an already validated code
        M_known = np.matrix(
            [
                [0.39814815, 0.0, 0.0462963, -0.15740741, 0.0, -1.0, 0.0],
                [0.0, 0.39814815, 0.0462963, 0.0, -0.15740741, 0.0, 1.0],
                [0.0462963, 0.0462963, 0.46296296, -0.00925926, -0.00925926, 1.0, -1.0],
                [-0.15740741, 0.0, -0.00925926, 0.34259259, 0.0, -1.0, 0.0],
                [0.0, -0.15740741, -0.00925926, 0.0, 0.34259259, 0.0, 1.0],
                [-1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )

        P_known = np.matrix(
            [
                [-0.41666667, 0.0, 0.16666667, 0.58333333, 0.0],
                [-0.58333333, 0.0, -0.16666667, 0.41666667, 0.0],
                [-0.20412415, 0.0, -0.40824829, -0.20412415, 0.0],
                [0.0, 0.58333333, 0.16666667, 0.0, -0.41666667],
                [0.0, 0.41666667, -0.16666667, 0.0, -0.58333333],
                [0.0, -0.20412415, -0.40824829, 0.0, -0.20412415],
            ]
        )

        self._assertion(M, P, M_known, P_known)

    """Test two-dimensional convergence (isotropic material)."""

    def test_convergence_2d_isotropic_permeability_exact(self):
        p_ex = lambda pt: 2 * pt[0, :] - 3 * pt[1, :] - 9
        u_ex = np.array([-2, 3, 0])

        for i in np.arange(5):
            sd = pp.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            sd.compute_geometry()

            kxx = np.ones(sd.num_cells)
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
            bf = sd.get_boundary_faces()
            bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
            bc_val = np.zeros(sd.num_faces)
            bc_val[bf] = p_ex(sd.face_centers[:, bf])

            solver = pp.RT0(keyword="flow")

            specified_parameters = {
                "bc": bc,
                "bc_values": bc_val,
                "second_order_tensor": perm,
            }
            data = pp.initialize_default_data(sd, {}, "flow", specified_parameters)

            solver.discretize(sd, data)
            M, rhs = solver.assemble_matrix_rhs(sd, data)
            up = sps.linalg.spsolve(M, rhs)
            p = solver.extract_pressure(sd, up, data)
            err = np.sum(np.abs(p - p_ex(sd.cell_centers)))

            assert np.isclose(err, 0)

            u = solver.extract_flux(sd, up, data)
            P0u = solver.project_flux(sd, u, data)
            err = np.sum(
                np.abs(P0u - np.tile(u_ex, sd.num_cells).reshape((3, -1), order="F"))
            )

            assert np.isclose(err, 0)

    """Test two-dimensional convergence variable rhs (isotropic material)."""

    def test_convergence_2d_isotropic_permeability_variable_rhs(self):
        a = 8 * np.pi**2
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
            sd = pp.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            sd.compute_geometry()

            kxx = np.ones(sd.num_cells)
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
            bf = sd.get_boundary_faces()
            bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
            bc_val = np.zeros(sd.num_faces)
            bc_val[bf] = p_ex(sd.face_centers[:, bf])
            # Minus sisdn to move to rhs
            source = np.multiply(sd.cell_volumes, rhs_ex(sd.cell_centers))

            solver = pp.RT0(keyword="flow")
            solver_rhs = pp.DualScalarSource(keyword="flow")

            specified_parameters = {
                "bc": bc,
                "bc_values": bc_val,
                "second_order_tensor": perm,
                "source": source,
            }
            data = pp.initialize_default_data(sd, {}, "flow", specified_parameters)

            solver.discretize(sd, data)
            solver_rhs.discretize(sd, data)

            M, rhs_bc = solver.assemble_matrix_rhs(sd, data)
            _, rhs = solver_rhs.assemble_matrix_rhs(sd, data)

            up = sps.linalg.spsolve(M, rhs_bc + rhs)
            p = solver.extract_pressure(sd, up, data)
            err = np.sqrt(
                np.sum(
                    np.multiply(sd.cell_volumes, np.power(p - p_ex(sd.cell_centers), 2))
                )
            )
            assert np.isclose(err, p_err_known)

            u = solver.extract_flux(sd, up, data)
            P0u = solver.project_flux(sd, u, data)
            uu_ex_0 = u_ex_0(sd.cell_centers)
            uu_ex_1 = u_ex_1(sd.cell_centers)
            uu_ex_2 = np.zeros(sd.num_cells)
            uu_ex = np.vstack((uu_ex_0, uu_ex_1, uu_ex_2))
            err = np.sqrt(
                np.sum(
                    np.multiply(
                        sd.cell_volumes, np.sum(np.power(P0u - uu_ex, 2), axis=0)
                    )
                )
            )
            assert np.isclose(err, u_err_known)

    """Test two-dimensional convergence (anisotropic material)."""

    def test_convergence_2d_anisotropic_permeability_constant_rhs(self):
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
            sd = pp.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            sd.compute_geometry()

            kxx = 2 * np.ones(sd.num_cells)
            kxy = np.ones(sd.num_cells)
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kxy=kxy, kzz=1)
            bf = sd.get_boundary_faces()
            bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
            bc_val = np.zeros(sd.num_faces)
            bc_val[bf] = p_ex(sd.face_centers[:, bf])
            # Minus sisdn to move to rhs
            source = np.multiply(sd.cell_volumes, rhs_ex(sd.cell_centers))

            solver = pp.RT0(keyword="flow")
            solver_rhs = pp.DualScalarSource(keyword="flow")

            specified_parameters = {
                "bc": bc,
                "bc_values": bc_val,
                "second_order_tensor": perm,
                "source": source,
            }
            data = pp.initialize_default_data(sd, {}, "flow", specified_parameters)

            solver.discretize(sd, data)
            solver_rhs.discretize(sd, data)
            M, rhs_bc = solver.assemble_matrix_rhs(sd, data)
            _, rhs = solver_rhs.assemble_matrix_rhs(sd, data)

            up = sps.linalg.spsolve(M, rhs_bc + rhs)
            p = solver.extract_pressure(sd, up, data)
            err = np.sqrt(
                np.sum(
                    np.multiply(sd.cell_volumes, np.power(p - p_ex(sd.cell_centers), 2))
                )
            )
            assert np.isclose(err, p_err_known)

            u = solver.extract_flux(sd, up, data)
            P0u = solver.project_flux(sd, u, data)
            uu_ex_0 = u_ex_0(sd.cell_centers)
            uu_ex_1 = u_ex_1(sd.cell_centers)
            uu_ex_2 = np.zeros(sd.num_cells)
            uu_ex = np.vstack((uu_ex_0, uu_ex_1, uu_ex_2))
            err = np.sqrt(
                np.sum(
                    np.multiply(
                        sd.cell_volumes, np.sum(np.power(P0u - uu_ex, 2), axis=0)
                    )
                )
            )
            assert np.isclose(err, u_err_known)


def matrix_for_test_rt0_3d():
    """Reference operator for three-dimensional discretization test."""
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


"""Test collections for RT0 consistency of the right-hand side (gravity term)"""


class TestRaviartThomasRHS:
    def _create_subdomain_grid(self, dimension):
        """Generates a mono-dimensional grid."""
        if dimension == 1:
            sd = pp.CartGrid(3, 1)
            sd.compute_geometry()
        elif dimension == 2:
            sd = pp.StructuredTriangleGrid([1, 1], [1, 1])
            sd.compute_geometry()
        else:
            sd = pp.StructuredTetrahedralGrid([1, 1, 1], [1, 1, 1])
            sd.compute_geometry()
        return sd

    def _create_single_cell_grid(self, dimension):
        """Generates a single cell grid"""
        if dimension == 1:
            raise NotImplementedError
        elif dimension == 2:
            nodes = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

            indices = [0, 1, 1, 2, 2, 0]
            indptr = [0, 2, 4, 6]
            face_nodes = sps.csc_matrix(([True] * 6, indices, indptr))
            cell_faces = sps.csc_matrix([[1], [1], [1]])
            name = "test"

            sd = pp.Grid(dimension, nodes, face_nodes, cell_faces, name)
            sd.compute_geometry()
        else:
            nodes = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

            indices = [0, 2, 1, 1, 2, 3, 2, 0, 3, 3, 0, 1]
            indptr = [0, 3, 6, 9, 12]
            face_nodes = sps.csc_matrix(([True] * 12, indices, indptr))
            cell_faces = sps.csc_matrix([[1], [1], [1], [1]])
            name = "test"

            sd = pp.Grid(dimension, nodes, face_nodes, cell_faces, name)
            sd.compute_geometry()
        return sd

    def _rhs(self, sd, perm, bc, vect):
        """Compute rhs for given subdomain"""
        solver = pp.RT0(keyword="flow")

        data = pp.initialize_default_data(
            sd,
            {},
            "flow",
            {"second_order_tensor": perm, "bc": bc, "vector_source": vect},
        )
        solver.discretize(sd, data)
        return solver.assemble_matrix_rhs(sd, data)[1]

    """Test one-dimensional for isotropic material data."""

    def test_1d_isotropic_permeability(self):
        sd = self._create_subdomain_grid(dimension=1)

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx, kyy=1, kzz=1)
        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        vect = np.vstack(
            (sd.cell_volumes, np.zeros(sd.num_cells), np.zeros(sd.num_cells))
        ).ravel(order="F")

        b = self._rhs(sd, perm, bc, vect)
        b_known = np.array(
            [0.16666667, 0.33333333, 0.33333333, 0.16666667, 0.0, 0.0, 0.0]
        )

        assert compare_arrays(b, b_known)

    """Test two-dimensional for isotropic material data."""

    def test_2d_isotropic_permeability(self):
        sd = self._create_subdomain_grid(dimension=2)

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)

        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        vect = np.vstack(
            (2 * sd.cell_volumes, 3 * sd.cell_volumes, np.zeros(sd.num_cells))
        ).ravel(order="F")
        b = self._rhs(sd, perm, bc, vect)

        b_known = np.array(
            [-1.33333333, 1.16666667, -0.33333333, 1.16666667, -1.33333333, 0.0, 0.0]
        )

        assert compare_arrays(b, b_known)

    """Test three-dimensional for isotropic material data."""

    def test_3d_isotropic_permeability(self):
        sd = self._create_subdomain_grid(dimension=3)

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=kxx)

        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        vect = np.vstack(
            (7 * sd.cell_volumes, 4 * sd.cell_volumes, 3 * sd.cell_volumes)
        ).ravel(order="F")

        b = self._rhs(sd, perm, bc, vect)
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

        assert compare_arrays(b, b_known)

    """Test for single 2d-cell with isotropic material data."""

    def test_single_triangle_discretization(self):
        sd = self._create_single_cell_grid(dimension=2)

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)

        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        vect = np.vstack(
            (sd.cell_volumes, 3 * sd.cell_volumes, np.zeros(sd.num_cells))
        ).ravel(order="F")

        b = self._rhs(sd, perm, bc, vect)
        b_known = np.array([-0.83333333, 0.66666667, 0.16666667, 0.0])

        assert compare_arrays(b, b_known)

    """Test for single 3d-cell with isotropic material data."""

    def test_single_tetrahedron_discretization(self):
        sd = self._create_single_cell_grid(dimension=3)

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=kxx)

        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        vect = np.vstack(
            (2 * sd.cell_volumes, 3 * sd.cell_volumes, np.zeros(sd.num_cells))
        ).ravel(order="F")

        b = self._rhs(sd, perm, bc, vect)
        b_known = np.array([0.41666667, 0.41666667, -0.25, -0.58333333, 0.0])

        assert compare_arrays(b, b_known)

    """Test one-dimensional R1-R3 for isotropic material data."""

    def test_1d_R1_R3_isotropic_permeability(self):
        sd = self._create_subdomain_grid(dimension=1)
        R = pp.map_geometry.rotation_matrix(np.pi / 6.0, [0, 0, 1])
        sd.nodes = np.dot(R, sd.nodes)
        sd.compute_geometry()

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx, kyy=1, kzz=1)
        perm.rotate(R)

        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        vect = np.vstack(
            (sd.cell_volumes, 0 * sd.cell_volumes, 0 * sd.cell_volumes)
        ).ravel(order="F")

        b = self._rhs(sd, perm, bc, vect)
        b_known = np.array(
            [0.14433757, 0.28867513, 0.28867513, 0.14433757, 0.0, 0.0, 0.0]
        )

        assert compare_arrays(b, b_known)

    """Test two-dimensional R2-R3 for isotropic material data."""

    def test_2d_R2_R3_isotropic_permeability(self):
        sd = self._create_subdomain_grid(dimension=2)
        R = pp.map_geometry.rotation_matrix(-np.pi / 4.0, [1, 1, -1])
        sd.nodes = np.dot(R, sd.nodes)
        sd.compute_geometry()

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
        perm.rotate(R)

        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        vect = np.vstack(
            (sd.cell_volumes, 2 * sd.cell_volumes, 0 * sd.cell_volumes)
        ).ravel(order="F")

        b = self._rhs(sd, perm, bc, vect)
        b_known = np.array(
            [-0.73570226, 0.82197528, 0.17254603, 0.82197528, -0.73570226, 0.0, 0.0]
        )

        assert compare_arrays(b, b_known)

    """Test two-dimensional convergence (isotropic material)."""

    def test_convergence_2d_isotropic_permeability_exact(self):
        p_ex = lambda pt: 2 * pt[0, :] - 3 * pt[1, :] - 9
        u_ex = np.array([-1, 4, 0])

        for i in np.arange(5):
            sd = pp.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            sd.compute_geometry()

            kxx = np.ones(sd.num_cells)
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
            bf = sd.get_boundary_faces()
            bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
            bc_val = np.zeros(sd.num_faces)
            bc_val[bf] = p_ex(sd.face_centers[:, bf])
            vect = np.vstack(
                (sd.cell_volumes, sd.cell_volumes, np.zeros(sd.num_cells))
            ).ravel(order="F")

            solver = pp.RT0(keyword="flow")

            specified_parameters = {
                "bc": bc,
                "bc_values": bc_val,
                "second_order_tensor": perm,
                "vector_source": vect,
            }
            data = pp.initialize_default_data(sd, {}, "flow", specified_parameters)

            solver.discretize(sd, data)
            M, rhs = solver.assemble_matrix_rhs(sd, data)
            up = sps.linalg.spsolve(M, rhs)
            p = solver.extract_pressure(sd, up, data)
            err = np.sum(np.abs(p - p_ex(sd.cell_centers)))

            assert np.isclose(err, 0)

            _ = data[pp.DISCRETIZATION_MATRICES]["flow"][solver.vector_proj_key]
            u = solver.extract_flux(sd, up, data)
            P0u = solver.project_flux(sd, u, data)
            err = np.sum(
                np.abs(P0u - np.tile(u_ex, sd.num_cells).reshape((3, -1), order="F"))
            )

            assert np.isclose(err, 0)

    """Test two-dimensional convergence variable rhs (isotropic material)."""

    def test_convergence_2d_isotropic_permeability_variable_rhs(self):
        a = 8 * np.pi**2
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
            sd = pp.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            sd.compute_geometry()

            kxx = np.ones(sd.num_cells)
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
            bf = sd.get_boundary_faces()
            bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
            bc_val = np.zeros(sd.num_faces)
            bc_val[bf] = p_ex(sd.face_centers[:, bf])
            # Minus sisdn to move to rhs
            source = np.multiply(sd.cell_volumes, rhs_ex(sd.cell_centers))
            vect = np.vstack(
                (sd.cell_volumes, np.zeros(sd.num_cells), np.zeros(sd.num_cells))
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
            data = pp.initialize_default_data(sd, {}, "flow", specified_parameters)

            solver.discretize(sd, data)
            solver_rhs.discretize(sd, data)

            M, rhs_bc = solver.assemble_matrix_rhs(sd, data)
            _, rhs = solver_rhs.assemble_matrix_rhs(sd, data)

            up = sps.linalg.spsolve(M, rhs_bc + rhs)
            p = solver.extract_pressure(sd, up, data)
            err = np.sqrt(
                np.sum(
                    np.multiply(sd.cell_volumes, np.power(p - p_ex(sd.cell_centers), 2))
                )
            )
            assert np.isclose(err, p_err_known)

            _ = data[pp.DISCRETIZATION_MATRICES]["flow"][solver.vector_proj_key]
            u = solver.extract_flux(sd, up, data)
            P0u = solver.project_flux(sd, u, data)
            uu_ex_0 = u_ex_0(sd.cell_centers)
            uu_ex_1 = u_ex_1(sd.cell_centers)
            uu_ex_2 = np.zeros(sd.num_cells)
            uu_ex = np.vstack((uu_ex_0, uu_ex_1, uu_ex_2))
            err = np.sqrt(
                np.sum(
                    np.multiply(
                        sd.cell_volumes, np.sum(np.power(P0u - uu_ex, 2), axis=0)
                    )
                )
            )
            assert np.isclose(err, u_err_known)

    """Test two-dimensional convergence (anisotropic material)."""

    def test_convergence_2d_anisotropic_permeability_constant_rhs(self):
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
            sd = pp.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            sd.compute_geometry()

            kxx = 2 * np.ones(sd.num_cells)
            kxy = np.ones(sd.num_cells)
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kxy=kxy, kzz=1)
            bf = sd.get_boundary_faces()
            bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
            bc_val = np.zeros(sd.num_faces)
            bc_val[bf] = p_ex(sd.face_centers[:, bf])
            # Minus sisdn to move to rhs
            source = np.multiply(sd.cell_volumes, rhs_ex(sd.cell_centers))
            vect = np.vstack(
                (sd.cell_volumes, 2 * sd.cell_volumes, np.zeros(sd.num_cells))
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
            data = pp.initialize_default_data(sd, {}, "flow", specified_parameters)

            solver.discretize(sd, data)
            solver_rhs.discretize(sd, data)
            M, rhs_bc = solver.assemble_matrix_rhs(sd, data)
            _, rhs = solver_rhs.assemble_matrix_rhs(sd, data)

            up = sps.linalg.spsolve(M, rhs_bc + rhs)
            p = solver.extract_pressure(sd, up, data)
            err = np.sqrt(
                np.sum(
                    np.multiply(sd.cell_volumes, np.power(p - p_ex(sd.cell_centers), 2))
                )
            )
            assert np.isclose(err, p_err_known)

            _ = data[pp.DISCRETIZATION_MATRICES]["flow"][solver.vector_proj_key]
            u = solver.extract_flux(sd, up, data)
            P0u = solver.project_flux(sd, u, data)
            uu_ex_0 = u_ex_0(sd.cell_centers)
            uu_ex_1 = u_ex_1(sd.cell_centers)
            uu_ex_2 = np.zeros(sd.num_cells)
            uu_ex = np.vstack((uu_ex_0, uu_ex_1, uu_ex_2))
            err = np.sqrt(
                np.sum(
                    np.multiply(
                        sd.cell_volumes, np.sum(np.power(P0u - uu_ex, 2), axis=0)
                    )
                )
            )
            assert np.isclose(err, u_err_known)
