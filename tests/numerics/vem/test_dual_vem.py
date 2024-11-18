""" Module contains two sets of unit tests for mixed Virtual Element Method (MVEM).
    - The first is dedicated for discrete operators;
    - The second is dedicated for the right-hand side (Gravitational forces).
"""


import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.applications.test_utils import reference_dense_arrays
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data


class TestMVEMDiscretization:
    def _matrix(self, sd, perm, bc):
        """Compute stiffness operator for a given subdomain"""
        solver = pp.MVEM(keyword="flow")

        data = pp.initialize_default_data(
            sd, {}, "flow", {"second_order_tensor": perm, "bc": bc}
        )
        solver.discretize(sd, data)

        return solver.assemble_matrix(sd, data).todense()

    def _create_cartesian_grid(self, dimension):
        """Generates a mono-dimensional cartesian grid."""
        if dimension == 1:
            sd = pp.CartGrid(3, 1)
            sd.compute_geometry()
        elif dimension == 2:
            sd = pp.CartGrid([2, 1], [1, 1])
            sd.compute_geometry()
        else:
            sd = pp.CartGrid([2, 2, 2], [1, 1, 1])
            sd.compute_geometry()
        return sd

    def test_1d_isotropic_permeability(self):
        sd = self._create_cartesian_grid(dimension=1)

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx)
        bf = sd.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])

        M = self._matrix(sd, perm, bc)

        M_known = 1e-2 * np.array(
            [
                [25, -(8 + 1 / 3.0), 0, 0, 1e2, 0, 0],
                [-(8 + 1 / 3.0), 50, -(8 + 1 / 3.0), 0, -1e2, 1e2, 0],
                [0, -(8 + 1 / 3.0), 50, -(8 + 1 / 3.0), 0, -1e2, 1e2],
                [0, 0, -(8 + 1 / 3.0), 25, 0, 0, -1e2],
                [1e2, -1e2, 0, 0, 0, 0, 0],
                [0, 1e2, -1e2, 0, 0, 0, 0],
                [0, 0, 1e2, -1e2, 0, 0, 0],
            ]
        )

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        assert np.allclose(M, M_known, rtol, atol)

    def test_1d_anisotropic_permeability(self):
        sd = self._create_cartesian_grid(dimension=1)

        kxx = np.sin(sd.cell_centers[0, :]) + 1
        perm = pp.SecondOrderTensor(
            kxx, kyy=np.ones(sd.num_cells), kzz=np.ones(sd.num_cells)
        )
        bf = sd.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])

        M = self._matrix(sd, perm, bc)
        M_known = 1e-2 * np.array(
            [
                [21.4427334468001192, -7.14757781560004, 0, 0, 1e2, 0, 0],
                [
                    -7.14757781560004,
                    38.3411844655562319,
                    -5.6328170062520355,
                    0,
                    -1e2,
                    1e2,
                    0,
                ],
                [
                    0,
                    -5.6328170062520355,
                    31.2648069176318977,
                    -4.788785299625264,
                    0,
                    -1e2,
                    1e2,
                ],
                [0, 0, -4.7887852996252649, 14.3663558988757991, 0, 0, -1e2],
                [1e2, -1e2, 0, 0, 0, 0, 0],
                [0, 1e2, -1e2, 0, 0, 0, 0],
                [0, 0, 1e2, -1e2, 0, 0, 0],
            ]
        )

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        assert np.allclose(M, M_known, rtol, atol)

    def test_2d_isotropic_permeability_cart(self):
        sd = self._create_cartesian_grid(dimension=2)

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=kxx)

        bf = sd.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])

        M = self._matrix(sd, perm, bc)
        # Matrix computed with an already validated code (MRST)
        M_known = np.array(
            [
                [0.625, -0.375, 0, 0, 0, 0, 0, 1, 0],
                [-0.375, 1.25, -0.375, 0, 0, 0, 0, -1, 1],
                [0, -0.375, 0.625, 0, 0, 0, 0, 0, -1],
                [0, 0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1, 0, -1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, -1],
                [1, -1, 0, 1, 0, -1, 0, 0, 0],
                [0, 1, -1, 0, 1, 0, -1, 0, 0],
            ]
        )

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        assert np.allclose(M, M_known, rtol, atol)

    def test_2d_anisotropic_permeability_cart(self):
        sd = self._create_cartesian_grid(dimension=2)

        kxx = np.square(sd.cell_centers[1, :]) + 1
        kyy = np.square(sd.cell_centers[0, :]) + 1
        kxy = -np.multiply(sd.cell_centers[0, :], sd.cell_centers[1, :])
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kxy=kxy)

        bf = sd.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])

        M = self._matrix(sd, perm, bc)
        # Matrix computed with an already validated code (MRST)
        M_known = np.array(
            [
                [
                    0.625000000000000,
                    -0.422619047619048,
                    0,
                    0.023809523809524,
                    0,
                    0.023809523809524,
                    0,
                    1,
                    0,
                ],
                [
                    -0.422619047619048,
                    1.267241379310345,
                    -0.426724137931035,
                    0.023809523809524,
                    0.051724137931034,
                    0.023809523809524,
                    0.051724137931034,
                    -1,
                    1,
                ],
                [
                    0,
                    -0.426724137931035,
                    0.642241379310345,
                    0,
                    0.051724137931034,
                    0,
                    0.051724137931034,
                    0,
                    -1,
                ],
                [
                    0.023809523809524,
                    0.023809523809524,
                    0,
                    1,
                    0,
                    -0.047619047619048,
                    0,
                    1,
                    0,
                ],
                [
                    0,
                    0.051724137931034,
                    0.051724137931034,
                    0,
                    0.879310344827586,
                    0,
                    -0.189655172413793,
                    0,
                    1,
                ],
                [
                    0.023809523809524,
                    0.023809523809524,
                    0,
                    -0.047619047619048,
                    0,
                    1,
                    0,
                    -1,
                    0,
                ],
                [
                    0,
                    0.051724137931034,
                    0.051724137931034,
                    0,
                    -0.189655172413793,
                    0,
                    0.879310344827586,
                    0,
                    -1,
                ],
                [1, -1, 0, 1, 0, -1, 0, 0, 0],
                [0, 1, -1, 0, 1, 0, -1, 0, 0],
            ]
        )

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        assert np.allclose(M, M_known, rtol, atol)

    def test_3d_isotropic_permeability_cart(self):
        sd = self._create_cartesian_grid(dimension=3)

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=kxx)

        bf = sd.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])

        M = self._matrix(sd, perm, bc)
        M_known = reference_dense_arrays.test_dual_vem[
            "test_3d_isotropic_permeability_cart"
        ]["M_ref"]
        # M_known = matrix_for_test_dual_vem_3d_iso_cart()

        rtol = 1e-14
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        assert np.allclose(M, M_known, rtol, atol)

    def test_3d_anisotropic_permeability_cart(self):
        sd = self._create_cartesian_grid(dimension=3)

        kxx = np.square(sd.cell_centers[1, :]) + 1
        kyy = np.square(sd.cell_centers[0, :]) + 1
        kzz = sd.cell_centers[2, :] + 1
        kxy = -np.multiply(sd.cell_centers[0, :], sd.cell_centers[1, :])
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kxy=kxy, kzz=kzz)

        bf = sd.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])

        M = self._matrix(sd, perm, bc)
        M_known = reference_dense_arrays.test_dual_vem[
            "test_3d_anisotropic_permeability_cart"
        ]["M_ref"]

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        assert np.allclose(M, M_known, rtol, atol)

    def test_2d_isotropic_permeability_simplex(self):
        sd = pp.StructuredTriangleGrid([1, 1], [1, 1])
        sd.compute_geometry()

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx)

        bf = sd.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])

        M = self._matrix(sd, perm, bc)
        # Matrix computed with an already validated code (MRST)
        faces = np.arange(5)
        M_known = np.array(
            [
                [0.611111111111111, 0.0, -0.277777777777778, 0.111111111111111, 0.0],
                [0.0, 0.611111111111111, -0.277777777777778, 0.0, 0.111111111111111],
                [
                    -0.277777777777778,
                    -0.277777777777778,
                    0.888888888888889,
                    -0.277777777777778,
                    -0.277777777777778,
                ],
                [0.111111111111111, 0.0, -0.277777777777778, 0.611111111111111, 0.0],
                [0.0, 0.111111111111111, -0.277777777777778, 0.0, 0.611111111111111],
            ]
        )

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        # We test only the mass-Hdiv part
        assert np.allclose(
            M[np.ix_(faces, faces)],
            M_known,
            rtol,
            atol,
        )

    def test_2d_anisotropic_permeability_simplex(self):
        sd = pp.StructuredTriangleGrid([1, 1], [1, 1])
        sd.compute_geometry()

        kxx = np.square(sd.cell_centers[1, :]) + 1
        kyy = np.square(sd.cell_centers[0, :]) + 1
        kxy = -np.multiply(sd.cell_centers[0, :], sd.cell_centers[1, :])
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kxy=kxy, kzz=1)

        bf = sd.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])

        M = self._matrix(sd, perm, bc)
        # Matrix computed with an already validated code (MRST)
        faces = np.arange(5)
        M_known = np.array(
            [
                [0.599206349206349, 0.0, -0.337301587301587, 0.134920634920635, 0.0],
                [0.0, 0.599206349206349, -0.337301587301587, 0.0, 0.134920634920635],
                [
                    -0.337301587301587,
                    -0.337301587301587,
                    0.865079365079365,
                    -0.301587301587302,
                    -0.301587301587302,
                ],
                [0.134920634920635, 0.0, -0.301587301587302, 0.634920634920635, 0.0],
                [0.0, 0.134920634920635, -0.301587301587302, 0.0, 0.634920634920635],
            ]
        )

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        # We test only the mass-Hdiv part
        assert np.allclose(
            M[np.ix_(faces, faces)],
            M_known,
            rtol,
            atol,
        )

    def test_2d_isotropic_permeability_simplex_mixed_bc(self):
        sd = pp.StructuredTriangleGrid([2, 2], [1, 1])
        sd.compute_geometry()

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx)

        tol = 1e-6
        bf = sd.tags["domain_boundary_faces"].nonzero()[0]
        bf_centers = sd.face_centers[:, bf]
        left = bf_centers[0, :] < 0 + tol
        right = bf_centers[0, :] > 1 - tol

        labels = np.array(["neu"] * bf.size)
        labels[left] = "rob"
        labels[right] = "dir"

        bc = pp.BoundaryCondition(sd, bf, labels)
        bc.robin_weight[bf[left]] = 2

        bc_val = np.zeros(sd.num_faces)
        bc_val[bf[left]] = 3

        solver = pp.MVEM(keyword="flow")

        data = pp.initialize_default_data(
            sd, {}, "flow", {"second_order_tensor": perm, "bc": bc, "bc_values": bc_val}
        )
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
        r_known = reference_dense_arrays.test_dual_vem[
            "test_2d_isotropic_permeability_simplex_mixed_bc"
        ]["rows"]
        c_known = reference_dense_arrays.test_dual_vem[
            "test_2d_isotropic_permeability_simplex_mixed_bc"
        ]["cols"]
        d_known = reference_dense_arrays.test_dual_vem[
            "test_2d_isotropic_permeability_simplex_mixed_bc"
        ]["data"]

        assert np.allclose(r, r_known)
        assert np.allclose(c, c_known)
        assert np.allclose(d, d_known)

    def test_1d_R1_R3_isotropic_permeability(self):
        sd = self._create_cartesian_grid(dimension=1)
        R = pp.map_geometry.rotation_matrix(np.pi / 6.0, [0, 0, 1])
        sd.nodes = np.dot(R, sd.nodes)
        sd.compute_geometry()

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx, kyy=1, kzz=1)
        perm.rotate(R)

        bf = sd.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])

        M = self._matrix(sd, perm, bc)
        # Matrix computed with an already validated code (MRST)
        M_known = 1e-2 * np.array(
            [
                [25, -(8 + 1 / 3.0), 0, 0, 1e2, 0, 0],
                [-(8 + 1 / 3.0), 50, -(8 + 1 / 3.0), 0, -1e2, 1e2, 0],
                [0, -(8 + 1 / 3.0), 50, -(8 + 1 / 3.0), 0, -1e2, 1e2],
                [0, 0, -(8 + 1 / 3.0), 25, 0, 0, -1e2],
                [1e2, -1e2, 0, 0, 0, 0, 0],
                [0, 1e2, -1e2, 0, 0, 0, 0],
                [0, 0, 1e2, -1e2, 0, 0, 0],
            ]
        )

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        assert np.allclose(M, M_known, rtol, atol)

    def test_2d_R2_R3_isotropic_permeability(self):
        sd = self._create_cartesian_grid(dimension=2)
        R = pp.map_geometry.rotation_matrix(np.pi / 4.0, [0, 1, 0])
        sd.nodes = np.dot(R, sd.nodes)
        sd.compute_geometry()

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
        perm.rotate(R)

        bf = sd.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])

        M = self._matrix(sd, perm, bc)
        # Matrix computed with an already validated code (MRST)
        M_known = np.array(
            [
                [0.625, -0.375, 0, 0, 0, 0, 0, 1, 0],
                [-0.375, 1.25, -0.375, 0, 0, 0, 0, -1, 1],
                [0, -0.375, 0.625, 0, 0, 0, 0, 0, -1],
                [0, 0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1, 0, -1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, -1],
                [1, -1, 0, 1, 0, -1, 0, 0, 0],
                [0, 1, -1, 0, 1, 0, -1, 0, 0],
            ]
        )

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        assert np.allclose(M, M_known, rtol, atol)

    def test_2d_R2_R3_anisotropic_permeability(self):
        sd = self._create_cartesian_grid(dimension=2)
        sd.compute_geometry()

        kxx = np.square(sd.cell_centers[1, :]) + 1
        kyy = np.square(sd.cell_centers[0, :]) + 1
        kxy = -np.multiply(sd.cell_centers[0, :], sd.cell_centers[1, :])
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kxy=kxy, kzz=1)

        R = pp.map_geometry.rotation_matrix(np.pi / 3.0, [1, 1, 0])
        perm.rotate(R)
        sd.nodes = np.dot(R, sd.nodes)
        sd.compute_geometry()

        bf = sd.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])

        M = self._matrix(sd, perm, bc)
        # Matrix computed with an already validated code (MRST)
        M_known = np.array(
            [
                [
                    0.625000000000000,
                    -0.422619047619048,
                    0,
                    0.023809523809524,
                    0,
                    0.023809523809524,
                    0,
                    1,
                    0,
                ],
                [
                    -0.422619047619048,
                    1.267241379310345,
                    -0.426724137931035,
                    0.023809523809524,
                    0.051724137931034,
                    0.023809523809524,
                    0.051724137931034,
                    -1,
                    1,
                ],
                [
                    0,
                    -0.426724137931035,
                    0.642241379310345,
                    0,
                    0.051724137931034,
                    0,
                    0.051724137931034,
                    0,
                    -1,
                ],
                [
                    0.023809523809524,
                    0.023809523809524,
                    0,
                    1,
                    0,
                    -0.047619047619048,
                    0,
                    1,
                    0,
                ],
                [
                    0,
                    0.051724137931034,
                    0.051724137931034,
                    0,
                    0.879310344827586,
                    0,
                    -0.189655172413793,
                    0,
                    1,
                ],
                [
                    0.023809523809524,
                    0.023809523809524,
                    0,
                    -0.047619047619048,
                    0,
                    1,
                    0,
                    -1,
                    0,
                ],
                [
                    0,
                    0.051724137931034,
                    0.051724137931034,
                    0,
                    -0.189655172413793,
                    0,
                    0.879310344827586,
                    0,
                    -1,
                ],
                [1, -1, 0, 1, 0, -1, 0, 0, 0],
                [0, 1, -1, 0, 1, 0, -1, 0, 0],
            ]
        )

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        assert np.allclose(M, M_known, rtol, atol)

    def test_2d_R2_R3_isotropic_permeability_simplex(self):
        sd = pp.StructuredTriangleGrid([1, 1], [1, 1])
        R = pp.map_geometry.rotation_matrix(-np.pi / 4.0, [1, 1, -1])
        sd.nodes = np.dot(R, sd.nodes)
        sd.compute_geometry()

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
        perm.rotate(R)

        bf = sd.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])

        M = self._matrix(sd, perm, bc)
        # Matrix computed with an already validated code (MRST)
        faces = np.arange(5)
        M_known = np.array(
            [
                [0.611111111111111, 0.0, -0.277777777777778, 0.111111111111111, 0.0],
                [0.0, 0.611111111111111, -0.277777777777778, 0.0, 0.111111111111111],
                [
                    -0.277777777777778,
                    -0.277777777777778,
                    0.888888888888889,
                    -0.277777777777778,
                    -0.277777777777778,
                ],
                [0.111111111111111, 0.0, -0.277777777777778, 0.611111111111111, 0.0],
                [0.0, 0.111111111111111, -0.277777777777778, 0.0, 0.611111111111111],
            ]
        )

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        # We test only the mass-Hdiv part
        assert np.allclose(
            M[np.ix_(faces, faces)],
            M_known,
            rtol,
            atol,
        )

    def test_2d_R2_R3_anisotropic_permeability_simplex(self):
        sd = pp.StructuredTriangleGrid([1, 1], [1, 1])
        sd.compute_geometry()

        kxx = np.square(sd.cell_centers[1, :]) + 1
        kyy = np.square(sd.cell_centers[0, :]) + 1
        kxy = -np.multiply(sd.cell_centers[0, :], sd.cell_centers[1, :])
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kxy=kxy, kzz=1)

        R = pp.map_geometry.rotation_matrix(np.pi / 3.0, [1, 1, 0])
        perm.rotate(R)
        sd.nodes = np.dot(R, sd.nodes)
        sd.compute_geometry()

        bf = sd.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])

        M = self._matrix(sd, perm, bc)
        # assemble_matrix_rhs computed with an already validated code (MRST)
        faces = np.arange(5)
        M_known = np.array(
            [
                [0.59920634920635, 0.0, -0.337301587301588, 0.134920634920635, 0.0],
                [0.0, 0.599206349206349, -0.337301587301588, 0.0, 0.134920634920635],
                [
                    -0.337301587301588,
                    -0.337301587301588,
                    0.865079365079365,
                    -0.301587301587301,
                    -0.301587301587302,
                ],
                [0.134920634920635, 0.0, -0.301587301587301, 0.634920634920635, 0.0],
                [0.0, 0.134920634920635, -0.301587301587302, 0.0, 0.634920634920634],
            ]
        )

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        # We test only the mass-Hdiv part
        assert np.allclose(
            M[np.ix_(faces, faces)],
            M_known,
            rtol,
            atol,
        )


class TestMVEMRHS:
    def _rhs(self, sd, perm, bc, vect):
        """Compute rhs for given subdomain"""
        solver = pp.MVEM(keyword="flow")

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
        sd = pp.CartGrid(3, 1)
        sd.compute_geometry()

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

        assert np.allclose(b, b_known)

    """Test two-dimensional for isotropic material data."""

    def test_2d_isotropic_permeability(self):
        sd = pp.StructuredTriangleGrid([1, 1], [1, 1])
        sd.compute_geometry()

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

        assert np.allclose(b, b_known)

    """Test three-dimensional for isotropic material data."""

    def test_3d_isotropic_permeability(self):
        sd = pp.StructuredTetrahedralGrid([1, 1, 1], [1, 1, 1])
        sd.compute_geometry()

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

        assert np.allclose(b, b_known)

    """Test for single 2d-cell with isotropic material data."""

    def test_single_triangle_discretization(self):
        dim = 2
        nodes = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

        indices = [0, 1, 1, 2, 2, 0]
        indptr = [0, 2, 4, 6]
        face_nodes = sps.csc_matrix(([True] * 6, indices, indptr))
        cell_faces = sps.csc_matrix([[1], [1], [1]])
        name = "test"

        sd = pp.Grid(dim, nodes, face_nodes, cell_faces, name)
        sd.compute_geometry()

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)

        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        vect = np.vstack(
            (sd.cell_volumes, 3 * sd.cell_volumes, np.zeros(sd.num_cells))
        ).ravel(order="F")

        b = self._rhs(sd, perm, bc, vect)
        b_known = np.array([-0.83333333, 0.66666667, 0.16666667, 0.0])

        assert np.allclose(b, b_known)

    """Test for single 3d-cell with isotropic material data."""

    def test_single_tetrahedron_discretization(self):
        dim = 3
        nodes = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        indices = [0, 2, 1, 1, 2, 3, 2, 0, 3, 3, 0, 1]
        indptr = [0, 3, 6, 9, 12]
        face_nodes = sps.csc_matrix(([True] * 12, indices, indptr))
        cell_faces = sps.csc_matrix([[1], [1], [1], [1]])
        name = "test"

        sd = pp.Grid(dim, nodes, face_nodes, cell_faces, name)
        sd.compute_geometry()

        kxx = np.ones(sd.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=kxx)

        bf = sd.get_boundary_faces()
        bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
        vect = np.vstack(
            (2 * sd.cell_volumes, 3 * sd.cell_volumes, np.zeros(sd.num_cells))
        ).ravel(order="F")

        b = self._rhs(sd, perm, bc, vect)
        b_known = np.array([0.41666667, 0.41666667, -0.25, -0.58333333, 0.0])

        assert np.allclose(b, b_known)

    """Test one-dimensional R1-R3 for isotropic material data."""

    def test_1d_R1_R3_isotropic_permeability(self):
        sd = pp.CartGrid(3, 1)
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

        assert np.allclose(b, b_known)

    """Test two-dimensional R2-R3 for isotropic material data."""

    def test_2d_R2_R3_isotropic_permeability(self):
        sd = pp.StructuredTriangleGrid([1, 1], [1, 1])
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

        assert np.allclose(b, b_known)

    # TODO: It checks for a functionality is already tested. This should be deleted.
    def test_convergence_mvem_2d_iso_simplex_exact(self):
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

            solver = pp.MVEM(keyword="flow")

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

    # TODO: It checks for a functionality is already tested. This should be deleted.
    def test_convergence_mvem_2d_iso_simplex(self):
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
            sd = pp.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            sd.compute_geometry()

            kxx = np.ones(sd.num_cells)
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
            bf = sd.get_boundary_faces()
            bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
            bc_val = np.zeros(sd.num_faces)
            bc_val[bf] = p_ex(sd.face_centers[:, bf])
            # Minus sign to move to rhs
            source = np.multiply(sd.cell_volumes, rhs_ex(sd.cell_centers))
            vect = np.vstack(
                (sd.cell_volumes, np.zeros(sd.num_cells), np.zeros(sd.num_cells))
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

    # TODO: It checks for a functionality is already tested. This should be deleted.
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
            sd = pp.StructuredTriangleGrid([3 + i] * 2, [1, 1])
            sd.compute_geometry()

            kxx = 2 * np.ones(sd.num_cells)
            kxy = np.ones(sd.num_cells)
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kxy=kxy, kzz=1)
            bf = sd.get_boundary_faces()
            bc = pp.BoundaryCondition(sd, bf, bf.size * ["dir"])
            bc_val = np.zeros(sd.num_faces)
            bc_val[bf] = p_ex(sd.face_centers[:, bf])
            # Minus sign to move to rhs
            source = np.multiply(sd.cell_volumes, rhs_ex(sd.cell_centers))
            vect = np.vstack(
                (sd.cell_volumes, 2 * sd.cell_volumes, np.zeros(sd.num_cells))
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


"""Test known convergence rate for 2d domain with varying permeability"""


class TestVEMConvergence:
    def _rhs(self, x, y, z, case):
        if case == 1:
            return (
                8.0
                * np.pi**2
                * np.sin(2.0 * np.pi * x)
                * np.sin(2.0 * np.pi * y)
                * (1 + 100.0 * x**2 + 100.0 * y**2)
                - 400.0 * np.pi * y * np.cos(2.0 * np.pi * y) * np.sin(2.0 * np.pi * x)
                - 400.0 * np.pi * x * np.cos(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
            )
        elif case == 2:
            return (
                7.0 * z * (x**2 + y**2 + 1.0)
                - y * (x**2 - 9.0 * z**2)
                - 4.0 * x**2 * z
                - (
                    8.0 * np.sin(np.pi * y)
                    - 4.0 * np.pi**2 * y**2 * np.sin(np.pi * y)
                    + 16.0 * np.pi * y * np.cos(np.pi * y)
                )
                * (x**2 / 2.0 + y**2 / 2.0 + 1.0 / 2.0)
                - 4.0
                * y**2
                * (2.0 * np.sin(np.pi * y) + np.pi * y * np.cos(np.pi * y))
            )
        else:
            return 8.0 * z * (125.0 * x**2 + 200.0 * y**2 + 425.0 * z**2 + 2.0)

    def _solution(self, x, y, z, case):
        if case == 1:
            return np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
        elif case == 2:
            return x**2 * z + 4.0 * y**2 * np.sin(np.pi * y) - 3.0 * z**3
        else:
            return x**2 * z + 4.0 * y**2 * np.sin(np.pi * y) - 3.0 * z**3

    def _permeability(self, x, y, z, case):
        if case == 1:
            return 1 + 100.0 * x**2 + 100.0 * y**2
        elif case == 2:
            return 1.0 + x**2 + y**2
        else:
            return 1.0 + 100.0 * (x**2 + y**2 + z**2.0)

    def _expected_order(self, case):
        if case == 1:
            return 2.00266229752
        elif case == 2:
            return 1.97928213116
        else:
            return 1.9890160655

    def _create_grid(self, N, case):
        Nx = Ny = N
        if case == 1:
            sd = pp.StructuredTriangleGrid([Nx, Ny], [1, 1])
            sd.compute_geometry()
        elif case == 2:
            sd = pp.StructuredTriangleGrid([Nx, Ny], [1, 1])
            R = pp.map_geometry.rotation_matrix(np.pi / 4.0, [1, 0, 0])
            sd.nodes = np.dot(R, sd.nodes)
            sd.compute_geometry()
        else:
            sd = pp.StructuredTriangleGrid([Nx, Ny], [1, 1])
            R = pp.map_geometry.rotation_matrix(np.pi / 2.0, [1, 0, 0])
            sd.nodes = np.dot(R, sd.nodes)
            sd.compute_geometry()
        return sd

    def _assign_parameters(self, sd, case):
        """
        Define the permeability, apertures, boundary conditions
        """
        # Permeability
        kxx = np.array([self._permeability(*pt, case) for pt in sd.cell_centers.T])
        perm = pp.SecondOrderTensor(kxx)

        # Source term
        source = sd.cell_volumes * np.array(
            [self._rhs(*pt, case) for pt in sd.cell_centers.T]
        )

        # Boundaries
        bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
        bound_face_centers = sd.face_centers[:, bound_faces]

        labels = np.array(["dir"] * bound_faces.size)

        bc_val = np.zeros(sd.num_faces)
        bc_val[bound_faces] = np.array(
            [self._solution(*pt, case) for pt in bound_face_centers.T]
        )

        bound = pp.BoundaryCondition(sd, bound_faces, labels)
        specified_parameters = {
            "second_order_tensor": perm,
            "source": source,
            "bc": bound,
            "bc_values": bc_val,
        }
        return pp.initialize_default_data(sd, {}, "flow", specified_parameters)

    def _error_p(self, sd, p, case):
        sol = np.array([self._solution(*pt, case) for pt in sd.cell_centers.T])
        return np.sqrt(np.sum(np.power(np.abs(p - sol), 2) * sd.cell_volumes))

    def _compute_approximation(self, N, case):
        sd = self._create_grid(N, case)
        data = self._assign_parameters(sd, case)

        # Choose and define the solvers
        solver_flow = pp.MVEM("flow")
        solver_flow.discretize(sd, data)
        A_flow, b_flow = solver_flow.assemble_matrix_rhs(sd, data)

        solver_source = pp.DualScalarSource("flow")
        solver_source.discretize(sd, data)
        A_source, b_source = solver_source.assemble_matrix_rhs(sd, data)

        up = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)
        p = solver_flow.extract_pressure(sd, up, data)

        diam = np.amax(sd.cell_diameters())
        return diam, self._error_p(sd, p, case)

    # Cases description:
    # 1) Variable permeability
    # 2) Variable permeability with Tilted Grid with plane rotation of Pi / 4.0
    # 3) Variable permeability with Tilted Grid with plane rotation of Pi / 2.0
    test_cases = [1, 2, 3]

    @pytest.mark.parametrize("case", test_cases)
    def test_expected_conv_rate(self, case):
        diam_10, error_10 = self._compute_approximation(10, case)
        diam_20, error_20 = self._compute_approximation(20, case)

        known_order = self._expected_order(case)
        order = np.log(error_10 / error_20) / np.log(diam_10 / diam_20)
        assert np.isclose(order, known_order)
