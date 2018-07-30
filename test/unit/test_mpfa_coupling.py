from __future__ import division
import numpy as np
import scipy.sparse as sps
import unittest
from porepy.fracs import meshing
import porepy.utils.comp_geom as cg
from porepy.params import bc, tensor
from porepy.params.data import Parameters

from porepy.numerics.fv import mpfa, tpfa
from porepy.numerics.mixed_dim import coupler


# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#

    def atest_mpfa_coupling_2d_1d_bottom_top_dir(self):
        """
        Grid: 2 x 2 matrix + 2 x 1 fracture from left to right.
        Dirichlet + no-flow, blocking fracture.
        """

        f = np.array([[0, 1], [.5, .5]])
        gb = meshing.cart_grid([f], [1, 2], **{"physdims": [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = mpfa.Mpfa(physics="flow")
        gb.add_node_props(["param"])
        a = 1e-2
        for g, d in gb:
            param = Parameters(g)

            aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
            param.set_aperture(aperture)

            p = tensor.SecondOrderTensor(
                3, np.ones(g.num_cells) * np.power(1e-3, g.dim < gb.dim_max())
            )
            param.set_tensor("flow", p)
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[1, :] > 1 - tol
            bottom = bound_face_centers[1, :] < tol

            labels = np.array(["neu"] * bound_faces.size)
            labels[np.logical_or(top, bottom)] = ["dir"]

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(top, bottom)]
            bc_val[bc_dir] = g.face_centers[1, bc_dir]

            param.set_bc(solver, bc.BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val(solver, bc_val)

            d["param"] = param

        coupling_conditions = tpfa.TpfaCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling_conditions)
        A, rhs = solver_coupler.matrix_rhs(gb)

        A_known = np.array(
            [
                [4.19047619, 0., -0.19047619],
                [0., 4.19047619, -0.19047619],
                [-0.19047619, -0.19047619, 0.38095238],
            ]
        )

        rhs_known = np.array([0., 4., 0.])

        rtol = 1e-6
        atol = rtol

        assert np.allclose(A.todense(), A_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def atest_mpfa_coupling_2d_1d_bottom_top_dir_neu(self):
        """
        Grid: 1 x 2 cells in matrix + 1 cell in the fracture from left to right.
        Dirichlet + inflow + no-flow, blocking fracture.
        """
        f = np.array([[0, 1], [.5, .5]])
        gb = meshing.cart_grid([f], [1, 2], **{"physdims": [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = mpfa.Mpfa(physics="flow")
        gb.add_node_props(["param"])
        a = 1e-2
        for g, d in gb:
            param = Parameters(g)

            a_dim = np.power(a, gb.dim_max() - g.dim)
            aperture = np.ones(g.num_cells) * a_dim
            param.set_aperture(aperture)

            p = tensor.SecondOrderTensor(
                3, np.ones(g.num_cells) * np.power(1e-3, g.dim < gb.dim_max())
            )
            param.set_tensor("flow", p)
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[1, :] > 1 - tol
            bottom = bound_face_centers[1, :] < tol

            labels = np.array(["neu"] * bound_faces.size)
            labels[bottom] = ["dir"]

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[bottom]
            bc_neu = bound_faces[top]
            bc_val[bc_dir] = g.face_centers[1, bc_dir]
            bc_val[bc_neu] = -g.face_areas[bc_neu] * a_dim

            param.set_bc(solver, bc.BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val(solver, bc_val)

            d["param"] = param

        coupling_conditions = tpfa.TpfaCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling_conditions)
        A, rhs = solver_coupler.matrix_rhs(gb)

        A_known = np.array(
            [
                [4.19047619, 0., -0.19047619],
                [0., 0.19047619, -0.19047619],
                [-0.19047619, -0.19047619, 0.38095238],
            ]
        )

        rhs_known = np.array([0, 1, 0])

        rtol = 1e-6
        atol = rtol

        assert np.allclose(A.todense(), A_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def atest_mpfa_coupling_2d_1d_left_right_dir(self):
        """
        Grid: 2 x 2 cells in matrix + 2 cells in the fracture from left to right.
        Dirichlet + no-flow, conductive fracture.
        """
        f = np.array([[0, 1], [.5, .5]])
        gb = meshing.cart_grid([f], [2, 2], **{"physdims": [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = mpfa.Mpfa(physics="flow")
        gb.add_node_props(["param"])
        a = 1e-2
        for g, d in gb:
            param = Parameters(g)

            aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
            param.set_aperture(aperture)

            p = tensor.SecondOrderTensor(
                3, np.ones(g.num_cells) * np.power(1e3, g.dim < gb.dim_max())
            )
            param.set_tensor("flow", p)
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = g.face_centers[:, bound_faces]

                left = bound_face_centers[0, :] > 1 - tol
                right = bound_face_centers[0, :] < tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[np.logical_or(left, right)] = ["dir"]

                bc_val = np.zeros(g.num_faces)
                bc_dir = bound_faces[np.logical_or(left, right)]
                bc_val[bc_dir] = g.face_centers[0, bc_dir]

                param.set_bc(solver, bc.BoundaryCondition(g, bound_faces, labels))
                param.set_bc_val(solver, bc_val)
            else:
                param.set_bc("flow", bc.BoundaryCondition(g, np.empty(0), np.empty(0)))
            d["param"] = param

        coupling_conditions = tpfa.TpfaCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling_conditions)
        A, rhs = solver_coupler.matrix_rhs(gb)

        A_known = np.array(
            [
                [4.99996, -1., 0., 0., 0., -1.99996],
                [-1., 4.99996, 0., 0., -1.99996, 0.],
                [0., 0., 4.99996, -1., 0., -1.99996],
                [0., 0., -1., 4.99996, -1.99996, 0.],
                [0., -1.99996, 0., -1.99996, 63.99992, -20.],
                [-1.99996, 0., -1.99996, 0., -20., 63.99992],
            ]
        )

        rhs_known = np.array([0., 2., 0., 2., 40., 0.])

        rtol = 1e-6
        atol = rtol

        assert np.allclose(A.todense(), A_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def atest_mpfa_coupling_2d_1d_left_right_dir_neu(self):
        """
        Grid: 2 x 2 cells in matrix + 2 cells in the fracture from left to right.
        Dirichlet + inflow + no-flow, conductive fracture.
        Tests pressure solution as well as matrix and rhs.
        """
        f = np.array([[0, 1], [.5, .5]])
        gb = meshing.cart_grid([f], [2, 2], **{"physdims": [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = mpfa.Mpfa(physics="flow")
        gb.add_node_props(["param"])
        a = 1e-2
        for g, d in gb:
            param = Parameters(g)

            a_dim = np.power(a, gb.dim_max() - g.dim)
            aperture = np.ones(g.num_cells) * a_dim
            param.set_aperture(aperture)

            p = tensor.SecondOrderTensor(
                3, np.ones(g.num_cells) * np.power(1e3, g.dim < gb.dim_max())
            )
            param.set_tensor("flow", p)
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            bound_face_centers = g.face_centers[:, bound_faces]

            right = bound_face_centers[0, :] > 1 - tol
            left = bound_face_centers[0, :] < tol

            labels = np.array(["neu"] * bound_faces.size)
            labels[right] = ["dir"]

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[right]
            bc_neu = bound_faces[left]
            bc_val[bc_dir] = g.face_centers[0, bc_dir]
            bc_val[bc_neu] = -g.face_areas[bc_neu] * a_dim

            param.set_bc(solver, bc.BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val(solver, bc_val)

            d["param"] = param

        coupling_conditions = tpfa.TpfaCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling_conditions)
        A, rhs = solver_coupler.matrix_rhs(gb)

        A_known = np.array(
            [
                [2.99996, -1., 0., 0., 0., -1.99996],
                [-1., 4.99996, 0., 0., -1.99996, 0.],
                [0., 0., 2.99996, -1., 0., -1.99996],
                [0., 0., -1., 4.99996, -1.99996, 0.],
                [0., -1.99996, 0., -1.99996, 63.99992, -20.],
                [-1.99996, 0., -1.99996, 0., -20., 23.99992],
            ]
        )

        rhs_known = np.array(
            [
                5.00000000e-01,
                2.00000000e+00,
                5.00000000e-01,
                2.00000000e+00,
                4.00000000e+01,
                1.00000000e-02,
            ]
        )
        p_known = np.array(
            [1.21984244, 1.05198918, 1.21984244, 1.05198918, 1.02005108, 1.05376576]
        )

        p = sps.linalg.spsolve(A, rhs)

        rtol = 1e-6
        atol = rtol

        assert np.allclose(A.todense(), A_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)
        assert np.allclose(p, p_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def atest_mpfa_coupling_2d_1d_left_right_cross_dir_neu(self):
        f1 = np.array([[0, 2], [.5, .5]])
        f2 = np.array([[.5, .5], [0, 2]])

        gb = meshing.cart_grid([f1, f2], [2, 2], **{"physdims": [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        # Enforce node orderning because of Python 3.5 and 2.7.
        # Don't do it in general.
        cell_centers_1 = np.array(
            [
                [7.50000000e-01, 2.500000000e-01],
                [5.00000000e-01, 5.00000000e-01],
                [-5.55111512e-17, 5.55111512e-17],
            ]
        )
        cell_centers_2 = np.array(
            [
                [5.00000000e-01, 5.00000000e-01],
                [7.50000000e-01, 2.500000000e-01],
                [-5.55111512e-17, 5.55111512e-17],
            ]
        )

        for g, d in gb:
            if g.dim == 1:
                if np.allclose(g.cell_centers, cell_centers_1):
                    d["node_number"] = 1
                elif np.allclose(g.cell_centers, cell_centers_2):
                    d["node_number"] = 2
                else:
                    raise ValueError("Grid not found")

        tol = 1e-3
        solver = mpfa.Mpfa()
        gb.add_node_props(["param"])
        a = 1e-2
        for g, d in gb:
            param = Parameters(g)

            a_dim = np.power(a, gb.dim_max() - g.dim)
            aperture = np.ones(g.num_cells) * a_dim
            param.set_aperture(aperture)

            kxx = np.ones(g.num_cells) * np.power(1e3, g.dim < gb.dim_max())

            p = tensor.SecondOrderTensor(3, kxx, kyy=kxx, kzz=kxx)

            param.set_tensor("flow", p)
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = g.face_centers[:, bound_faces]

                right = bound_face_centers[0, :] > 1 - tol
                left = bound_face_centers[0, :] < tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[right] = ["dir"]

                bc_val = np.zeros(g.num_faces)
                bc_dir = bound_faces[right]
                bc_neu = bound_faces[left]
                bc_val[bc_dir] = g.face_centers[0, bc_dir]
                bc_val[bc_neu] = -g.face_areas[bc_neu] * a_dim

                param.set_bc(solver, bc.BoundaryCondition(g, bound_faces, labels))
                param.set_bc_val(solver, bc_val)
            else:
                param.set_bc("flow", bc.BoundaryCondition(g, np.empty(0), np.empty(0)))
            d["param"] = param

        coupling_conditions = tpfa.TpfaCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling_conditions)
        A, rhs = solver_coupler.matrix_rhs(gb)

        A_known, rhs_known = matrix_rhs_for_2d_1d_cross()

        rtol = 1e-6
        atol = rtol

        assert np.allclose(A.todense(), A_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def atest_mpfa_coupling_3d_2d_1d_0d_dir(self):
        f1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [.5, .5, .5, .5]])
        f2 = np.array([[.5, .5, .5, .5], [0, 1, 1, 0], [0, 0, 1, 1]])
        f3 = np.array([[0, 1, 1, 0], [.5, .5, .5, .5], [0, 0, 1, 1]])

        gb = meshing.cart_grid([f1, f2, f3], [2, 2, 2], **{"physdims": [1, 1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()
        # Remove flag for dual
        cell_centers1 = np.array(
            [[0.25, 0.75, 0.25, 0.75], [0.25, 0.25, 0.75, 0.75], [0.5, 0.5, 0.5, 0.5]]
        )
        cell_centers2 = np.array(
            [[0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75], [0.75, 0.25, 0.75, 0.25]]
        )
        cell_centers3 = np.array(
            [[0.25, 0.75, 0.25, 0.75], [0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]]
        )
        cell_centers4 = np.array([[0.5], [0.25], [0.5]])
        cell_centers5 = np.array([[0.5], [0.75], [0.5]])
        cell_centers6 = np.array([[0.75], [0.5], [0.5]])
        cell_centers7 = np.array([[0.25], [0.5], [0.5]])
        cell_centers8 = np.array([[0.5], [0.5], [0.25]])
        cell_centers9 = np.array([[0.5], [0.5], [0.75]])

        for g, d in gb:
            if np.allclose(g.cell_centers[:, 0], cell_centers1[:, 0]):
                d["node_number"] = 1
            elif np.allclose(g.cell_centers[:, 0], cell_centers2[:, 0]):
                d["node_number"] = 2
            elif np.allclose(g.cell_centers[:, 0], cell_centers3[:, 0]):
                d["node_number"] = 3
            elif np.allclose(g.cell_centers[:, 0], cell_centers4[:, 0]):
                d["node_number"] = 4
            elif np.allclose(g.cell_centers[:, 0], cell_centers5[:, 0]):
                d["node_number"] = 5
            elif np.allclose(g.cell_centers[:, 0], cell_centers6[:, 0]):
                d["node_number"] = 6
            elif np.allclose(g.cell_centers[:, 0], cell_centers7[:, 0]):
                d["node_number"] = 7
            elif np.allclose(g.cell_centers[:, 0], cell_centers8[:, 0]):
                d["node_number"] = 8
            elif np.allclose(g.cell_centers[:, 0], cell_centers9[:, 0]):
                d["node_number"] = 9
            else:
                pass

        tol = 1e-3
        solver = mpfa.Mpfa()
        gb.add_node_props(["param"])

        a = 1e-2
        for g, d in gb:
            param = Parameters(g)

            aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
            param.set_aperture(aperture)

            p = tensor.SecondOrderTensor(
                3, np.ones(g.num_cells) * np.power(1e3, g.dim < gb.dim_max())
            )
            param.set_tensor("flow", p)
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = g.face_centers[:, bound_faces]

                left = bound_face_centers[0, :] > 1 - tol
                right = bound_face_centers[0, :] < tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[np.logical_or(left, right)] = ["dir"]

                bc_val = np.zeros(g.num_faces)
                bc_dir = bound_faces[np.logical_or(left, right)]
                bc_val[bc_dir] = g.face_centers[0, bc_dir]

                param.set_bc(solver, bc.BoundaryCondition(g, bound_faces, labels))
                param.set_bc_val(solver, bc_val)
            else:
                param.set_bc("flow", bc.BoundaryCondition(g, np.empty(0), np.empty(0)))
            d["param"] = param

        coupling_conditions = tpfa.TpfaCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling_conditions)
        A, rhs = solver_coupler.matrix_rhs(gb)

        A_known, rhs_known, p_known = (
            matrix_rhs_pressure_for_test_mpfa_coupling_3d_2d_1d_0d()
        )

        p = sps.linalg.spsolve(A, rhs)

        rtol = 1e-6
        atol = rtol

        assert np.allclose(A.todense(), A_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)
        assert np.allclose(p, p_known, rtol, atol)


# ------------------------------------------------------------------------------#


def amatrix_rhs_pressure_for_test_mpfa_coupling_3d_2d_1d_0d():
    A = np.array(
        [
            [
                3.99994,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                -0.99998,
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
                3.99994,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                -0.99998,
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
                3.99994,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                0.,
                -0.99998,
                -0.99998,
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
                3.99994,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                -0.99998,
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
                3.99994,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                0.,
                0.,
                -0.99998,
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
                3.99994,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -0.99998,
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
                3.99994,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                -0.99998,
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
                3.99994,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
            ],
            [
                -0.99998,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                61.21564628,
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
                -19.60784314,
                0.,
                0.,
                -19.60784314,
                0.,
                0.,
                0.,
            ],
            [
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                61.21564628,
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
                -19.60784314,
                0.,
                -19.60784314,
                0.,
                0.,
                0.,
                0.,
            ],
            [
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                61.21564628,
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
                -19.60784314,
                0.,
                -19.60784314,
                0.,
                0.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                0.,
                0.,
                61.21564628,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -19.60784314,
                -19.60784314,
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
                -0.99998,
                -0.99998,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                41.21564628,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -19.60784314,
                0.,
                0.,
                0.,
                0.,
                -19.60784314,
                0.,
            ],
            [
                -0.99998,
                -0.99998,
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
                41.21564628,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -19.60784314,
                0.,
                0.,
                0.,
                -19.60784314,
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
                -0.99998,
                -0.99998,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                41.21564628,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -19.60784314,
                0.,
                0.,
                0.,
                -19.60784314,
                0.,
            ],
            [
                0.,
                0.,
                -0.99998,
                -0.99998,
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
                41.21564628,
                0.,
                0.,
                0.,
                0.,
                0.,
                -19.60784314,
                0.,
                0.,
                -19.60784314,
                0.,
                0.,
            ],
            [
                -0.99998,
                0.,
                -0.99998,
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
                61.21564628,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -19.60784314,
                -19.60784314,
                0.,
                0.,
            ],
            [
                0.,
                -0.99998,
                0.,
                -0.99998,
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
                61.21564628,
                0.,
                0.,
                0.,
                0.,
                -19.60784314,
                0.,
                -19.60784314,
                0.,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                -0.99998,
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
                61.21564628,
                0.,
                0.,
                0.,
                0.,
                -19.60784314,
                0.,
                -19.60784314,
                0.,
            ],
            [
                0.,
                0.,
                0.,
                0.,
                0.,
                -0.99998,
                0.,
                -0.99998,
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
                61.21564628,
                0.,
                0.,
                -19.60784314,
                0.,
                0.,
                -19.60784314,
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
                -19.60784314,
                -19.60784314,
                0.,
                0.,
                -19.60784314,
                -19.60784314,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                78.82352941,
                0.,
                0.,
                0.,
                0.,
                0.,
                -0.39215686,
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
                -19.60784314,
                -19.60784314,
                0.,
                0.,
                -19.60784314,
                -19.60784314,
                0.,
                0.,
                0.,
                0.,
                0.,
                78.82352941,
                0.,
                0.,
                0.,
                0.,
                -0.39215686,
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
                -19.60784314,
                0.,
                -19.60784314,
                0.,
                0.,
                0.,
                0.,
                0.,
                -19.60784314,
                0.,
                -19.60784314,
                0.,
                0.,
                79.22352941,
                0.,
                0.,
                0.,
                -0.39215686,
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
                -19.60784314,
                0.,
                -19.60784314,
                0.,
                0.,
                0.,
                0.,
                0.,
                -19.60784314,
                0.,
                -19.60784314,
                0.,
                0.,
                0.,
                0.,
                79.22352941,
                0.,
                0.,
                -0.39215686,
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
                -19.60784314,
                0.,
                -19.60784314,
                -19.60784314,
                -19.60784314,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                78.82352941,
                0.,
                -0.39215686,
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
                -19.60784314,
                0.,
                -19.60784314,
                0.,
                0.,
                0.,
                -19.60784314,
                -19.60784314,
                0.,
                0.,
                0.,
                0.,
                0.,
                78.82352941,
                -0.39215686,
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
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -0.39215686,
                -0.39215686,
                -0.39215686,
                -0.39215686,
                -0.39215686,
                -0.39215686,
                2.35294118,
            ],
        ]
    )

    rhs = np.array(
        [
            0.,
            1.,
            0.,
            1.,
            0.,
            1.,
            0.,
            1.,
            0.,
            20.,
            0.,
            20.,
            0.,
            0.,
            0.,
            0.,
            0.,
            20.,
            0.,
            20.,
            0.,
            0.,
            0.4,
            0.,
            0.,
            0.,
            0.,
        ]
    )

    pressure = np.array(
        [
            0.24879143,
            0.75120857,
            0.24879143,
            0.75120857,
            0.24879143,
            0.75120857,
            0.24879143,
            0.75120857,
            0.24758535,
            0.75241465,
            0.24758535,
            0.75241465,
            0.5,
            0.5,
            0.5,
            0.5,
            0.24758535,
            0.75241465,
            0.24758535,
            0.75241465,
            0.5,
            0.5,
            0.75241525,
            0.24758475,
            0.5,
            0.5,
            0.5,
        ]
    )
    return A, rhs, pressure


# ------------------------------------------------------------------------------#


def matrix_rhs_for_2d_1d_cross():
    A = np.array(
        [
            [3.99992, 0., 0., 0., 0., -1.99996, 0., -1.99996, 0.],
            [0., 5.99992, 0., 0., -1.99996, 0., 0., -1.99996, 0.],
            [0., 0., 3.99992, 0., 0., -1.99996, -1.99996, 0., 0.],
            [0., 0., 0., 5.99992, -1.99996, 0., -1.99996, 0., 0.],
            [0., -1.99996, 0., -1.99996, 83.21560628, 0., 0., 0., -39.21568627],
            [-1.99996, 0., -1.99996, 0., 0., 43.21560628, 0., 0., -39.21568627],
            [0., 0., -1.99996, -1.99996, 0., 0., 43.21560628, 0., -39.21568627],
            [-1.99996, -1.99996, 0., 0., 0., 0., 0., 43.21560628, -39.21568627],
            [
                0.,
                0.,
                0.,
                0.,
                -39.21568627,
                -39.21568627,
                -39.21568627,
                -39.21568627,
                156.8627451,
            ],
        ]
    )
    rhs = np.array(
        [
            5.00000000e-01,
            2.00000000e+00,
            5.00000000e-01,
            2.00000000e+00,
            4.00000000e+01,
            1.00000000e-02,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
        ]
    )

    return A, rhs
