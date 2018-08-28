"""
Tests for the static condensation.
Darcy problems are discretized (TPFA) and then solved both with and without the
condensation.
Solutions are then compared, so failures will probably be due to the condensation
itself or reorderings in the grid buckets.
"""

from __future__ import division
import numpy as np
import scipy.sparse as sps
import unittest

from porepy.fracs import meshing
from porepy.params import bc, tensor
from porepy.params.data import Parameters
from porepy.numerics.fv import tpfa
from porepy.numerics.mixed_dim import coupler, condensation
from porepy.utils import error

# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#

    def atest_0d_elimination_2d_1d_cross(self):
        """
        Simplest case possible:
        2d case with two fractures intersecting in a single 0d grid
        at the center of the domain.
        """
        f1 = np.array([[0, 1], [.5, .5]])
        f2 = np.array([[.5, .5], [0, 1]])

        gb = meshing.cart_grid([f1, f2], [2, 2], **{"physdims": [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = tpfa.Tpfa()
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

        p = sps.linalg.spsolve(A, rhs)
        p_cond, _, _, _ = condensation.solve_static_condensation(A, rhs, gb, dim=0)

        solver_coupler.split(gb, "pressure", p)
        solver_coupler.split(gb, "p_cond", p_cond)

        tol = 1e-10
        assert (np.amax(np.absolute(p - p_cond))) < tol
        assert (
            np.sum(error.error_L2(g, d["pressure"], d["p_cond"]) for g, d in gb) < tol
        )

    # ------------------------------------------------------------------------------#

    def atest_0d_elimination_two_0d_grids(self):
        """
        2d case involving two 0d grids.
        """
        f1 = np.array([[0, 1], [.5, .5]])
        f2 = np.array([[.5, .5], [0, 1]])
        f3 = np.array([[.25, .25], [0, 1]])

        gb = meshing.cart_grid([f1, f2, f3], [4, 2], **{"physdims": [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = tpfa.Tpfa()
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
        p = sps.linalg.spsolve(A, rhs)
        p_cond, _, _, _ = condensation.solve_static_condensation(A, rhs, gb, dim=0)

        solver_coupler.split(gb, "pressure", p)
        solver_coupler.split(gb, "p_cond", p_cond)

        tol = 1e-10
        assert (np.amax(np.absolute(p - p_cond))) < tol
        assert (
            np.sum(error.error_L2(g, d["pressure"], d["p_cond"]) for g, d in gb) < tol
        )

    # ------------------------------------------------------------------------------#

    def atest_0d_elimination_3d_2d_1d_0d(self):
        """
        3d case with a single 0d grid.
        """
        f1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [.5, .5, .5, .5]])
        f2 = np.array([[.5, .5, .5, .5], [0, 1, 1, 0], [0, 0, 1, 1]])
        f3 = np.array([[0, 1, 1, 0], [.5, .5, .5, .5], [0, 0, 1, 1]])

        gb = meshing.cart_grid([f1, f2, f3], [2, 2, 2], **{"physdims": [1, 1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

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
        solver = tpfa.Tpfa()
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

        p = sps.linalg.spsolve(A, rhs)
        p_cond, _, _, _ = condensation.solve_static_condensation(A, rhs, gb, dim=0)

        solver_coupler.split(gb, "pressure", p)
        solver_coupler.split(gb, "p_cond", p_cond)

        tol = 1e-10
        assert (np.amax(np.absolute(p - p_cond))) < tol
        assert (
            np.sum(error.error_L2(g, d["pressure"], d["p_cond"]) for g, d in gb) < tol
        )


# ------------------------------------------------------------------------------#
#    if __name__ == '__main__':
#        unittest.main()
