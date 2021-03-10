"""
Module for testing the discrete fracture network (DFN) with continuous pressure and normal flux at the intersection.
"""

import unittest

import numpy as np
import scipy.sparse as sps

import porepy as pp


class TestDFN(unittest.TestCase):
    def test_mvem_0(self):
        """
        In this test we set up a network with 2 1d fractures that intersect in a point.
        We validate the resulting matrices and right hand side.
        We use the numerical scheme MVEM.
        """
        dfn_dim = 1

        # create the grid bucket
        gb, _ = pp.grid_buckets_2d.two_intersecting([2, 2], simplex=False)
        create_dfn(gb, dfn_dim)

        # setup data and assembler
        setup_data(gb)
        assembler, _ = setup_discr_mvem(gb)
        dof_manager = assembler._dof_manager
        assembler.discretize()
        A, b = assembler.assemble_matrix_rhs()
        A = A.todense()

        A_f1 = np.matrix(
            [
                [0.75, 0.0, 0.0, -0.25, 1, 0],
                [0.0, 0.75, 0.0, 0.0, 0, 0],
                [0.0, -0.25, 0.75, 0.0, 0, -1],
                [0.0, 0.0, 0.0, 0.75, 0, 0],
                [1.0, 0.0, 0.0, -1.0, 0, 0],
                [0.0, 1.0, -1.0, 0.0, 0, 0],
            ]
        )
        b_f1 = np.array([2, 0, 0, 0, 0, 0])

        A_f2 = np.matrix(
            [
                [0.75, 0.0, 0.0, -0.25, 1, 0],
                [0.0, 0.75, 0.0, 0.0, 0, 0],
                [0.0, -0.25, 0.75, 0.0, 0, -1],
                [0.0, 0.0, 0.0, 0.75, 0, 0],
                [1.0, 0.0, 0.0, -1.0, 0, 0],
                [0.0, 1.0, -1.0, 0.0, 0, 0],
            ]
        )
        b_f2 = np.array([1, 0, -1, 0, 0, 0])

        A_0 = np.matrix([[0.0]])
        b_0 = np.array([0])

        global_dof = np.cumsum(np.append(0, np.asarray(dof_manager.full_dof)))

        for g, _ in gb:
            block = dof_manager.block_dof[(g, "flow")]
            dof = np.arange(global_dof[block], global_dof[block + 1])

            if g.dim == 1 and np.allclose(g.nodes[0], 1):  # f1
                self.assertTrue(np.allclose(A[dof, :][:, dof], A_f1))
                self.assertTrue(np.allclose(b[dof], b_f1))
            elif g.dim == 1 and np.allclose(g.nodes[1], 1):  # f2
                self.assertTrue(np.allclose(A[dof, :][:, dof], A_f2))
                self.assertTrue(np.allclose(b[dof], b_f2))
            elif g.dim == 0:  # intersection
                self.assertTrue(np.allclose(A[dof, :][:, dof], A_0))
                self.assertTrue(np.allclose(b[dof], b_0))

        # known matrices associate to the edge where f1 is involved
        A_e1_gh_e = np.matrix(
            [
                [0.0, -0.25],
                [0.0, 0.0],
                [0.25, 0.0],
                [0.0, 0.0],
                [0.0, -1.0],
                [-1.0, 0.0],
            ]
        )

        A_e1_e_gh = np.matrix(
            [[0.0, 0.75, -0.25, 0.0, 0, 1], [0.25, 0.0, 0.0, -0.75, 1, 0]]
        )

        A_e1_gh_gl = np.matrix([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        A_e1_gl_gh = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        A_e1_e_e = np.matrix([[-0.75, 0.0], [0.0, -0.75]])

        A_e1_e_gl = np.matrix([[-1.0], [-1.0]])
        A_e1_gl_e = np.matrix([[-1.0, -1.0]])

        b_e1 = np.array([0.0, 0.0])

        A_e2_gh_e = np.matrix(
            [
                [0.0, -0.25],
                [0.0, 0.0],
                [0.25, 0.0],
                [0.0, 0.0],
                [0.0, -1.0],
                [-1.0, 0.0],
            ]
        )

        A_e2_e_gh = np.matrix(
            [[0.0, 0.75, -0.25, 0.0, 0.0, 1.0], [0.25, 0.0, 0.0, -0.75, 1.0, 0.0]]
        )

        A_e2_gh_gl = np.matrix([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        A_e2_gl_gh = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        A_e2_e_e = np.matrix([[-0.75, 0.0], [0.0, -0.75]])

        A_e2_e_gl = np.matrix([[-1.0], [-1.0]])
        A_e2_gl_e = np.matrix([[-1.0, -1.0]])

        b_e2 = np.array([0.0, 0.0])

        for e, d in gb.edges():
            gl, gh = gb.nodes_of_edge(e)

            block_e = dof_manager.block_dof[(e, "flow")]
            block_gl = dof_manager.block_dof[(gl, "flow")]
            block_gh = dof_manager.block_dof[(gh, "flow")]

            dof_e = np.arange(global_dof[block_e], global_dof[block_e + 1])
            dof_gl = np.arange(global_dof[block_gl], global_dof[block_gl + 1])
            dof_gh = np.arange(global_dof[block_gh], global_dof[block_gh + 1])

            if np.allclose(gh.nodes[0], 1):  # f1
                self.assertTrue(np.allclose(A[dof_gh, :][:, dof_e], A_e1_gh_e))
                self.assertTrue(np.allclose(A[dof_e, :][:, dof_gh], A_e1_e_gh))
                self.assertTrue(np.allclose(A[dof_gh, :][:, dof_gl], A_e1_gh_gl))
                self.assertTrue(np.allclose(A[dof_gl, :][:, dof_gh], A_e1_gl_gh))
                self.assertTrue(np.allclose(A[dof_e, :][:, dof_e], A_e1_e_e))
                self.assertTrue(np.allclose(A[dof_e, :][:, dof_gl], A_e1_e_gl))
                self.assertTrue(np.allclose(A[dof_gl, :][:, dof_e], A_e1_gl_e))
                self.assertTrue(np.allclose(b[dof_e], b_e1))

            elif np.allclose(gh.nodes[1], 1):  # f2
                self.assertTrue(np.allclose(A[dof_gh, :][:, dof_e], A_e2_gh_e))
                self.assertTrue(np.allclose(A[dof_e, :][:, dof_gh], A_e2_e_gh))
                self.assertTrue(np.allclose(A[dof_gh, :][:, dof_gl], A_e2_gh_gl))
                self.assertTrue(np.allclose(A[dof_gl, :][:, dof_gh], A_e2_gl_gh))
                self.assertTrue(np.allclose(A[dof_e, :][:, dof_e], A_e2_e_e))
                self.assertTrue(np.allclose(A[dof_e, :][:, dof_gl], A_e2_e_gl))
                self.assertTrue(np.allclose(A[dof_gl, :][:, dof_e], A_e2_gl_e))
                self.assertTrue(np.allclose(b[dof_e], b_e2))

    def test_tpfa_0(self):
        """
        In this test we set up a network with 2 1d fractures that intersect in a point.
        We validate the resulting matrices and right hand side.
        We use the numerical scheme Tpfa.
        """
        dfn_dim = 1
        gb, _ = pp.grid_buckets_2d.two_intersecting([2, 2], simplex=False)
        create_dfn(gb, dfn_dim)

        # setup data and assembler
        setup_data(gb)
        assembler, _ = setup_discr_tpfa(gb)
        dof_manager = assembler._dof_manager

        assembler.discretize()
        A, b = assembler.assemble_matrix_rhs()
        A = A.todense()

        A_f1 = np.matrix([[2, 0], [0, 2]])
        b_f1 = np.array([4, 0])

        A_f2 = np.matrix([[2, 0], [0, 2]])
        b_f2 = np.array([2, 2])

        A_0 = np.matrix([[0.0]])
        b_0 = np.array([0])

        global_dof = np.cumsum(np.append(0, np.asarray(dof_manager.full_dof)))

        for g, _ in gb:
            block = dof_manager.block_dof[(g, "flow")]
            dof = np.arange(global_dof[block], global_dof[block + 1])

            if g.dim == 1 and np.allclose(g.nodes[0], 1):  # f1

                self.assertTrue(np.allclose(A[dof, :][:, dof], A_f1))
                self.assertTrue(np.allclose(b[dof], b_f1))
            elif g.dim == 1 and np.allclose(g.nodes[1], 1):  # f2
                self.assertTrue(np.allclose(A[dof, :][:, dof], A_f2))
                self.assertTrue(np.allclose(b[dof], b_f2))
            elif g.dim == 0:  # intersection
                self.assertTrue(np.allclose(A[dof, :][:, dof], A_0))
                self.assertTrue(np.allclose(b[dof], b_0))

        # known matrices associate to the edge where f1 is involved
        A_e1_gh_e = np.matrix([[0, 1], [1, 0]])

        A_e1_e_gh = np.matrix([[0, 1], [1, 0]])

        A_e1_gh_gl = np.matrix([[0.0], [0.0]])
        A_e1_gl_gh = np.matrix([[0.0, 0.0]])

        A_e1_e_e = np.matrix([[-0.5, 0.0], [0.0, -0.5]])

        A_e1_e_gl = np.matrix([[-1.0], [-1.0]])
        A_e1_gl_e = np.matrix([[-1.0, -1.0]])

        b_e1 = np.array([0.0, 0.0])

        A_e2_gh_e = np.matrix([[0, 1], [1, 0]])

        A_e2_e_gh = np.matrix([[0, 1], [1, 0]])

        A_e2_gh_gl = np.matrix([[0.0], [0.0]])
        A_e2_gl_gh = np.matrix([[0.0, 0.0]])

        A_e2_e_e = np.matrix([[-0.5, 0.0], [0.0, -0.5]])

        A_e2_e_gl = np.matrix([[-1.0], [-1.0]])
        A_e2_gl_e = np.matrix([[-1.0, -1.0]])

        b_e2 = np.array([0.0, 0.0])

        for e, _ in gb.edges():
            gl, gh = gb.nodes_of_edge(e)

            block_e = dof_manager.block_dof[(e, "flow")]
            block_gl = dof_manager.block_dof[(gl, "flow")]
            block_gh = dof_manager.block_dof[(gh, "flow")]

            dof_e = np.arange(global_dof[block_e], global_dof[block_e + 1])
            dof_gl = np.arange(global_dof[block_gl], global_dof[block_gl + 1])
            dof_gh = np.arange(global_dof[block_gh], global_dof[block_gh + 1])

            if np.allclose(gh.nodes[0], 1):  # f1
                self.assertTrue(np.allclose(A[dof_gh, :][:, dof_e], A_e1_gh_e))
                self.assertTrue(np.allclose(A[dof_e, :][:, dof_gh], A_e1_e_gh))
                self.assertTrue(np.allclose(A[dof_gh, :][:, dof_gl], A_e1_gh_gl))
                self.assertTrue(np.allclose(A[dof_gl, :][:, dof_gh], A_e1_gl_gh))
                self.assertTrue(np.allclose(A[dof_e, :][:, dof_e], A_e1_e_e))
                self.assertTrue(np.allclose(A[dof_e, :][:, dof_gl], A_e1_e_gl))
                self.assertTrue(np.allclose(A[dof_gl, :][:, dof_e], A_e1_gl_e))
                self.assertTrue(np.allclose(b[dof_e], b_e1))

            elif np.allclose(gh.nodes[1], 1):  # f2
                self.assertTrue(np.allclose(A[dof_gh, :][:, dof_e], A_e2_gh_e))
                self.assertTrue(np.allclose(A[dof_e, :][:, dof_gh], A_e2_e_gh))
                self.assertTrue(np.allclose(A[dof_gh, :][:, dof_gl], A_e2_gh_gl))
                self.assertTrue(np.allclose(A[dof_gl, :][:, dof_gh], A_e2_gl_gh))
                self.assertTrue(np.allclose(A[dof_e, :][:, dof_e], A_e2_e_e))
                self.assertTrue(np.allclose(A[dof_e, :][:, dof_gl], A_e2_e_gl))
                self.assertTrue(np.allclose(A[dof_gl, :][:, dof_e], A_e2_gl_e))
                self.assertTrue(np.allclose(b[dof_e], b_e2))

    def test_mvem_1(self):
        """
        In this test we set up a network with 5 1d fractures that intersect in a point.
        We validate the resulting solution.
        We use the numerical scheme MVEM.
        """
        dfn_dim = 1

        N = 8
        f1 = N * np.array([[0, 1], [0.5, 0.5]])
        f2 = N * np.array([[0.5, 0.5], [0, 1]])
        f3 = N * np.array([[0.625, 0.625], [0.5, 0.75]])
        f4 = N * np.array([[0.25, 0.75], [0.25, 0.25]])
        f5 = N * np.array([[0.75, 0.75], [0.125, 0.375]])

        # create the grid bucket
        gb = pp.meshing.cart_grid([f1, f2, f3, f4, f5], [N, N])
        gb.compute_geometry()
        create_dfn(gb, dfn_dim)

        # setup data and assembler
        setup_data(gb)
        assembler, (discr, _) = setup_discr_mvem(gb)

        assembler.discretize()
        A, b = assembler.assemble_matrix_rhs()
        x = sps.linalg.spsolve(A, b)

        assembler.distribute_variable(x)
        for g, d in gb:
            discr = d["discretization"]["flow"]["flux"]
            d["pressure"] = discr.extract_pressure(g, d[pp.STATE]["flow"], d)

        for g, d in gb:

            if g.dim == 1:
                if np.all(g.cell_centers[1] == 0.5 * N):  # f1
                    known = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
                elif np.all(g.cell_centers[0] == 0.5 * N):  # f2
                    known = np.array([7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5])
                elif np.all(g.cell_centers[0] == 0.625 * N):  # f3
                    known = np.array([4.0, 4.0])
                elif np.all(g.cell_centers[1] == 0.25 * N):  # f4
                    known = np.array([2.0, 2.0, 2.0, 2.0])
                elif np.all(g.cell_centers[0] == 0.75 * N):  # f5
                    known = np.array([2.0, 2.0])
                else:
                    raise ValueError

            else:  # g.dim == 0
                if np.allclose(g.cell_centers, np.array([[0.5], [0.5], [0.0]]) * N):
                    known = np.array([4.0])
                elif np.allclose(g.cell_centers, np.array([[0.625], [0.5], [0.0]]) * N):
                    known = np.array([4.0])
                elif np.allclose(g.cell_centers, np.array([[0.5], [0.25], [0.0]]) * N):
                    known = np.array([2.0])
                elif np.allclose(g.cell_centers, np.array([[0.75], [0.25], [0.0]]) * N):
                    known = np.array([2.0])
                else:
                    raise ValueError

            self.assertTrue(np.allclose(d["pressure"], known))

    def test_tpfa_1(self):
        """
        In this test we set up a network with 5 1d fractures that intersect in a point.
        We validate the resulting solution.
        We use the numerical scheme Tpfa.
        """
        dfn_dim = 1

        N = 8
        f1 = N * np.array([[0, 1], [0.5, 0.5]])
        f2 = N * np.array([[0.5, 0.5], [0, 1]])
        f3 = N * np.array([[0.625, 0.625], [0.5, 0.75]])
        f4 = N * np.array([[0.25, 0.75], [0.25, 0.25]])
        f5 = N * np.array([[0.75, 0.75], [0.125, 0.375]])

        # create the grid bucket
        gb = pp.meshing.cart_grid([f1, f2, f3, f4, f5], [N, N])
        gb.compute_geometry()
        create_dfn(gb, dfn_dim)

        # setup data and assembler
        setup_data(gb)
        assembler, _ = setup_discr_tpfa(gb)
        assembler.discretize()
        A, b = assembler.assemble_matrix_rhs()
        x = sps.linalg.spsolve(A, b)

        assembler.distribute_variable(x)

        for g, d in gb:

            if g.dim == 1:
                if np.all(g.cell_centers[1] == 0.5 * N):  # f1
                    known = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
                elif np.all(g.cell_centers[0] == 0.5 * N):  # f2
                    known = np.array([7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5])
                elif np.all(g.cell_centers[0] == 0.625 * N):  # f3
                    known = np.array([4, 4])
                elif np.all(g.cell_centers[1] == 0.25 * N):  # f4
                    known = np.array([2, 2, 2, 2])
                elif np.all(g.cell_centers[0] == 0.75 * N):  # f5
                    known = np.array([2, 2])
                else:
                    raise ValueError

            else:  # g.dim == 0
                if np.allclose(g.cell_centers, np.array([[0.5], [0.5], [0]]) * N):
                    known = np.array([4])
                elif np.allclose(g.cell_centers, np.array([[0.625], [0.5], [0]]) * N):
                    known = np.array([4])
                elif np.allclose(g.cell_centers, np.array([[0.5], [0.25], [0]]) * N):
                    known = np.array([2])
                elif np.allclose(g.cell_centers, np.array([[0.75], [0.25], [0]]) * N):
                    known = np.array([2])
                else:
                    raise ValueError

            self.assertTrue(np.allclose(d[pp.STATE]["flow"], known))


# ------------------------- HELP FUNCTIONS --------------------------------#


def setup_data(gb, key="flow"):
    """Setup the data"""
    for g, d in gb:
        param = {}
        kxx = np.ones(g.num_cells)
        param["second_order_tensor"] = pp.SecondOrderTensor(kxx)

        if g.dim == gb.dim_max():
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            bound = pp.BoundaryCondition(
                g, bound_faces.ravel("F"), ["dir"] * bound_faces.size
            )
            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces] = g.face_centers[1, bound_faces]
            param["bc"] = bound
            param["bc_values"] = bc_val
            param["aperture"] = np.ones(g.num_cells)
            d[pp.PARAMETERS] = pp.Parameters(g, key, param)
            d[pp.DISCRETIZATION_MATRICES] = {key: {}}

    for _, d in gb.edges():
        d[pp.DISCRETIZATION_MATRICES] = {key: {}}


def setup_discr_mvem(gb, key="flow"):
    """ Setup the discretization MVEM. """
    discr = pp.MVEM(key)
    p_trace = pp.CellDofFaceDofMap(key)
    interface = pp.FluxPressureContinuity(key, discr, p_trace)

    for g, d in gb:
        if g.dim == gb.dim_max():
            d[pp.PRIMARY_VARIABLES] = {key: {"cells": 1, "faces": 1}}
            d[pp.DISCRETIZATION] = {key: {"flux": discr}}
        else:
            d[pp.PRIMARY_VARIABLES] = {key: {"cells": 1}}
            d[pp.DISCRETIZATION] = {key: {"flux": p_trace}}

    for e, d in gb.edges():
        g_secondary, g_primary = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {key: {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {
            "flux": {
                g_secondary: (key, "flux"),
                g_primary: (key, "flux"),
                e: (key, interface),
            }
        }

    return pp.Assembler(gb), (discr, p_trace)


def setup_discr_tpfa(gb, key="flow"):
    """ Setup the discretization Tpfa. """
    discr = pp.Tpfa(key)
    p_trace = pp.CellDofFaceDofMap(key)
    interface = pp.FluxPressureContinuity(key, discr, p_trace)

    for g, d in gb:
        if g.dim == gb.dim_max():
            d[pp.PRIMARY_VARIABLES] = {key: {"cells": 1}}
            d[pp.DISCRETIZATION] = {key: {"flux": discr}}
        else:
            d[pp.PRIMARY_VARIABLES] = {key: {"cells": 1}}
            d[pp.DISCRETIZATION] = {key: {"flux": p_trace}}

    for e, d in gb.edges():
        g_secondary, g_primary = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {key: {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {
            "flux": {
                g_secondary: (key, "flux"),
                g_primary: (key, "flux"),
                e: (key, interface),
            }
        }

    return pp.Assembler(gb), (discr, p_trace)


def create_dfn(gb, dim):
    """given a GridBucket remove the higher dimensional node and
    fix the internal mapping."""
    # remove the +1 and -2 dimensional grids with respect to the
    # considered dfn, and re-write the node number
    gd = np.hstack((gb.grids_of_dimension(dim + 1), gb.grids_of_dimension(dim - 2)))

    for g in gd:
        node_number = gb.node_props(g, "node_number")
        gb.remove_node(g)
        gb.update_node_ordering(node_number)


if __name__ == "__main__":
    unittest.main()
