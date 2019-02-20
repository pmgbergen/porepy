"""
Module for testing the vector elliptic couplings in the interface_laws.
"""


import numpy as np
import scipy.sparse as sps
import unittest

import porepy as pp


class TestTwoGridCoupling(unittest.TestCase):
    """
    In this test we set up a coupling between two grids of dimension 2:
    g_slave     g_master
    |-----| | |------|
    |     | x |      |
    |-----| | |------|
           g_mortar
    There is one cell per grid and they are coupled together by a single mortar
    variable.
    """

    def test_robin_assembly_master(self):
        """
        We test the RobinContact interface law. This gives a Robin condition
        on the mortar grid.

        Test the assembly of the master terms.
        """
        self.kw = "mech"
        gb = define_gb()
        mortar_weight = np.zeros(gb.dim_max())
        robin_weight = np.ones(gb.dim_max())
        rhs = np.ones(gb.dim_max())
        self.assign_parameters(gb, mortar_weight, robin_weight, rhs)
        varargs = get_variables(gb)

        robin_contact = pp.RobinContact(self.kw, MockId(), MockZero())
        matrix, rhs = robin_contact.assemble_matrix_rhs(*varargs)

        # known coupling
        A = np.array(
            [
                [2, 0, 2, 0, -2, 0],
                [0, 2, 0, 2, 0, -2],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [-2, 0, 0, 0, 2, 0],
                [0, -2, 0, 0, 0, 2],
            ]
        )

        matrix = sps.bmat(matrix)
        self.assertTrue(np.allclose(A, matrix.A))
        self.assertTrue(
            np.allclose(np.hstack(rhs.ravel()), np.array([0, 0, 0, 0, 1, 1]))
        )

    def test_robin_assembly_slave(self):
        """
        We test the RobinContact interface law. This gives a Robin condition
        on the mortar grid.

        Test the assembly of the slave terms.
        """
        self.kw = "mech"
        gb = define_gb()
        mortar_weight = np.zeros(gb.dim_max())
        robin_weight = np.ones(gb.dim_max())
        rhs = np.ones(gb.dim_max())
        self.assign_parameters(gb, mortar_weight, robin_weight, rhs)
        varargs = get_variables(gb)

        robin_contact = pp.RobinContact(self.kw, MockZero(), MockId())
        matrix, rhs = robin_contact.assemble_matrix_rhs(*varargs)

        # known coupling
        A = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [2, 0, 2, 0, 2, 0],
                [0, 2, 0, 2, 0, 2],
                [0, 0, 2, 0, 2, 0],
                [0, 0, 0, 2, 0, 2],
            ]
        )

        matrix = sps.bmat(matrix)
        self.assertTrue(np.allclose(A, matrix.A))
        self.assertTrue(
            np.allclose(np.hstack(rhs.ravel()), np.array([0, 0, 0, 0, 1, 1]))
        )

    def test_robin_assembly_mortar(self):
        """
        We test the RobinContact interface law. This gives a Robin condition
        on the mortar grid.

        Test the assembly of the mortar terms.
        """
        self.kw = "mech"
        gb = define_gb()
        mortar_weight = np.ones(gb.dim_max())
        robin_weight = np.zeros(gb.dim_max())
        rhs = np.ones(gb.dim_max())
        self.assign_parameters(gb, mortar_weight, robin_weight, rhs)
        varargs = get_variables(gb)

        robin_contact = pp.RobinContact(self.kw, MockZero(), MockZero())
        matrix, rhs = robin_contact.assemble_matrix_rhs(*varargs)

        # known coupling
        A = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        matrix = sps.bmat(matrix)
        self.assertTrue(np.allclose(A, matrix.A))
        self.assertTrue(
            np.allclose(np.hstack(rhs.ravel()), np.array([0, 0, 0, 0, 1, 1]))
        )

    def test_continuity_assembly_master(self):
        """
        We test the RobinContact interface law. This gives a Robin condition
        on the mortar grid.

        Test the assembly of the master terms.
        """
        self.kw = "mech"
        gb = define_gb()
        varargs = get_variables(gb)
        robin_contact = pp.StressDisplacementContinuity(self.kw, MockId(), MockZero())
        matrix, rhs = robin_contact.assemble_matrix_rhs(*varargs)

        # known coupling
        A = np.array(
            [
                [2, 0, 2, 0, -2, 0],
                [0, 2, 0, 2, 0, -2],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [-2, 0, 0, 0, 2, 0],
                [0, -2, 0, 0, 0, 2],
            ]
        )

        matrix = sps.bmat(matrix)

        self.assertTrue(np.allclose(A, matrix.A))
        self.assertTrue(
            np.allclose(np.hstack(rhs.ravel()), np.array([0, 0, 0, 0, 0, 0]))
        )

    def test_continuity_assembly_slave(self):
        """
        We test the StressDisplacementContinuity interface law. This gives a continuity
        of stress and displacement on the mortar grid.

        Test the assembly of the slave terms.
        """
        self.kw = "mech"
        gb = define_gb()
        varargs = get_variables(gb)

        robin_contact = pp.StressDisplacementContinuity(self.kw, MockZero(), MockId())
        matrix, rhs = robin_contact.assemble_matrix_rhs(*varargs)

        # known coupling
        A = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [2, 0, 2, 0, 2, 0],
                [0, 2, 0, 2, 0, 2],
                [0, 0, 2, 0, 2, 0],
                [0, 0, 0, 2, 0, 2],
            ]
        )

        matrix = sps.bmat(matrix)
        self.assertTrue(np.allclose(A, matrix.A))
        self.assertTrue(
            np.allclose(np.hstack(rhs.ravel()), np.array([0, 0, 0, 0, 0, 0]))
        )

    def test_continuity_assembly_mortar(self):
        """
        We test the StressDisplacementContinuity interface law. This gives a continuity
        of stress and displacement on the mortar grid.

        Test the assembly of the mortar terms.
        """
        self.kw = "mech"
        gb = define_gb()
        varargs = get_variables(gb)

        robin_contact = pp.StressDisplacementContinuity(self.kw, MockZero(), MockZero())
        matrix, rhs = robin_contact.assemble_matrix_rhs(*varargs)

        # known coupling
        A = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        matrix = sps.bmat(matrix)
        self.assertTrue(np.allclose(A, matrix.A))
        self.assertTrue(
            np.allclose(np.hstack(rhs.ravel()), np.array([0, 0, 0, 0, 0, 0]))
        )

    def assign_parameters(self, gb, mortar_weight, robin_weight, rhs):
        for e, d in gb.edges():
            mg = d["mortar_grid"]
            MW = sps.diags(mortar_weight)
            RW = sps.diags(robin_weight)
            data = {
                "mortar_weight": [MW] * mg.num_cells,
                "robin_weight": [RW] * mg.num_cells,
                "robin_rhs": np.tile(rhs, (mg.num_cells)),
            }
            d[pp.PARAMETERS] = pp.Parameters(e, self.kw, data)
            d[pp.DISCRETIZATION_MATRICES] = {self.kw: {}}


class MockId(object):
    """
    returns an identity mapping
    """

    def ndof(self, g):
        return g.dim * g.num_cells

    def assemble_int_bound_displacement_trace(self, *vargs):
        identity_mapping(self, *vargs)

    def assemble_int_bound_stress(self, *vargs):
        identity_mapping(self, *vargs)

    def enforce_neumann_int_bound(self, *varargs):
        pass


class MockZero(object):
    """
    Do nothing
    """

    def ndof(self, g):
        return g.dim * g.num_cells

    def assemble_int_bound_displacement_trace(self, *vargs):
        pass

    def assemble_int_bound_stress(self, *vargs):
        pass

    def enforce_neumann_int_bound(self, *varargs):
        pass


def identity_mapping(RC, g, data, data_edge, swap, cc, matrix, rhs, ind):
    dof_master = g.dim * g.num_cells
    dof_slave = g.dim * g.num_cells
    dof_mortar = g.dim * data_edge["mortar_grid"].num_cells

    cc[ind, 0] += sps.diags(np.ones(dof_slave), shape=(dof_slave, dof_master))
    cc[ind, 1] += sps.diags(np.ones(dof_slave), shape=(dof_slave, dof_slave))
    cc[ind, 2] += sps.diags(np.ones(dof_slave), shape=(dof_slave, dof_mortar))

    cc[2, ind] += sps.diags(np.ones(dof_mortar), shape=(dof_mortar, dof_master))
    cc[2, 2] += sps.diags(np.ones(dof_mortar), shape=(dof_mortar, dof_mortar))


def get_variables(gb):
    for e, data_edge in gb.edges():
        g_slave, g_master = gb.nodes_of_edge(e)
        data_slave = gb.node_props(g_slave)
        data_master = gb.node_props(g_master)
        break
    dof = np.array(
        [
            g_master.dim * g_master.num_cells,
            g_slave.dim * g_slave.num_cells,
            g_master.dim * data_edge["mortar_grid"].num_cells,
        ]
    )
    matrix = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
    matrix = matrix.reshape((3, 3))

    rhs = np.empty(3, dtype=np.object)
    rhs[0] = np.zeros(g_master.dim * g_master.num_cells)
    rhs[1] = np.zeros(g_slave.dim * g_slave.num_cells)
    rhs[2] = np.zeros(data_edge['mortar_grid'].num_cells * g_slave.dim)

    return g_master, g_slave, data_master, data_slave, data_edge, matrix


def define_gb():
    """
    Construct grids
    """
    g1 = pp.CartGrid([1, 1])
    g2 = pp.CartGrid([1, 1])
    g1.compute_geometry()
    g2.compute_geometry()

    g1.grid_num = 1
    g2.grid_num = 2

    gb = pp.GridBucket()
    gb.add_nodes([g1, g2])
    gb.add_edge([g1, g2], None)
    mortar_grid = pp.Grid(
        1,
        np.array([[0, 0, 0], [0, 1, 0]]).T,
        sps.csc_matrix(([True, True], [0, 1], [0, 1, 2])),
        sps.csc_matrix(([1, -1], [0, 1], [0, 2])),
        "mortar_grid",
    )
    mortar_grid.compute_geometry()
    face_faces = sps.csc_matrix(([True], [0], [0, 0, 1, 1, 1]), shape=(4, 4))
    mg = pp.BoundaryMortar(1, mortar_grid, face_faces)
    mg.num_cells = 1
    gb.set_edge_prop([g1, g2], "mortar_grid", mg)
    return gb


if __name__ == "__main__":
    unittest.main()
