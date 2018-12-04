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
    
    We define a random Robin and Mortar weights and test if we recover the
    condition on the interface.
    """

    def test_robin_coupling(self):
        """
        Test a Robin condition on the interface
        """
        self.kw = "mech"
        gb = define_gb()
        mortar_weight = np.random.rand(gb.dim_max())
        robin_weight = np.random.rand(gb.dim_max())
        rhs = np.random.rand(gb.dim_max())
        self.assign_parameters(gb, mortar_weight, robin_weight, rhs)
        self.assign_discretization(gb)
        assembler = pp.Assembler()
        matrix, rhs, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
        u = sps.linalg.spsolve(matrix, rhs)
        assembler.distribute_variable(gb, u, block_dof, full_dof)
        self.check_solution(gb)

    def test_continuity_coupling(self):
        """
        Test a continuity condition on the interface. This is equivalent to
        zero mortar weight and identity matrix for the robin weight. These
        matrices are only used to check the solution.
        """
        self.kw = "mech"
        gb = define_gb()
        # We assign weighs according to the condition.
        mortar_weight = np.zeros(gb.dim_max())
        robin_weight = np.ones(gb.dim_max())
        rhs = np.zeros(gb.dim_max())
        self.assign_parameters(gb, mortar_weight, robin_weight, rhs)
        self.assign_discretization(gb, robin=False)
        assembler = pp.Assembler()
        matrix, rhs, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
        u = sps.linalg.spsolve(matrix, rhs)
        assembler.distribute_variable(gb, u, block_dof, full_dof)
        self.check_solution(gb)

    def check_solution(self, gb):
        for e, d in gb.edges():
            mg = d["mortar_grid"]
            gs, gm = gb.nodes_of_edge(e)
            if gs.grid_num == 2:
                g_temp = gs
                gs = gm
                gm = g_temp

            ds = gb.node_props(gs)
            dm = gb.node_props(gm)

            us = ds[self.kw]
            um = dm[self.kw]
            lam = d[self.kw]

            bdcs = ds[pp.DISCRETIZATION_MATRICES][self.kw]["bound_displacement_cell"]
            bdcm = dm[pp.DISCRETIZATION_MATRICES][self.kw]["bound_displacement_cell"]
            bdfs = ds[pp.DISCRETIZATION_MATRICES][self.kw]["bound_displacement_face"]
            bdfm = dm[pp.DISCRETIZATION_MATRICES][self.kw]["bound_displacement_face"]

            bc_val_s = ds[pp.PARAMETERS][self.kw]["bc_values"]
            bc_val_m = dm[pp.PARAMETERS][self.kw]["bc_values"]

            RW = d[pp.PARAMETERS][self.kw]["robin_weight"]
            MW = d[pp.PARAMETERS][self.kw]["mortar_weight"]
            rhs = d[pp.PARAMETERS][self.kw]["robin_rhs"].reshape(
                (gs.dim, -1), order="F"
            )
            slv2mrt_nd = sps.kron(mg.slave_to_mortar_int, sps.eye(gs.dim)).tocsr()
            mstr2mrt_nd = sps.kron(mg.master_to_mortar_int, sps.eye(gs.dim)).tocsr()

            hf2fs = pp.fvutils.map_hf_2_f(g=gs) / 2
            hf2fm = pp.fvutils.map_hf_2_f(g=gm) / 2
            jump_u = (
                slv2mrt_nd
                * hf2fs
                * (bdcs * us + bdfs * (bc_val_s + slv2mrt_nd.T * lam))
                - mstr2mrt_nd
                * hf2fm
                * (bdcm * um - bdfm * (bc_val_m + mstr2mrt_nd.T * lam))
            ).reshape((gs.dim, -1), order="F")
            lam_nd = lam.reshape((gs.dim, -1), order="F")

            for i in range(len(RW)):
                rhs_robin = MW[i].dot(lam_nd[:, i]) + RW[i].dot(jump_u[:, i])
                self.assertTrue(np.allclose(rhs_robin, rhs[:, i]))

    def solve_and_compare(self, gb):
        assembler = pp.Assembler()
        matrix, rhs, block_dof, full_dof = assembler.assemble_matrix_rhs(
            gb, variables=["mech"]
        )
        u = sps.linalg.spsolve(matrix, rhs)
        assembler.distribute_variable(gb, u, block_dof, full_dof)
        uc = sps.linalg.spsolve(matrix_c, rhs_c)

        self.assertTrue(np.allclose(u, uc))

    def assign_discretization(self, gb, robin=True):
        for _, d in gb:
            d[pp.PRIMARY_VARIABLES] = {self.kw: {"cells": gb.dim_max()}}
            d[pp.DISCRETIZATION] = {self.kw: {"mortar": pp.Mpsa(self.kw)}}

        if robin:
            contact = pp.RobinContact(self.kw, pp.Mpsa(self.kw))
        else:
            contact = pp.StressDisplacementContinuity(self.kw, pp.Mpsa(self.kw))
        for e, d in gb.edges():
            g1, g2 = gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES] = {self.kw: {"cells": gb.dim_max()}}
            d[pp.COUPLING_DISCRETIZATION] = {
                self.kw: {
                    g1: (self.kw, "mortar"),
                    g2: (self.kw, "mortar"),
                    e: (self.kw, contact),
                }
            }

    def assign_parameters(self, gb, mortar_weight, robin_weight, rhs):
        for g, d in gb:
            if g.grid_num == 1:
                dir_faces = g.face_centers[0] < 1e-10
            elif g.grid_num == 2:
                dir_faces = g.face_centers[0] > 2 - 1e-10

            bc_val = np.zeros((g.dim, g.num_faces))
            bc_val[0, dir_faces] = 0.1 * g.face_centers[0, dir_faces]
            bc = pp.BoundaryConditionVectorial(g, dir_faces, "dir")
            C = pp.FourthOrderTensor(
                gb.dim_max(), np.ones(g.num_cells), np.ones(g.num_cells)
            )
            data = {
                "bc": bc,
                "bc_values": bc_val.ravel("F"),
                "fourth_order_tensor": C,
                "source": np.zeros(g.num_cells * g.dim),
                "inverter": "python",
            }
            pp.initialize_data(d, g, self.kw, data)

        for e, d in gb.edges():
            mg = d["mortar_grid"]
            MW = sps.diags(mortar_weight)
            RW = sps.diags(robin_weight)
            data = {
                "mortar_weight": [MW] * mg.num_cells,
                "robin_weight": [RW] * mg.num_cells,
                "robin_rhs": np.tile(rhs, (mg.num_cells)),
            }
            pp.initialize_data(d, mg, self.kw, data)


def define_gb():
    """
    Construct grids
    """
    g_s = pp.CartGrid([1, 2], [1, 2])
    g_m = pp.CartGrid([1, 2], [1, 2])
    g_m.nodes[0] += 1
    g_s.compute_geometry()
    g_m.compute_geometry()

    g_s.grid_num = 1
    g_m.grid_num = 2

    gb = pp.GridBucket()
    gb.add_nodes([g_s, g_m])
    contact_s = np.where(g_s.face_centers[0] > 1 - 1e-10)[0]
    contact_m = np.where(g_m.face_centers[0] < 1 + 1e-10)[0]
    data = np.ones(contact_s.size, dtype=np.bool)

    shape = (g_s.num_faces, g_m.num_faces)
    slave_master = sps.csc_matrix((data, (contact_m, contact_s)), shape=shape)

    gb.add_edge([g_s, g_m], slave_master)

    mortar_grid, _, _ = pp.grids.partition.extract_subgrid(g_s, contact_s, faces=True)
    mg = pp.BoundaryMortar(mortar_grid.dim, mortar_grid, slave_master.T)
    gb.set_edge_prop([g_s, g_m], "mortar_grid", mg)
    return gb
