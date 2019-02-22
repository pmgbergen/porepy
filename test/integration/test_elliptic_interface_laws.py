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
        gb, _ = define_gb()
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
        matrices are only used to check the solution. We check the solution
        against a reference computed on a single grid including both g_s and g_m.
        """
        self.kw = "mech"
        gb, gb_full = define_gb()
        # We assign weighs according to the condition.
        mortar_weight = np.zeros(gb.dim_max())
        robin_weight = np.ones(gb.dim_max())
        rhs = np.zeros(gb.dim_max())
        # Assign data to coupling gb
        self.assign_parameters(gb, mortar_weight, robin_weight, rhs)
        self.assign_discretization(gb, robin=False)
        # Assign data to mono gb
        self.assign_parameters(gb_full, mortar_weight, robin_weight, rhs)
        self.assign_discretization(gb_full, robin=False)

        assembler = pp.Assembler()
        matrix, rhs, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
        u = sps.linalg.spsolve(matrix, rhs)
        assembler.distribute_variable(gb, u, block_dof, full_dof)
        self.check_solution(gb)

        matrix, rhs, block_dof, full_dof = assembler.assemble_matrix_rhs(gb_full)
        u_full = sps.linalg.spsolve(matrix, rhs)
        # compare solutions
        # We need to rearange the solutions because the ordering of the dofs are not the same
        # Also, we don't have equality because the weak symmetry breaks when a cell has to many
        # Neumann conditions (see comments in mpsa)
        us = []
        ID = []  # to appease porpy 3.5
        for g, d in gb:
            us.append(d[self.kw])
            ID.append(g.grid_num - 1)
        us = np.hstack([np.array(us)[ID].ravel()])
        IA = np.array([0, 1, 4, 5, 2, 3, 6, 7])
        sol = us[IA]
        self.assertTrue(np.all(np.abs(sol - u_full) < 1e-4))

    def check_solution(self, gb):
        for e, d in gb.edges():
            mg = d["mortar_grid"]
            gs, gm = gb.nodes_of_edge(e)

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
            slv2mrt_nd = sps.kron(mg.slave_to_mortar_int(), sps.eye(gs.dim)).tocsr()
            mstr2mrt_nd = sps.kron(mg.master_to_mortar_int(), sps.eye(gs.dim)).tocsr()

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
            elif g.grid_num == 3:
                dir_faces = (g.face_centers[0] < 1e-10) + (
                    g.face_centers[0] > 2 - 1e-10
                )

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
            pp.initialize_data(g, d, self.kw, data)

        for _, d in gb.edges():
            mg = d["mortar_grid"]
            MW = sps.diags(mortar_weight)
            RW = sps.diags(robin_weight)
            data = {
                "mortar_weight": [MW] * mg.num_cells,
                "robin_weight": [RW] * mg.num_cells,
                "robin_rhs": np.tile(rhs, (mg.num_cells)),
            }
            pp.initialize_data(mg, d, self.kw, data)


class TestBiotTwoGridCoupling(unittest.TestCase):
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
        self.kw_f = "flow"
        gb, _ = define_gb()
        mortar_weight = np.random.rand(gb.dim_max())
        robin_weight = np.random.rand(gb.dim_max())
        rhs = np.random.rand(gb.dim_max())
        self.assign_parameters(gb, mortar_weight, robin_weight, rhs)
        self.assign_discretization(gb)

        # Discretize
        for g, d in gb:
            pp.Biot(self.kw, self.kw_f).discretize(g, d)

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
        self.kw_f = "flow"
        gb, gb_full = define_gb()
        # We assign weighs according to the condition.
        mortar_weight = np.zeros(gb.dim_max())
        robin_weight = np.ones(gb.dim_max())
        rhs = np.zeros(gb.dim_max())
        # Assign data to coupling gb
        self.assign_parameters(gb, mortar_weight, robin_weight, rhs)
        self.assign_discretization(gb)
        # Assign data to mono gb
        self.assign_parameters(gb_full, mortar_weight, robin_weight, rhs)
        self.assign_discretization(gb_full)

        # Discretize
        for g, d in gb:
            pp.Biot(self.kw, self.kw_f).discretize(g, d)
        for g, d in gb_full:
            pp.Biot(self.kw, self.kw_f).discretize(g, d)

        # Assemble and solve
        assembler = pp.Assembler()
        matrix, rhs, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
        u = sps.linalg.spsolve(matrix, rhs)
        assembler.distribute_variable(gb, u, block_dof, full_dof)
        self.check_solution(gb)

        matrix, rhs, block_dof, full_dof = assembler.assemble_matrix_rhs(gb_full)
        u_full = sps.linalg.spsolve(matrix, rhs)
        assembler.distribute_variable(gb_full, u_full, block_dof, full_dof)
        # Compare solutions:
        # We need to rearange the solutions because the ordering of the dofs are not the
        # same for the two grids.
        us = []
        ps = []
        ID = []
        for g, d in gb:
            us.append(d["u"])
            ps.append(d["p"])
            ID.append(g.grid_num - 1)
        us = np.hstack([np.array(us)[ID].ravel()])
        ps = np.hstack([np.array(ps)[ID].ravel()])
        IA = np.array([0, 1, 4, 5, 2, 3, 6, 7])
        IAp = np.array([0, 2, 1, 3])
        sol = np.hstack([us[IA], ps[IAp]])

        us = []
        ps = []
        for g, d in gb_full:
            us.append(d["u"])
            ps.append(d["p"])
        sol_full = np.hstack([np.array(us), np.array(ps)])
        # Note, we don't have equality because the weak symmetry breaks when a cell has to many
        # Neumann conditions (see comments in mpsa)
        self.assertTrue(np.all(np.abs(sol - sol_full) < 5e-4))

    def check_solution(self, gb):
        for e, d in gb.edges():
            mg = d["mortar_grid"]
            gs, gm = gb.nodes_of_edge(e)

            ds = gb.node_props(gs)
            dm = gb.node_props(gm)

            us = ds["u"]
            um = dm["u"]
            lam = d["lam_u"]
            ps = ds["p"]
            pm = dm["p"]

            bdcs = ds[pp.DISCRETIZATION_MATRICES][self.kw]["bound_displacement_cell"]
            bdcm = dm[pp.DISCRETIZATION_MATRICES][self.kw]["bound_displacement_cell"]
            bdfs = ds[pp.DISCRETIZATION_MATRICES][self.kw]["bound_displacement_face"]
            bdfm = dm[pp.DISCRETIZATION_MATRICES][self.kw]["bound_displacement_face"]
            bdps = ds[pp.DISCRETIZATION_MATRICES][self.kw][
                "bound_displacement_pressure"
            ]
            bdpm = dm[pp.DISCRETIZATION_MATRICES][self.kw][
                "bound_displacement_pressure"
            ]

            bc_val_s = ds[pp.PARAMETERS][self.kw]["bc_values"]
            bc_val_m = dm[pp.PARAMETERS][self.kw]["bc_values"]

            RW = d[pp.PARAMETERS][self.kw]["robin_weight"]
            MW = d[pp.PARAMETERS][self.kw]["mortar_weight"]
            rhs = d[pp.PARAMETERS][self.kw]["robin_rhs"].reshape(
                (gs.dim, -1), order="F"
            )
            slv2mrt_nd = sps.kron(mg.slave_to_mortar_int(), sps.eye(gs.dim)).tocsr()
            mstr2mrt_nd = sps.kron(mg.master_to_mortar_int(), sps.eye(gs.dim)).tocsr()

            hf2fs = pp.fvutils.map_hf_2_f(g=gs) / 2
            hf2fm = pp.fvutils.map_hf_2_f(g=gm) / 2
            jump_u = (
                slv2mrt_nd
                * hf2fs
                * (bdcs * us + bdfs * (bc_val_s + slv2mrt_nd.T * lam) + bdps * ps)
                - mstr2mrt_nd
                * hf2fm
                * (bdcm * um - bdfm * (bc_val_m + mstr2mrt_nd.T * lam) + bdpm * pm)
            ).reshape((gs.dim, -1), order="F")
            lam_nd = lam.reshape((gs.dim, -1), order="F")
            for i in range(len(RW)):
                rhs_robin = MW[i].dot(lam_nd[:, i]) + RW[i].dot(jump_u[:, i])
                self.assertTrue(np.allclose(rhs_robin, rhs[:, i]))

    def assign_discretization(self, gb):
        for g, d in gb:
            d[pp.DISCRETIZATION] = {
                "u": {"div_sigma": pp.Mpsa(self.kw)},
                "p": {
                    "flux": pp.Mpfa(self.kw_f),
                    "mass": pp.MassMatrix(self.kw_f),
                    "stab": pp.BiotStabilization(self.kw_f),
                },
                "u_p": {"grad_p": pp.GradP(self.kw)},
                "p_u": {"div_u": pp.DivD(self.kw)},
            }

            d[pp.PRIMARY_VARIABLES] = {"u": {"cells": g.dim}, "p": {"cells": 1}}

        gradP_disp = pp.numerics.interface_laws.elliptic_interface_laws.RobinContactBiotPressure(
            self.kw, pp.numerics.fv.biot.GradP(self.kw)
        )
        div_u_lam = pp.numerics.interface_laws.elliptic_interface_laws.DivU_StressMortar(
            self.kw, pp.numerics.fv.biot.DivD(self.kw)
        )

        for e, d in gb.edges():
            g1, g2 = gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES] = {"lam_u": {"cells": 2}, "lam_p": {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                "u_contribution": {
                    g1: ("u", "div_sigma"),
                    g2: ("u", "div_sigma"),
                    (g1, g2): ("lam_u", pp.RobinContact(self.kw, pp.Mpsa(self.kw))),
                },
                "p_contribution_to_displacement": {
                    g1: ("p", "flux"),
                    g2: ("p", "flux"),
                    (g1, g2): ("lam_u", gradP_disp),
                },
                "lam_u_contr_2_div_u": {
                    g1: ("p", "flux"),
                    g2: ("p", "flux"),
                    (g1, g2): ("lam_u", div_u_lam),
                },
                "lam_p": {
                    g1: ("p", "flux"),
                    g2: ("p", "flux"),
                    (g1, g2): (
                        "lam_p",
                        pp.FluxPressureContinuity(self.kw_f, pp.Mpfa(self.kw_f)),
                    ),
                },
            }

    def assign_parameters(self, gb, mortar_weight, robin_weight, rhs):
        for g, d in gb:
            if g.grid_num == 1:
                dir_faces = g.face_centers[0] < 1e-10
            elif g.grid_num == 2:
                dir_faces = g.face_centers[0] > 2 - 1e-10
            elif g.grid_num == 3:
                dir_faces = (g.face_centers[0] < 1e-10) + (
                    g.face_centers[0] > 2 - 1e-10
                )
            bc_val = np.zeros((g.dim, g.num_faces))
            bc_val[0, dir_faces] = 0.1 * g.face_centers[0, dir_faces]
            bc = pp.BoundaryConditionVectorial(g, dir_faces, "dir")
            C = pp.FourthOrderTensor(
                gb.dim_max(), np.ones(g.num_cells), np.ones(g.num_cells)
            )
            alpha = 1 / np.pi
            data = {
                "bc": bc,
                "bc_values": bc_val.ravel("F"),
                "fourth_order_tensor": C,
                "source": np.zeros(g.num_cells * g.dim),
                "inverter": "python",
                "biot_alpha": alpha,
                "state": np.zeros(g.dim * g.num_cells),
            }
            data_f = {
                "bc": pp.BoundaryCondition(g, g.get_boundary_faces(), "dir"),
                "bc_values": np.zeros(g.num_faces),
                "second_order_tensor": pp.SecondOrderTensor(
                    g.dim, np.ones(g.num_cells)
                ),
                "inverter": "python",
                "aperture": np.ones(g.num_cells),
                "biot_alpha": alpha,
                "mass_weight": 1e-1,
                "state": np.zeros(g.num_cells),
            }
            pp.initialize_data(g, d, self.kw, data)
            pp.initialize_data(g, d, self.kw_f, data_f)

        for _, d in gb.edges():
            mg = d["mortar_grid"]
            MW = sps.diags(mortar_weight)
            RW = sps.diags(robin_weight)
            data = {
                "mortar_weight": [MW] * mg.num_cells,
                "robin_weight": [RW] * mg.num_cells,
                "robin_rhs": np.tile(rhs, (mg.num_cells)),
                "state": np.zeros((mg.dim + 1) * mg.num_cells),
            }
            pp.initialize_data(mg, d, self.kw, data)


def define_gb():
    """
    Construct grids
    """
    g_s = pp.CartGrid([1, 2], [1, 2])
    g_m = pp.CartGrid([1, 2], [1, 2])
    g_full = pp.CartGrid([2, 2], [2, 2])
    g_m.nodes[0] += 1
    g_s.compute_geometry()
    g_m.compute_geometry()
    g_full.compute_geometry()

    g_s.grid_num = 1
    g_m.grid_num = 2
    g_full.grid_num = 3

    gb = pp.GridBucket()
    gb_full = pp.GridBucket()
    gb.add_nodes([g_s, g_m])
    gb_full.add_nodes([g_full])

    contact_s = np.where(g_s.face_centers[0] > 1 - 1e-10)[0]
    contact_m = np.where(g_m.face_centers[0] < 1 + 1e-10)[0]
    data = np.ones(contact_s.size, dtype=np.bool)

    shape = (g_s.num_faces, g_m.num_faces)
    slave_master = sps.csc_matrix((data, (contact_m, contact_s)), shape=shape)

    mortar_grid, _, _ = pp.grids.partition.extract_subgrid(g_s, contact_s, faces=True)

    gb.add_edge([g_s, g_m], slave_master)

    gb.assign_node_ordering()
    gb_full.assign_node_ordering()

    # Slave and master is defined by the node number.
    # In python 3.5 the node-nombering does not follow the one given in gb.add_edge
    # I guess also the face_face mapping given on the edge also should change,
    # but this is not used
    g_1, _ = gb.nodes_of_edge([g_s, g_m])
    if g_1.grid_num == 2:
        g_m = g_s
        g_s = g_1
        slave_master = slave_master.T

    mg = pp.BoundaryMortar(mortar_grid.dim, mortar_grid, slave_master.T)
    gb.set_edge_prop([g_s, g_m], "mortar_grid", mg)
    return gb, gb_full


if __name__ == "__main__":
    unittest.main()
