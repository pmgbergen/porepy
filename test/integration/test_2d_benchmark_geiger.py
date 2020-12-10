import test.common.flow_benchmark_2d_geiger_setup as setup
import unittest

import scipy.sparse as sps

import porepy as pp


class TestVEMOnBenchmark(unittest.TestCase):
    def solve(self, kf, description, is_coarse=False):
        gb, domain = pp.grid_buckets_2d.benchmark_regular(
            {"mesh_size_frac": 0.045}, is_coarse
        )
        # Assign parameters
        setup.add_data(gb, domain, kf)
        key = "flow"

        method = pp.MVEM(key)

        coupler = pp.RobinCoupling(key, method)

        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {"pressure": {"cells": 1, "faces": 1}}
            d[pp.DISCRETIZATION] = {"pressure": {"diffusive": method}}
        for e, d in gb.edges():
            g1, g2 = gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES] = {"mortar_solution": {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                "lambda": {
                    g1: ("pressure", "diffusive"),
                    g2: ("pressure", "diffusive"),
                    e: ("mortar_solution", coupler),
                }
            }
            d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

        assembler = pp.Assembler(gb)

        # Discretize
        assembler.discretize()
        A, b = assembler.assemble_matrix_rhs()
        p = sps.linalg.spsolve(A, b)

        assembler.distribute_variable(p)

        for g, d in gb:
            d[pp.STATE]["darcy_flux"] = d[pp.STATE]["pressure"][: g.num_faces]
            d[pp.STATE]["pressure"] = d[pp.STATE]["pressure"][g.num_faces :]

    def test_vem_blocking(self):
        kf = 1e-4
        self.solve(kf, "blocking")

    def test_vem_blocking_coarse(self):
        kf = 1e-4
        self.solve(kf, "blocking_coarse", is_coarse=True)

    def test_vem_permeable(self):
        kf = 1e4
        self.solve(kf, "permeable")

    def test_vem_permeable_coarse(self):
        kf = 1e4
        self.solve(kf, "permeable_coarse", is_coarse=True)


class TestFVOnBenchmark(unittest.TestCase):
    def solve(self, kf, description, multi_point):
        gb, domain = pp.grid_buckets_2d.benchmark_regular({"mesh_size_frac": 0.045})
        # Assign parameters
        setup.add_data(gb, domain, kf)

        key = "flow"
        if multi_point:
            method = pp.Mpfa(key)
        else:
            method = pp.Tpfa(key)

        coupler = pp.RobinCoupling(key, method)

        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {"pressure": {"cells": 1}}
            d[pp.DISCRETIZATION] = {"pressure": {"diffusive": method}}
        for e, d in gb.edges():
            g1, g2 = gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES] = {"mortar_solution": {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                "lambda": {
                    g1: ("pressure", "diffusive"),
                    g2: ("pressure", "diffusive"),
                    e: ("mortar_solution", coupler),
                }
            }
            d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

        assembler = pp.Assembler(gb)

        # Discretize
        assembler.discretize()
        A, b = assembler.assemble_matrix_rhs()
        p = sps.linalg.spsolve(A, b)

        assembler.distribute_variable(p)

    def test_tpfa_blocking(self):
        kf = 1e-4
        self.solve(kf, "blocking_tpfa", multi_point=False)

    def test_tpfa_permeable(self):
        kf = 1e4
        self.solve(kf, "permeable_tpfa", multi_point=False)

    def test_mpfa_blocking(self):
        kf = 1e-4
        self.solve(kf, "blocking_mpfa", multi_point=True)

    def test_mpfa_permeable(self):
        kf = 1e4
        self.solve(kf, "permeable_mpfa", multi_point=True)


if __name__ == "__main__":
    unittest.main()
