import scipy.sparse as sps
import porepy as pp
import unittest

import test.common.flow_benchmark_2d_geiger_setup as setup


class TestVEMOnBenchmark(unittest.TestCase):
    def solve(self, kf, description, is_coarse=False, if_export=False):
        mesh_size = 0.045
        gb, domain = setup.make_grid_bucket(mesh_size, is_coarse)
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

        if if_export:
            save = pp.Exporter(gb, "vem", folder="vem_" + description)
            # save.write_vtk(["pressure", "P0u"])
            save.write_vtk(["pressure"])

    def test_vem_blocking(self):
        kf = 1e-4
        if_export = False
        self.solve(kf, "blocking", if_export=if_export)

    def test_vem_blocking_coarse(self):
        kf = 1e-4
        if_export = False
        self.solve(kf, "blocking_coarse", is_coarse=True, if_export=if_export)

    def test_vem_permeable(self):
        kf = 1e4
        if_export = False
        self.solve(kf, "permeable", if_export=if_export)

    def test_vem_permeable_coarse(self):
        kf = 1e4
        if_export = False
        self.solve(kf, "permeable_coarse", is_coarse=True, if_export=if_export)


class TestFVOnBenchmark(unittest.TestCase):
    def solve(self, kf, description, multi_point, if_export=False):
        mesh_size = 0.045
        gb, domain = setup.make_grid_bucket(mesh_size)
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

        if if_export:
            save = pp.Exporter(gb, "fv", folder="fv_" + description)
            save.write_vtk(["pressure"])

    def test_tpfa_blocking(self):
        kf = 1e-4
        if_export = True
        self.solve(kf, "blocking_tpfa", multi_point=False, if_export=if_export)

    def test_tpfa_permeable(self):
        kf = 1e4
        if_export = True
        self.solve(kf, "permeable_tpfa", multi_point=False, if_export=if_export)

    def test_mpfa_blocking(self):
        kf = 1e-4
        if_export = True
        self.solve(kf, "blocking_mpfa", multi_point=True, if_export=if_export)

    def test_mpfa_permeable(self):
        kf = 1e4
        if_export = True
        self.solve(kf, "permeable_mpfa", multi_point=True, if_export=if_export)


if __name__ == "__main__":
    unittest.main()
