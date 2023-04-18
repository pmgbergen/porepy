import unittest

import scipy.sparse as sps

import porepy as pp
import tests.common.flow_benchmark_2d_geiger_setup as setup


class TestVEMOnBenchmark(unittest.TestCase):
    def solve(self, kf, description, is_coarse=False):
        mdg, domain = pp.md_grids_2d.benchmark_regular(
            {"mesh_size_frac": 0.045}, is_coarse
        )
        # Assign parameters
        setup.add_data(mdg, domain, kf)
        key = "flow"

        method = pp.MVEM(key)

        coupler = pp.RobinCoupling(key, method)

        for sd, sd_data in mdg.subdomains(return_data=True):
            sd_data[pp.PRIMARY_VARIABLES] = {"pressure": {"cells": 1, "faces": 1}}
            sd_data[pp.DISCRETIZATION] = {"pressure": {"diffusive": method}}
        for intf, intf_data in mdg.interfaces(return_data=True):
            sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)
            intf_data[pp.PRIMARY_VARIABLES] = {"mortar_solution": {"cells": 1}}
            intf_data[pp.COUPLING_DISCRETIZATION] = {
                "lambda": {
                    sd_primary: ("pressure", "diffusive"),
                    sd_secondary: ("pressure", "diffusive"),
                    intf: ("mortar_solution", coupler),
                }
            }
            intf_data[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

        assembler = pp.Assembler(mdg)

        # Discretize
        assembler.discretize()
        A, b = assembler.assemble_matrix_rhs()
        p = sps.linalg.spsolve(A, b)

        assembler.distribute_variable(p)

        for sd, sd_data in mdg.subdomains(return_data=True):
            darcy_flux_values = sd_data[pp.TIME_STEP_SOLUTIONS]["pressure"][0][
                : sd.num_faces
            ]
            pp.set_solution_values(
                name="darcy_flux",
                values=darcy_flux_values,
                data=sd_data,
                time_step_index=0,
            )

            pressure_values = sd_data[pp.TIME_STEP_SOLUTIONS]["pressure"][0][
                sd.num_faces :
            ]
            pp.set_solution_values(
                name="pressure", values=pressure_values, data=sd_data, time_step_index=0
            )

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
        mdg, domain = pp.md_grids_2d.benchmark_regular({"mesh_size_frac": 0.045})
        # Assign parameters
        setup.add_data(mdg, domain, kf)

        key = "flow"
        if multi_point:
            method = pp.Mpfa(key)
        else:
            method = pp.Tpfa(key)

        coupler = pp.RobinCoupling(key, method)

        for sd, sd_data in mdg.subdomains(return_data=True):
            sd_data[pp.PRIMARY_VARIABLES] = {"pressure": {"cells": 1}}
            sd_data[pp.DISCRETIZATION] = {"pressure": {"diffusive": method}}
        for intf, intf_data in mdg.interfaces(return_data=True):
            sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)
            intf_data[pp.PRIMARY_VARIABLES] = {"mortar_solution": {"cells": 1}}
            intf_data[pp.COUPLING_DISCRETIZATION] = {
                "lambda": {
                    sd_primary: ("pressure", "diffusive"),
                    sd_secondary: ("pressure", "diffusive"),
                    intf: ("mortar_solution", coupler),
                }
            }
            intf_data[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

        assembler = pp.Assembler(mdg)

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
