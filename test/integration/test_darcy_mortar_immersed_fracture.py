# -*- coding: utf-8 -*-
"""
Test of Robin-type coupling for a grid with immersed fractures.
The non-immersed case is kind of tested in the module
    test/unit/test_darcy_mortar.

"""
import unittest
from test import test_grids, test_utils

import numpy as np
import scipy.sparse as sps

import porepy as pp


class TestImmersedFracture(unittest.TestCase):
    def create_grid(
        self, num_1d=3, pert_node=False, flip_normals=False, rotate_fracture=False
    ):

        generator = test_grids.SimplexGrid2dDomainOneImmersedFracture()

        gb = generator.generate_grid(
            num_1d=num_1d,
            pert_node=pert_node,
            flip_normals=flip_normals,
            rotate_fracture=rotate_fracture,
        )

        return gb

    def set_params(self, gb, kn, kf):

        kw = "flow"

        for g, d in gb:
            parameter_dictionary = {}

            aperture = np.power(1e-2, gb.dim_max() - g.dim)
            parameter_dictionary["aperture"] = aperture * np.ones(g.num_cells)

            b_val = np.zeros(g.num_faces)
            if g.dim == 2:
                bound_faces = pp.face_on_side(g, ["ymin", "ymax"])
                bound_faces = np.hstack((bound_faces[0], bound_faces[1]))
                labels = np.array(["dir"] * bound_faces.size)

                parameter_dictionary["bc"] = pp.BoundaryCondition(
                    g, bound_faces, labels
                )

                y_max_faces = pp.face_on_side(g, "ymax")[0]
                b_val[y_max_faces] = 1

                perm = pp.SecondOrderTensor(kxx=np.ones(g.num_cells))
                parameter_dictionary["second_order_tensor"] = perm

            else:
                perm = pp.SecondOrderTensor(kxx=kf * np.ones(g.num_cells))
                parameter_dictionary["second_order_tensor"] = perm

                parameter_dictionary["bc"] = pp.BoundaryCondition(g)
            parameter_dictionary["bc_values"] = b_val
            parameter_dictionary["mpfa_inverter"] = "python"

            d[pp.PARAMETERS] = pp.Parameters(g, [kw], [parameter_dictionary])
            d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

        gb.add_edge_props("kn")
        for _, d in gb.edges():
            mg = d["mortar_grid"]
            flow_dictionary = {"normal_diffusivity": kn * np.ones(mg.num_cells)}
            d[pp.PARAMETERS] = pp.Parameters(
                keywords=["flow"], dictionaries=[flow_dictionary]
            )
            d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

    def verify_cv(self, gb, tol=1e-6):
        # The tolerance level here is a bit touchy: With an unstructured grid,
        # and with the flux between subdomains computed as differences between
        # point pressures, uniform flow may not be reproduced if the meshes
        # are not matching (one may get lucky, though). Thus the coarse error
        # tolerance. The current value turned out to be sufficient for all
        # tests considered herein.
        for g, d in gb.nodes():
            p = d[pp.STATE]["pressure"]
            self.assertTrue(np.allclose(p, g.cell_centers[1], rtol=tol, atol=tol))

    def _solve(self, gb, method, key):
        assembler = test_utils.setup_flow_assembler(gb, method, key)
        assembler.discretize()
        A_flow, b_flow = assembler.assemble_matrix_rhs()
        up = sps.linalg.spsolve(A_flow, b_flow)
        assembler.distribute_variable(up)

    def run_mpfa(self, gb):
        key = "flow"
        method = pp.Mpfa(key)
        self._solve(gb, method, key)

    def run_vem(self, gb):
        key = "flow"
        method = pp.MVEM(key)
        self._solve(gb, method, key)
        for g, d in gb:
            d[pp.STATE]["darcy_flux"] = d[pp.STATE]["pressure"][: g.num_faces]
            d[pp.STATE]["pressure"] = d[pp.STATE]["pressure"][g.num_faces :]

    def run_RT0(self, gb):
        key = "flow"
        method = pp.RT0(key)
        self._solve(gb, method, key)
        for g, d in gb:
            d[pp.STATE]["darcy_flux"] = d[pp.STATE]["pressure"][: g.num_faces]
            d[pp.STATE]["pressure"] = d[pp.STATE]["pressure"][g.num_faces :]

    def test_mpfa_blocking_fracture(self):
        # blocking fracture, essentially uniform flow in a 2d domain
        gb = self.create_grid()
        self.set_params(gb, kn=1e-6, kf=1e-6)

        self.run_mpfa(gb)
        g_2d = gb.grids_of_dimension(2)[0]
        d = gb.node_props(g_2d)
        p = d[pp.STATE]["pressure"]
        self.assertTrue(np.allclose(p, g_2d.cell_centers[1], rtol=1e-5))

    def test_mvem_blocking_fracture(self):
        # blocking fracture, essentially uniform flow in a 2d domain
        gb = self.create_grid()
        self.set_params(gb, kn=1e-6, kf=1e-6)

        self.run_vem(gb)
        g_2d = gb.grids_of_dimension(2)[0]
        d = gb.node_props(g_2d)
        p = d[pp.STATE]["pressure"]
        self.assertTrue(np.allclose(p, g_2d.cell_centers[1], rtol=1e-5))

    def test_rt0_blocking_fracture(self):
        # blocking fracture, essentially uniform flow in a 2d domain
        gb = self.create_grid()
        self.set_params(gb, kn=1e-6, kf=1e-6)

        self.run_RT0(gb)
        g_2d = gb.grids_of_dimension(2)[0]
        d = gb.node_props(g_2d)
        p = d[pp.STATE]["pressure"]
        self.assertTrue(np.allclose(p, g_2d.cell_centers[1], rtol=1e-5))

    def test_mpfa_flip_normal(self):
        # With and without flipped normal vector. Results should be the same
        gb = self.create_grid()
        self.set_params(gb, kn=1e4, kf=1e3)

        self.run_mpfa(gb)
        g_2d = gb.grids_of_dimension(2)[0]
        d = gb.node_props(g_2d)
        p = d[pp.STATE]["pressure"]
        gb = self.create_grid()
        self.set_params(gb, kn=1e4, kf=1e3)

        self.run_mpfa(gb)
        g_2d = gb.grids_of_dimension(2)[0]
        d = gb.node_props(g_2d)
        p_flipped = d[pp.STATE]["pressure"]

        self.assertTrue(np.allclose(p, p_flipped, rtol=1e-10))

    def test_mvem_flip_normal(self):
        # With and without flipped normal vector. Results should be the same
        gb = self.create_grid()
        self.set_params(gb, kn=1e4, kf=1e3)

        self.run_vem(gb)
        g_2d = gb.grids_of_dimension(2)[0]
        d = gb.node_props(g_2d)
        p = d[pp.STATE]["pressure"]
        gb = self.create_grid()
        self.set_params(gb, kn=1e4, kf=1e3)

        self.run_vem(gb)
        g_2d = gb.grids_of_dimension(2)[0]
        d = gb.node_props(g_2d)
        p_flipped = d[pp.STATE]["pressure"]

        self.assertTrue(np.allclose(p, p_flipped, rtol=1e-10))

    def test_RT0_flip_normal(self):
        # With and without flipped normal vector. Results should be the same
        gb = self.create_grid()
        self.set_params(gb, kn=1e4, kf=1e3)

        self.run_RT0(gb)
        g_2d = gb.grids_of_dimension(2)[0]
        d = gb.node_props(g_2d)
        p = d[pp.STATE]["pressure"]

        gb = self.create_grid()
        self.set_params(gb, kn=1e4, kf=1e3)

        self.run_RT0(gb)
        g_2d = gb.grids_of_dimension(2)[0]
        d = gb.node_props(g_2d)
        p_flipped = d[pp.STATE]["pressure"]

        self.assertTrue(np.allclose(p, p_flipped, rtol=1e-10))


if __name__ == "__main__":
    unittest.main()
