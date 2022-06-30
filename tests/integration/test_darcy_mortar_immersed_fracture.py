# -*- coding: utf-8 -*-
"""
Test of Robin-type coupling for a grid with immersed fractures.
The non-immersed case is kind of tested in the module
    test/unit/test_darcy_mortar.

"""
import unittest
from tests import test_grids, test_utils

import numpy as np
import scipy.sparse as sps

import porepy as pp


class TestImmersedFracture(unittest.TestCase):
    def create_grid(
        self, num_1d=3, pert_node=False, flip_normals=False, rotate_fracture=False
    ):

        generator = test_grids.SimplexGrid2dDomainOneImmersedFracture()

        mdg = generator.generate_grid(
            num_1d=num_1d,
            pert_node=pert_node,
            flip_normals=flip_normals,
            rotate_fracture=rotate_fracture,
        )

        return mdg

    def set_params(self, mdg, kn, kf):

        kw = "flow"

        for sd, data in mdg.subdomains(return_data=True):
            parameter_dictionary = {}

            aperture = np.power(1e-2, mdg.dim_max() - sd.dim)
            parameter_dictionary["aperture"] = aperture * np.ones(sd.num_cells)

            b_val = np.zeros(sd.num_faces)
            if sd.dim == 2:
                bound_faces = pp.face_on_side(sd, ["ymin", "ymax"])
                bound_faces = np.hstack((bound_faces[0], bound_faces[1]))
                labels = np.array(["dir"] * bound_faces.size)

                parameter_dictionary["bc"] = pp.BoundaryCondition(
                    sd, bound_faces, labels
                )

                y_max_faces = pp.face_on_side(sd, "ymax")[0]
                b_val[y_max_faces] = 1

                perm = pp.SecondOrderTensor(kxx=np.ones(sd.num_cells))
                parameter_dictionary["second_order_tensor"] = perm

            else:
                perm = pp.SecondOrderTensor(kxx=kf * np.ones(sd.num_cells))
                parameter_dictionary["second_order_tensor"] = perm

                parameter_dictionary["bc"] = pp.BoundaryCondition(sd)
            parameter_dictionary["bc_values"] = b_val
            parameter_dictionary["mpfa_inverter"] = "python"

            data[pp.PARAMETERS] = pp.Parameters(sd, [kw], [parameter_dictionary])
            data[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

        for intf, data in mdg.interfaces(return_data=True):
            flow_dictionary = {"normal_diffusivity": kn * np.ones(intf.num_cells)}
            data[pp.PARAMETERS] = pp.Parameters(
                keywords=["flow"], dictionaries=[flow_dictionary]
            )
            data[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

    def verify_cv(self, mdg, tol=1e-6):
        # The tolerance level here is a bit touchy: With an unstructured grid,
        # and with the flux between subdomains computed as differences between
        # point pressures, uniform flow may not be reproduced if the meshes
        # are not matching (one may get lucky, though). Thus the coarse error
        # tolerance. The current value turned out to be sufficient for all
        # tests considered herein.
        for g, d in mdg.nodes():
            p = d[pp.STATE]["pressure"]
            self.assertTrue(np.allclose(p, g.cell_centers[1], rtol=tol, atol=tol))

    def _solve(self, mdg, method, key):
        assembler = test_utils.setup_flow_assembler(mdg, method, key)
        assembler.discretize()
        A_flow, b_flow = assembler.assemble_matrix_rhs()
        up = sps.linalg.spsolve(A_flow, b_flow)
        assembler.distribute_variable(up)

    def run_mpfa(self, mdg):
        key = "flow"
        method = pp.Mpfa(key)
        self._solve(mdg, method, key)

    def run_vem(self, mdg):
        key = "flow"
        method = pp.MVEM(key)
        self._solve(mdg, method, key)
        for sd, data in mdg.subdomains(return_data=True):
            data[pp.STATE]["darcy_flux"] = data[pp.STATE]["pressure"][: sd.num_faces]
            data[pp.STATE]["pressure"] = data[pp.STATE]["pressure"][sd.num_faces :]

    def run_RT0(self, mdg):
        key = "flow"
        method = pp.RT0(key)
        self._solve(mdg, method, key)
        for sd, data in mdg.subdomains(return_data=True):
            data[pp.STATE]["darcy_flux"] = data[pp.STATE]["pressure"][: sd.num_faces]
            data[pp.STATE]["pressure"] = data[pp.STATE]["pressure"][sd.num_faces :]

    def test_mpfa_blocking_fracture(self):
        # blocking fracture, essentially uniform flow in a 2d domain
        mdg = self.create_grid()
        self.set_params(mdg, kn=1e-6, kf=1e-6)

        self.run_mpfa(mdg)
        sd_2d = list(mdg.subdomains(dim=2))[0]
        data = mdg.subdomain_data(sd_2d)
        p = data[pp.STATE]["pressure"]
        self.assertTrue(np.allclose(p, sd_2d.cell_centers[1], rtol=1e-5))

    def test_mvem_blocking_fracture(self):
        # blocking fracture, essentially uniform flow in a 2d domain
        mdg = self.create_grid()
        self.set_params(mdg, kn=1e-6, kf=1e-6)

        self.run_vem(mdg)
        sd_2d = list(mdg.subdomains(dim=2))[0]
        data = mdg.subdomain_data(sd_2d)
        p = data[pp.STATE]["pressure"]
        self.assertTrue(np.allclose(p, sd_2d.cell_centers[1], rtol=1e-5))

    def test_rt0_blocking_fracture(self):
        # blocking fracture, essentially uniform flow in a 2d domain
        mdg = self.create_grid()
        self.set_params(mdg, kn=1e-6, kf=1e-6)

        self.run_RT0(mdg)
        sd_2d = list(mdg.subdomains(dim=2))[0]
        data = mdg.subdomain_data(sd_2d)
        p = data[pp.STATE]["pressure"]
        self.assertTrue(np.allclose(p, sd_2d.cell_centers[1], rtol=1e-5))

    def test_mpfa_flip_normal(self):
        # With and without flipped normal vector. Results should be the same
        mdg = self.create_grid()
        self.set_params(mdg, kn=1e4, kf=1e3)

        self.run_mpfa(mdg)
        sd_2d = list(mdg.subdomains(dim=2))[0]
        data = mdg.subdomain_data(sd_2d)
        p = data[pp.STATE]["pressure"]
        mdg = self.create_grid()
        self.set_params(mdg, kn=1e4, kf=1e3)

        self.run_mpfa(mdg)
        sd_2d = list(mdg.subdomains(dim=2))[0]
        data = mdg.subdomain_data(sd_2d)
        p_flipped = data[pp.STATE]["pressure"]

        self.assertTrue(np.allclose(p, p_flipped, rtol=1e-10))

    def test_mvem_flip_normal(self):
        # With and without flipped normal vector. Results should be the same
        mdg = self.create_grid()
        self.set_params(mdg, kn=1e4, kf=1e3)

        self.run_vem(mdg)
        sd_2d = list(mdg.subdomains(dim=2))[0]
        data = mdg.subdomain_data(sd_2d)
        p = data[pp.STATE]["pressure"]

        mdg = self.create_grid()
        self.set_params(mdg, kn=1e4, kf=1e3)

        self.run_vem(mdg)
        sd_2d = list(mdg.subdomains(dim=2))[0]
        data = mdg.subdomain_data(sd_2d)
        p_flipped = data[pp.STATE]["pressure"]

        self.assertTrue(np.allclose(p, p_flipped, rtol=1e-10))

    def test_RT0_flip_normal(self):
        # With and without flipped normal vector. Results should be the same
        mdg = self.create_grid()
        self.set_params(mdg, kn=1e4, kf=1e3)

        self.run_RT0(mdg)
        sd_2d = list(mdg.subdomains(dim=2))[0]
        data = mdg.subdomain_data(sd_2d)
        p = data[pp.STATE]["pressure"]

        mdg = self.create_grid()
        self.set_params(mdg, kn=1e4, kf=1e3)

        self.run_RT0(mdg)
        sd_2d = list(mdg.subdomains(dim=2))[0]
        data = mdg.subdomain_data(sd_2d)
        p_flipped = data[pp.STATE]["pressure"]

        self.assertTrue(np.allclose(p, p_flipped, rtol=1e-10))


if __name__ == "__main__":
    unittest.main()
