""" Test interface discretization of darcy flow problem
"""
import unittest

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.fracs.utils import pts_edges_to_linefractures
from tests import test_utils


class TestMortar2dSingleFractureCartesianGrid(unittest.TestCase):
    def set_param_flow(self, mdg, no_flow=False, kn=1e3, method="mpfa"):
        # Set up flow field with uniform flow in y-direction
        kw = "flow"
        for sd, data in mdg.subdomains(return_data=True):
            parameter_dictionary = {}

            perm = pp.SecondOrderTensor(kxx=np.ones(sd.num_cells))
            parameter_dictionary["second_order_tensor"] = perm

            b_val = np.zeros(sd.num_faces)
            if sd.dim == 2:
                bound_faces = pp.face_on_side(sd, ["ymin", "ymax"])
                if no_flow:
                    b_val[bound_faces[0]] = 1
                    b_val[bound_faces[1]] = 1
                bound_faces = np.hstack((bound_faces[0], bound_faces[1]))
                labels = np.array(["dir"] * bound_faces.size)
                parameter_dictionary["bc"] = pp.BoundaryCondition(
                    sd, bound_faces, labels
                )

                y_max_faces = pp.face_on_side(sd, "ymax")[0]
                b_val[y_max_faces] = 1
            else:
                parameter_dictionary["bc"] = pp.BoundaryCondition(sd)
            parameter_dictionary["bc_values"] = b_val
            parameter_dictionary["mpfa_inverter"] = "python"

            data[pp.PARAMETERS] = pp.Parameters(sd, [kw], [parameter_dictionary])
            data[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

        for mg, data in mdg.interfaces(return_data=True):
            flow_dictionary = {"normal_diffusivity": 2 * kn * np.ones(mg.num_cells)}
            data[pp.PARAMETERS] = pp.Parameters(
                keywords=["flow"], dictionaries=[flow_dictionary]
            )
            data[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

        discretization_key = kw + "_" + pp.DISCRETIZATION

        for sd, data in mdg.subdomains(return_data=True):
            # Choose discretization and define the solver
            if method == "mpfa":
                discr = pp.Mpfa(kw)
            elif method == "mvem":
                discr = pp.MVEM(kw)
            else:
                discr = pp.Tpfa(kw)

            data[discretization_key] = discr

        for _, data in mdg.interfaces(return_data=True):
            data[discretization_key] = pp.RobinCoupling(kw, discr)

    def set_grids(self, N, num_nodes_mortar, num_nodes_1d, physdims=[1, 1]):
        f1 = np.array([[0, physdims[0]], [0.5, 0.5]])

        mdg = pp.meshing.cart_grid([f1], N, **{"physdims": physdims})
        mdg.compute_geometry()

        for intf in mdg.interfaces():
            pass

        new_side_grids = {
            s: pp.refinement.remesh_1d(g, num_nodes=num_nodes_mortar)
            for s, g in intf.side_grids.items()
        }

        intf.update_mortar(new_side_grids, tol=1e-4)

        # refine the 1d-physical grid
        old_g = mdg.interface_to_subdomain_pair(intf)[1]
        new_g = pp.refinement.remesh_1d(old_g, num_nodes=num_nodes_1d)
        new_g.compute_geometry()

        mdg.replace_subdomains_and_interfaces(sd_map={old_g: new_g})

        intf.update_secondary(new_g, tol=1e-4)

        return mdg

    def solve(self, mdg, method=None):
        key = "flow"
        if method is None:
            discretization = pp.Tpfa(key)
        elif method == "mpfa":
            discretization = pp.Mpfa(key)
        elif method == "mvem":
            discretization = pp.MVEM(key)
        assembler = test_utils.setup_flow_assembler(mdg, discretization, key)
        assembler.discretize()
        A_flow, b_flow = assembler.assemble_matrix_rhs()
        p = sps.linalg.spsolve(A_flow, b_flow)
        assembler.distribute_variable(p)

        if method == "mvem":
            g2d = mdg.subdomains(dim=2)[0]
            p_n = np.zeros(sum([g.num_cells for g in mdg.subdomains()]))
            for g, d in mdg.subdomains(return_data=True):
                if g.dim == 2:
                    p_n[: g2d.num_cells] = d[pp.STATE]["pressure"][g.num_faces :]
                else:
                    p_n[g2d.num_cells :] = d[pp.STATE]["pressure"][g.num_faces :]
                d[pp.STATE]["pressure"] = d[pp.STATE]["pressure"][g.num_faces :]
            p = p_n
        return p

    def verify_cv(self, mdg):
        for g, d in mdg.subdomains(return_data=True):
            p = d[pp.STATE]["pressure"]
            self.assertTrue(np.allclose(p, g.cell_centers[1], rtol=1e-3, atol=1e-3))

    def test_tpfa_matching_grids_no_flow(self):
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(mdg, no_flow=True, method="tpfa")

        p = self.solve(mdg)

        self.assertTrue(np.allclose(p[:3], 1))
        self.assertTrue(np.allclose(p[3:], 0))

    def test_tpfa_matching_grids_refine_1d_no_flow(self):
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(mdg, no_flow=True, method="tpfa")

        p = self.solve(mdg)
        self.assertTrue(np.allclose(p[:4], 1))
        self.assertTrue(np.allclose(p[4:], 0))

    def test_tpfa_matching_grids_refine_mortar_no_flow(self):
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(mdg, no_flow=True, method="tpfa")

        p = self.solve(mdg)
        self.assertTrue(np.allclose(p[:3], 1))
        self.assertTrue(np.allclose(p[3:], 0))

    def test_tpfa_matching_grids_uniform_flow(self):

        kn = 1e4
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(mdg, no_flow=False, method="tpfa", kn=kn)

        self.solve(mdg)

        self.verify_cv(mdg)

    def test_tpfa_matching_grids_refine_1d_uniform_flow(self):

        kn = 1e4
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(mdg, no_flow=False, method="tpfa", kn=kn)

        self.solve(mdg)
        self.verify_cv(mdg)

    def test_tpfa_matching_grids_refine_mortar_uniform_flow(self):

        kn = 1e4
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(mdg, no_flow=False, method="tpfa", kn=kn)

        self.solve(mdg)
        self.verify_cv(mdg)

    def test_tpfa_matching_grids_refine_2d_uniform_flow(self):

        kn = 1e4
        mdg = self.set_grids(N=[2, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(mdg, no_flow=False, method="tpfa", kn=kn)

        self.solve(mdg)
        self.verify_cv(mdg)

    def test_tpfa_matching_grids_uniform_flow_larger_domain(self):

        kn = 1e4
        mdg = self.set_grids(
            N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(mdg, no_flow=False, method="tpfa", kn=kn)

        self.solve(mdg)
        self.verify_cv(mdg)

    def test_tpfa_matching_grids_refine_1d_uniform_flow_larger_domain(self):

        kn = 1e4
        mdg = self.set_grids(
            N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3, physdims=[2, 1]
        )
        self.set_param_flow(mdg, no_flow=False, method="tpfa", kn=kn)

        self.solve(mdg)
        self.verify_cv(mdg)

    def test_tpfa_matching_grids_refine_mortar_uniform_flow_larger_domain(self):

        kn = 1e4
        mdg = self.set_grids(
            N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(mdg, no_flow=False, method="tpfa", kn=kn)

        self.solve(mdg)
        self.verify_cv(mdg)

    def test_tpfa_matching_grids_refine_2d_uniform_flow_larger_domain(self):

        kn = 1e4
        mdg = self.set_grids(
            N=[2, 2], num_nodes_mortar=2, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(mdg, no_flow=False, method="tpfa", kn=kn)

        self.solve(mdg)
        self.verify_cv(mdg)

    def test_mpfa_matching_grids_no_flow(self):
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(mdg, no_flow=True, method="mpfa")

        p = self.solve(mdg, "mpfa")

        self.assertTrue(np.allclose(p[:3], 1))
        self.assertTrue(np.allclose(p[3:], 0))

    def test_mvem_matching_grids_no_flow(self):
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(mdg, no_flow=True, method="mvem")

        p = self.solve(mdg, "mvem")

        self.assertTrue(np.allclose(p[:3], 1))
        self.assertTrue(np.allclose(p[3:], 0))

    def test_mpfa_matching_grids_refine_1d_no_flow(self):
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(mdg, no_flow=True, method="mpfa")

        p = self.solve(mdg, "mpfa")

        self.assertTrue(np.allclose(p[:4], 1))
        self.assertTrue(np.allclose(p[4:], 0))

    def test_mvem_matching_grids_refine_1d_no_flow(self):
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(mdg, no_flow=True, method="mvem")

        p = self.solve(mdg, "mvem")

        self.assertTrue(np.allclose(p[:4], 1))

    def test_mpfa_matching_grids_refine_mortar_no_flow(self):
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(mdg, no_flow=True, method="mpfa")

        p = self.solve(mdg, "mpfa")

        self.assertTrue(np.allclose(p[:3], 1))
        self.assertTrue(np.allclose(p[3:], 0))

    def test_mvem_matching_grids_refine_mortar_no_flow(self):
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(mdg, no_flow=True, method="mvem")

        p = self.solve(mdg, "mvem")

        self.assertTrue(np.allclose(p[:3], 1))

    def test_mpfa_matching_grids_uniform_flow(self):

        kn = 1e4
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(mdg, no_flow=False, method="mpfa", kn=kn)

        self.solve(mdg, "mpfa")
        self.verify_cv(mdg)

    def test_mvem_matching_grids_uniform_flow(self):

        kn = 1e4
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(mdg, no_flow=False, method="mvem", kn=kn)

        self.solve(mdg, "mvem")
        self.verify_cv(mdg)

    def test_mpfa_matching_grids_refine_1d_uniform_flow(self):

        kn = 1e4
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(mdg, no_flow=False, method="mpfa", kn=kn)

        self.solve(mdg, "mpfa")
        self.verify_cv(mdg)

    def test_mvem_matching_grids_refine_1d_uniform_flow(self):

        kn = 1e4
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(mdg, no_flow=False, method="mvem", kn=kn)

        self.solve(mdg, "mvem")
        self.verify_cv(mdg)

    def test_mpfa_matching_grids_refine_mortar_uniform_flow(self):

        kn = 1e4
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(mdg, no_flow=False, method="mpfa", kn=kn)

        self.solve(mdg, "mpfa")
        self.verify_cv(mdg)

    def test_mvem_matching_grids_refine_mortar_uniform_flow(self):

        kn = 1e4
        mdg = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(mdg, no_flow=False, method="mvem", kn=kn)

        self.solve(mdg, "mvem")
        self.verify_cv(mdg)

    def test_mpfa_matching_grids_refine_2d_uniform_flow(self):

        kn = 1e4
        mdg = self.set_grids(N=[2, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(mdg, no_flow=False, method="mpfa", kn=kn)

        self.solve(mdg, "mpfa")
        self.verify_cv(mdg)

    def test_mvem_matching_grids_refine_2d_uniform_flow(self):

        kn = 1e4
        mdg = self.set_grids(N=[2, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(mdg, no_flow=False, method="mvem", kn=kn)

        self.solve(mdg, "mvem")

        self.verify_cv(mdg)

    def test_mpfa_matching_grids_uniform_flow_larger_domain(self):

        kn = 1e4
        mdg = self.set_grids(
            N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(mdg, no_flow=False, method="mpfa", kn=kn)

        self.solve(mdg, "mpfa")
        self.verify_cv(mdg)

    def test_mvem_matching_grids_uniform_flow_larger_domain(self):

        kn = 1e4
        mdg = self.set_grids(
            N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(mdg, no_flow=False, method="mvem", kn=kn)

        self.solve(mdg, "mvem")
        self.verify_cv(mdg)

    def test_mpfa_matching_grids_refine_1d_uniform_flow_larger_domain(self):

        kn = 1e4
        mdg = self.set_grids(
            N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3, physdims=[2, 1]
        )
        self.set_param_flow(mdg, no_flow=False, method="mpfa", kn=kn)

        self.solve(mdg, "mpfa")
        self.verify_cv(mdg)

    def test_mvem_matching_grids_refine_1d_uniform_flow_larger_domain(self):

        kn = 1e4
        mdg = self.set_grids(
            N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3, physdims=[2, 1]
        )
        self.set_param_flow(mdg, no_flow=False, method="mvem", kn=kn)

        self.solve(mdg, "mvem")
        self.verify_cv(mdg)

    def test_mpfa_matching_grids_refine_mortar_uniform_flow_larger_domain(self):

        kn = 1e4
        mdg = self.set_grids(
            N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(mdg, no_flow=False, method="mpfa", kn=kn)

        self.solve(mdg, "mpfa")
        self.verify_cv(mdg)

    def test_mvem_matching_grids_refine_mortar_uniform_flow_larger_domain(self):

        kn = 1e4
        mdg = self.set_grids(
            N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(mdg, no_flow=False, method="mvem", kn=kn)

        self.solve(mdg, "mvem")
        self.verify_cv(mdg)

    def test_mpfa_matching_grids_refine_2d_uniform_flow_larger_domain(self):

        kn = 1e4
        mdg = self.set_grids(
            N=[2, 2], num_nodes_mortar=2, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(mdg, no_flow=False, method="mpfa", kn=kn)

        self.solve(mdg, "mpfa")
        self.verify_cv(mdg)

    def test_mvem_matching_grids_refine_2d_uniform_flow_larger_domain(self):

        kn = 1e4
        mdg = self.set_grids(
            N=[2, 2], num_nodes_mortar=4, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(mdg, no_flow=False, method="mvem", kn=kn)

        self.solve(mdg, "mvem")
        self.verify_cv(mdg)


class TestMortar2DSimplexGridStandardMeshing(unittest.TestCase):
    def setup(
        self,
        num_fracs=1,
        remove_tags=False,
        alpha_1d=None,
        alpha_mortar=None,
        alpha_2d=None,
    ):

        domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})

        if num_fracs == 0:
            p = np.zeros((2, 0))
            e = np.zeros((2, 0))
            fractures = pts_edges_to_linefractures(p, e)

        elif num_fracs == 1:
            p = np.array([[0, 1], [0.5, 0.5]])
            e = np.array([[0], [1]])
            fractures = pts_edges_to_linefractures(p, e)

        #            p = [np.array([[0.5, 0.5], [0, 1]])]
        elif num_fracs == 2:
            raise ValueError("Not implemented")

        mesh_size = {"mesh_size_frac": 0.3, "mesh_size_bound": 0.3}
        network = pp.FractureNetwork2d(fractures, domain)
        mdg = network.mesh(mesh_size)

        gmap = {}

        # Refine 2D grid?
        if alpha_2d is not None:
            mesh_size = {
                "mesh_size_frac": 0.3 * alpha_2d,
                "mesh_size_bound": 0.3 * alpha_2d,
            }
            mdgn = network.mesh(mesh_size)
            go = mdg.subdomains(dim=2)[0]
            gn = mdgn.subdomains(dim=2)[0]
            gn.compute_geometry()
            gmap[go] = gn

        # Refine 1d grids
        if alpha_1d is not None:
            for g, d in mdg.subdomains(return_data=True):
                if g.dim == 1:
                    if alpha_1d > 1:
                        num_nodes = 1 + int(alpha_1d) * g.num_cells
                    else:
                        num_nodes = 1 + int(alpha_1d * g.num_cells)
                    num_nodes = 1 + int(alpha_1d * g.num_cells)
                    gmap[g] = pp.refinement.remesh_1d(g, num_nodes=num_nodes)
                    gmap[g].compute_geometry()

        # Refine mortar grid
        mg_map = {}
        if alpha_mortar is not None:
            for mg, d in mdg.interfaces(return_data=True):
                if mg.dim == 1:
                    mg_map[mg] = {}
                    for s, g in mg.side_grids.items():
                        num_nodes = int(g.num_nodes * alpha_mortar)
                        mg_map[mg][s] = pp.refinement.remesh_1d(g, num_nodes=num_nodes)

        mdg.replace_subdomains_and_interfaces(sd_map=gmap, intf_map=mg_map, tol=1e-4)

        self.set_params(mdg)

        return mdg

    def set_params(self, mdg):
        kw = "flow"
        for g, d in mdg.subdomains(return_data=True):
            parameter_dictionary = {}

            perm = pp.SecondOrderTensor(kxx=np.ones(g.num_cells))
            parameter_dictionary["second_order_tensor"] = perm

            parameter_dictionary["source"] = np.zeros(g.num_cells)

            yf = g.face_centers[1]
            bound_faces = [
                np.where(np.abs(yf - 1) < 1e-4)[0],
                np.where(np.abs(yf) < 1e-4)[0],
            ]
            bound_faces = np.hstack((bound_faces[0], bound_faces[1]))
            labels = np.array(["dir"] * bound_faces.size)
            parameter_dictionary["bc"] = pp.BoundaryCondition(g, bound_faces, labels)

            bv = np.zeros(g.num_faces)
            bound_faces = np.where(np.abs(yf - 1) < 1e-4)[0]
            bv[bound_faces] = 1
            parameter_dictionary["bc_values"] = bv
            parameter_dictionary["mpfa_inverter"] = "python"

            d[pp.PARAMETERS] = pp.Parameters(g, [kw], [parameter_dictionary])
            d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

        kn = 1e7
        for mg, d in mdg.interfaces(return_data=True):

            flow_dictionary = {"normal_diffusivity": 2 * kn * np.ones(mg.num_cells)}
            d[pp.PARAMETERS] = pp.Parameters(
                keywords=["flow"], dictionaries=[flow_dictionary]
            )
            d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

    def verify_cv(self, mdg, tol=1e-2):
        # The tolerance level here is a bit touchy: With an unstructured grid,
        # and with the flux between subdomains computed as differences between
        # point pressures, uniform flow may not be reproduced if the meshes
        # are not matching (one may get lucky, though). Thus the coarse error
        # tolerance. The current value turned out to be sufficient for all
        # tests considered herein.
        for g, d in mdg.subdomains(return_data=True):
            p = d[pp.STATE]["pressure"]
            self.assertTrue(np.allclose(p, g.cell_centers[1], rtol=tol, atol=tol))

    def run_mpfa(self, mdg):
        key = "flow"
        method = pp.Mpfa(key)
        assembler = test_utils.setup_flow_assembler(mdg, method, key)
        assembler.discretize()
        A_flow, b_flow = assembler.assemble_matrix_rhs()
        p = sps.linalg.spsolve(A_flow, b_flow)
        assembler.distribute_variable(p)

    def run_vem(self, mdg):
        key = "flow"
        method = pp.MVEM(key)
        assembler = test_utils.setup_flow_assembler(mdg, method, key)
        assembler.discretize()
        A_flow, b_flow = assembler.assemble_matrix_rhs()
        p = sps.linalg.spsolve(A_flow, b_flow)
        assembler.distribute_variable(p)
        for g, d in mdg.subdomains(return_data=True):
            d[pp.STATE]["pressure"] = d[pp.STATE]["pressure"][g.num_faces :]

    def test_mpfa_one_frac(self):
        mdg = self.setup(num_fracs=1)
        self.run_mpfa(mdg)
        self.verify_cv(mdg)

    def test_mpfa_one_frac_refine_2d(self):
        mdg = self.setup(num_fracs=1, alpha_2d=2)
        self.run_mpfa(mdg)
        self.verify_cv(mdg)

    def test_mpfa_one_frac_coarsen_2d(self):
        mdg = self.setup(num_fracs=1, alpha_2d=0.5)
        self.run_mpfa(mdg)
        self.verify_cv(mdg)

    def test_mpfa_one_frac_refine_1d(self):
        mdg = self.setup(num_fracs=1, alpha_1d=2)
        self.run_mpfa(mdg)
        self.verify_cv(mdg)

    def test_mpfa_one_frac_coarsen_1d(self):
        mdg = self.setup(num_fracs=1, alpha_1d=0.5)
        self.run_mpfa(mdg)
        self.verify_cv(mdg)

    def test_mpfa_one_frac_refine_mg(self):
        mdg = self.setup(num_fracs=1, alpha_mortar=2)
        self.run_mpfa(mdg)
        self.verify_cv(mdg)

    def test_mpfa_one_frac_coarsen_mg(self):
        mdg = self.setup(num_fracs=1, alpha_mortar=0.5)
        self.run_mpfa(mdg)
        self.verify_cv(mdg)

    def test_vem_one_frac(self):
        mdg = self.setup(num_fracs=1, remove_tags=True)
        self.run_vem(mdg)
        self.verify_cv(mdg, tol=1e-7)

    def test_vem_one_frac_refine_2d(self):
        mdg = self.setup(num_fracs=1, alpha_2d=2, remove_tags=True)
        self.run_vem(mdg)
        self.verify_cv(mdg)

    def test_vem_one_frac_coarsen_2d(self):
        mdg = self.setup(num_fracs=1, alpha_2d=0.5, remove_tags=True)
        self.run_vem(mdg)
        self.verify_cv(mdg)

    def test_vem_one_frac_refine_1d(self):
        mdg = self.setup(num_fracs=1, alpha_1d=2, remove_tags=True)
        self.run_vem(mdg)
        self.verify_cv(mdg)

    def test_vem_one_frac_coarsen_1d(self):
        mdg = self.setup(num_fracs=1, alpha_1d=0.5, remove_tags=True)
        self.run_vem(mdg)
        self.verify_cv(mdg)

    def test_vem_one_frac_refine_mg(self):
        mdg = self.setup(num_fracs=1, alpha_mortar=2, remove_tags=True)
        self.run_vem(mdg)
        self.verify_cv(mdg)

    def test_vem_one_frac_coarsen_mg(self):
        mdg = self.setup(num_fracs=1, alpha_mortar=0.5, remove_tags=True)
        self.run_vem(mdg)
        self.verify_cv(mdg)


#    def test_mpfa_two_fracs(self):
#        mdg = self.setup(num_fracs=2)
#        self.run_mpfa(mdg)
#        self.verify_cv(mdg)
#
#    def test_mpfa_two_fracs_refine_2d(self):
#        mdg = self.setup(num_fracs=2, alpha_2d=2)
#        self.run_mpfa(mdg)
#        self.verify_cv(mdg)
#
#    def test_mpfa_two_fracs_coarsen_2d(self):
#        mdg = self.setup(num_fracs=2, alpha_2d=0.5)
#        self.run_mpfa(mdg)
#        self.verify_cv(mdg)
#
#    def test_mpfa_two_fracs_refine_1d(self):
#        mdg = self.setup(num_fracs=2, alpha_1d=2)
#        self.run_mpfa(mdg)
#        self.verify_cv(mdg)
#
#    def test_mpfa_two_fracs_coarsen_1d(self):
#        mdg = self.setup(num_fracs=2, alpha_1d=0.5)
#        self.run_mpfa(mdg)
#        self.verify_cv(mdg)
#
#    def test_mpfa_two_fracs_refine_mg(self):
#        mdg = self.setup(num_fracs=2, alpha_mortar=2)
#        self.run_mpfa(mdg)
#        self.verify_cv(mdg)
#
#    def test_mpfa_two_fracs_coarsen_mg(self):
#        mdg = self.setup(num_fracs=2, alpha_mortar=0.5)
#        self.run_mpfa(mdg)
#        self.verify_cv(mdg)


class TestMortar3D(unittest.TestCase):
    def setup(self, num_fracs=1, remove_tags=False):

        bbox = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
        domain = pp.Domain(bbox)

        if num_fracs == 0:
            fl = []

        elif num_fracs == 1:
            fl = [
                pp.PlaneFracture(
                    np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])
                )
            ]
        elif num_fracs == 2:
            fl = [
                pp.PlaneFracture(
                    np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])
                ),
                pp.PlaneFracture(
                    np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0], [0, 0, 1, 1]])
                ),
            ]

        elif num_fracs == 3:
            fl = [
                pp.PlaneFracture(
                    np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])
                ),
                pp.PlaneFracture(
                    np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0], [0, 0, 1, 1]])
                ),
                pp.PlaneFracture(
                    np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
                ),
            ]

        network = pp.FractureNetwork3d(fl, domain)
        mesh_args = {"mesh_size_frac": 0.5, "mesh_size_min": 0.5}
        mdg = network.mesh(mesh_args)

        self.set_params(mdg)

        return mdg

    def set_params(self, mdg):
        kw = "flow"
        for g, d in mdg.subdomains(return_data=True):
            parameter_dictionary = {}

            perm = pp.SecondOrderTensor(kxx=np.ones(g.num_cells))
            parameter_dictionary["second_order_tensor"] = perm

            yf = g.face_centers[1]
            bound_faces = [
                np.where(np.abs(yf - 1) < 1e-4)[0],
                np.where(np.abs(yf) < 1e-4)[0],
            ]
            bound_faces = np.hstack((bound_faces[0], bound_faces[1]))
            labels = np.array(["dir"] * bound_faces.size)
            parameter_dictionary["bc"] = pp.BoundaryCondition(g, bound_faces, labels)

            bv = np.zeros(g.num_faces)
            bound_faces = np.where(np.abs(yf - 1) < 1e-4)[0]
            bv[bound_faces] = 1
            parameter_dictionary["bc_values"] = bv
            parameter_dictionary["mpfa_inverter"] = "python"

            d[pp.PARAMETERS] = pp.Parameters(g, [kw], [parameter_dictionary])
            d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}
        kn = 1e7
        for intf, d in mdg.interfaces():

            flow_dictionary = {"normal_diffusivity": 2 * kn * np.ones(intf.num_cells)}
            d[pp.PARAMETERS] = pp.Parameters(
                keywords=["flow"], dictionaries=[flow_dictionary]
            )
            d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

    def verify_cv(self, mdg):
        for g, d in mdg.subdomains(return_data=True):
            p = d[pp.STATE]["pressure"]
            self.assertTrue(np.allclose(p, g.cell_centers[1], rtol=1e-3, atol=1e-3))

    def run_mpfa(self, mdg):
        key = "flow"
        method = pp.Mpfa(key)
        assembler = test_utils.setup_flow_assembler(mdg, method, key)
        assembler.discretize()
        A_flow, b_flow = assembler.assemble_matrix_rhs()
        p = sps.linalg.spsolve(A_flow, b_flow)
        assembler.distribute_variable(p)

    def run_vem(self, mdg):
        solver_flow = pp.MVEM("flow")

        A_flow, b_flow = solver_flow.matrix_rhs(mdg)

        up = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(mdg, "up", up)
        solver_flow.extract_p(mdg, "up", "pressure")
        self.verify_cv(mdg)

    def test_mpfa_no_fracs(self):
        mdg = self.setup(num_fracs=0)
        self.run_mpfa(mdg)
        self.verify_cv(mdg)


#    def test_mpfa_1_frac_no_refinement(self):
#
#
#        if False:
#            mdg = self.setup(num_fracs=1)
#            self.run_mpfa(mdg)
#        else:
#        # Choose and define the solvers and coupler
#            mdg = self.setup(num_fracs=3, remove_tags=True)
#            self.run_vem(mdg)
#
#        self.verify_cv(mdg)


# TODO: Add check that mortar flux scales with mortar area


class TestMortar2DSimplexGrid(unittest.TestCase):
    """Simplex grid with a hardcoded geometry. The domain is a unit cell, cut
    in two by a fracture at y=0.5. There are three cells in the lower half
    of the domain, with two of them having a face towards the fracture.
    The grid in the upper half is mirrored from the lower half, however,
    some of the tests pertub the node geometry.

    The flow regime imposed is uniform in the y-direction. The computed pressure
    should then be equal to the y-coordinates of the cells.
    """

    def grid_2d(self, pert_node=False, flip_normal=False):
        # pert_node pertubes one node in the grid. Leads to non-matching cells.
        # flip_normal flips one normal vector in 2d grid adjacent to the fracture.
        #   Tests that there is no assumptions on direction of fluxes in the
        #   mortar coupling.
        nodes = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 0.5, 0],
                [0.5, 0.5, 0],
                [0, 0.5, 0],
                [0, 0.5, 0],
                [0.5, 0.5, 0],
                [1, 0.5, 0],
                [1, 1, 0],
                [0, 1, 0],
            ]
        ).T
        if pert_node:
            nodes[0, 3] = 0.75

        fn = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 0],
                [0, 3],
                [3, 1],
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 9],
                [9, 5],
                [9, 6],
                [6, 8],
            ]
        ).T
        cf = np.array(
            [[3, 4, 5], [0, 6, 5], [1, 2, 6], [7, 12, 11], [12, 13, 10], [8, 9, 13]]
        ).T
        cols = np.tile(np.arange(fn.shape[1]), (fn.shape[0], 1)).ravel("F")
        face_nodes = sps.csc_matrix((np.ones_like(cols), (fn.ravel("F"), cols)))

        cols = np.tile(np.arange(cf.shape[1]), (cf.shape[0], 1)).ravel("F")
        data = np.array([1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1])

        cell_faces = sps.csc_matrix((data, (cf.ravel("F"), cols)))

        g = pp.Grid(2, nodes, face_nodes, cell_faces, "TriangleGrid")
        g.compute_geometry()
        g.tags["fracture_faces"][[2, 3, 7, 8]] = 1

        if flip_normal:
            g.face_normals[:, [2]] *= -1
            g.cell_faces[2, 2] *= -1
        g.global_point_ind = np.arange(nodes.shape[1])

        return g

    def grid_1d(self, num_pts=3):
        g = pp.TensorGrid(np.arange(num_pts))
        g.nodes = np.vstack(
            (np.linspace(0, 1, num_pts), 0.5 * np.ones(num_pts), np.zeros(num_pts))
        )
        g.compute_geometry()
        g.global_point_ind = np.arange(g.num_nodes)
        return g

    def setup(self, remove_tags=False, num_1d=3, pert_node=False, flip_normal=False):
        g2 = self.grid_2d()
        g1 = self.grid_1d()
        mdg, sd_map = pp.meshing._assemble_mdg([[g2], [g1]])

        a = np.zeros((g2.num_faces, g1.num_cells))
        a[2, 1] = 1
        a[3, 0] = 1
        a[7, 0] = 1
        a[8, 1] = 1
        face_cells = sps.csc_matrix(a.T)

        sd_map = {(g2, g1): face_cells}

        pp.meshing.create_interfaces(mdg, sd_map)

        g_new_2d = self.grid_2d(pert_node=pert_node, flip_normal=flip_normal)
        g_new_1d = self.grid_1d(num_1d)
        mdg.replace_subdomains_and_interfaces(sd_map={g2: g_new_2d, g1: g_new_1d})

        self.set_params(mdg)
        return mdg

    def set_params(self, mdg, no_flow=False):
        # Set up flow field with uniform flow in y-direction
        kw = "flow"
        kn = 1e6
        for g, d in mdg.subdomains(return_data=True):
            parameter_dictionary = {}

            perm = pp.SecondOrderTensor(kxx=np.ones(g.num_cells))
            parameter_dictionary["second_order_tensor"] = perm

            b_val = np.zeros(g.num_faces)
            if g.dim == 2:
                bound_faces = pp.face_on_side(g, ["ymin", "ymax"])
                if no_flow:
                    b_val[bound_faces[0]] = 1
                    b_val[bound_faces[1]] = 1
                bound_faces = np.hstack((bound_faces[0], bound_faces[1]))
                labels = np.array(["dir"] * bound_faces.size)
                parameter_dictionary["bc"] = pp.BoundaryCondition(
                    g, bound_faces, labels
                )

                y_max_faces = pp.face_on_side(g, "ymax")[0]
                b_val[y_max_faces] = 1
            else:
                parameter_dictionary["bc"] = pp.BoundaryCondition(g)
            parameter_dictionary["bc_values"] = b_val
            parameter_dictionary["mpfa_inverter"] = "python"

            d[pp.PARAMETERS] = pp.Parameters(g, [kw], [parameter_dictionary])
            d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

        for mg, d in mdg.interfaces(return_data=True):
            flow_dictionary = {"normal_diffusivity": 2 * kn * np.ones(mg.num_cells)}
            d[pp.PARAMETERS] = pp.Parameters(
                keywords=["flow"], dictionaries=[flow_dictionary]
            )
            d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

    def verify_cv(self, mdg, tol=1e-6):
        # The tolerance level here is a bit touchy: With an unstructured grid,
        # and with the flux between subdomains computed as differences between
        # point pressures, uniform flow may not be reproduced if the meshes
        # are not matching (one may get lucky, though). Thus the coarse error
        # tolerance. The current value turned out to be sufficient for all
        # tests considered herein.
        for g, d in mdg.subdomains(return_data=True):
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
        for g, d in mdg.subdomains(return_data=True):
            d[pp.STATE]["darcy_flux"] = d[pp.STATE]["pressure"][: g.num_faces]
            d[pp.STATE]["pressure"] = d[pp.STATE]["pressure"][g.num_faces :]

    def run_RT0(self, mdg):
        key = "flow"
        method = pp.RT0(key)
        self._solve(mdg, method, key)
        for g, d in mdg.subdomains(return_data=True):
            d[pp.STATE]["darcy_flux"] = d[pp.STATE]["pressure"][: g.num_faces]
            d[pp.STATE]["pressure"] = d[pp.STATE]["pressure"][g.num_faces :]

    def test_mpfa(self):
        mdg = self.setup(False)
        self.run_mpfa(mdg)
        self.verify_cv(mdg)

    def test_mpfa_pert_2d_node(self):
        # Perturb one node facing the fracture in the lower half. This breaks
        # symmetry of the grid, and lead to non-mathching grids.
        mdg = self.setup(False, pert_node=True)
        self.run_mpfa(mdg)
        self.verify_cv(mdg)

    def test_mpfa_flip_normal(self):
        # Perturb one node facing the fracture in the lower half. This breaks
        # symmetry of the grid, and lead to non-mathching grids.
        mdg = self.setup(False, flip_normal=True)
        self.run_mpfa(mdg)
        self.verify_cv(mdg)

    def test_mpfa_refined_1d(self):
        # Refine grid in the fracture.
        mdg = self.setup(False, num_1d=4)
        self.run_mpfa(mdg)
        self.verify_cv(mdg)

    def test_vem(self):
        mdg = self.setup(False)
        self.run_vem(mdg)
        self.verify_cv(mdg)

    def test_vem_pert_2d_node(self):
        # Perturb one node facing the fracture in the lower half. This breaks
        # symmetry of the grid, and lead to non-mathching grids.
        mdg = self.setup(False, pert_node=True)
        self.run_vem(mdg)
        self.verify_cv(mdg)

    def test_vem_flip_normal(self):
        # Perturb one node facing the fracture in the lower half. This breaks
        # symmetry of the grid, and lead to non-mathching grids.
        mdg = self.setup(False, flip_normal=True)
        self.run_vem(mdg)
        self.verify_cv(mdg)

    def test_vem_refined_1d(self):
        # Refine grid in the fracture.
        mdg = self.setup(False, num_1d=4)
        self.run_vem(mdg)
        self.verify_cv(mdg)

    def test_RT0(self):
        mdg = self.setup(False)
        self.run_RT0(mdg)
        self.verify_cv(mdg)

    def test_RT0_pert_2d_node(self):
        # Perturb one node facing the fracture in the lower half. This breaks
        # symmetry of the grid, and lead to non-mathching grids.
        mdg = self.setup(False, pert_node=True)
        self.run_RT0(mdg)
        self.verify_cv(mdg)

    def test_RT0_flip_normal(self):
        # Perturb one node facing the fracture in the lower half. This breaks
        # symmetry of the grid, and lead to non-mathching grids.
        mdg = self.setup(False, flip_normal=True)
        self.run_RT0(mdg)
        self.verify_cv(mdg)

    def test_RT0_refined_1d(self):
        # Refine grid in the fracture.
        mdg = self.setup(False, num_1d=4)
        self.run_RT0(mdg)
        self.verify_cv(mdg)


if __name__ == "__main__":
    unittest.main()
