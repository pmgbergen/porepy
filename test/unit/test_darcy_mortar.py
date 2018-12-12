#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:25:01 2017

@author: Eirik Keilegavlens
"""
import numpy as np
import scipy.sparse as sps
import unittest

import porepy as pp
from test import test_utils


class TestMortar2dSingleFractureCartesianGrid(unittest.TestCase):
    def set_param_flow(self, gb, no_flow=False, kn=1e3, multi_point=True):
        # Set up flow field with uniform flow in y-direction
        kw = "flow"
        for g, d in gb:
            parameter_dictionary = {}

            perm = pp.SecondOrderTensor(g.dim, kxx=np.ones(g.num_cells))
            parameter_dictionary["second_order_tensor"] = perm

            aperture = np.power(1e-3, gb.dim_max() - g.dim)
            parameter_dictionary["aperture"] = aperture * np.ones(g.num_cells)

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

        gb.add_edge_props("kn")
        for e, d in gb.edges():
            mg = d["mortar_grid"]
            flow_dictionary = {"normal_diffusivity": kn * np.ones(mg.num_cells)}
            d[pp.PARAMETERS] = pp.Parameters(
                keywords=["flow"], dictionaries=[flow_dictionary]
            )
            d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

        discretization_key = kw + "_" + pp.DISCRETIZATION

        for g, d in gb:
            # Choose discretization and define the solver
            if multi_point:
                discr = pp.Mpfa(kw)
            else:
                discr = pp.Tpfa(kw)

            d[discretization_key] = discr

        for _, d in gb.edges():
            d[discretization_key] = pp.RobinCoupling(kw, discr)

    def set_grids(self, N, num_nodes_mortar, num_nodes_1d, physdims=[1, 1]):
        f1 = np.array([[0, physdims[0]], [0.5, 0.5]])

        gb = pp.meshing.cart_grid([f1], N, **{"physdims": physdims})
        gb.compute_geometry()
        gb.assign_node_ordering()

        for e, d in gb.edges():
            mg = d["mortar_grid"]
            new_side_grids = {
                s: pp.refinement.remesh_1d(g, num_nodes=num_nodes_mortar)
                for s, g in mg.side_grids.items()
            }

            pp.mortars.update_mortar_grid(mg, new_side_grids, tol=1e-4)

            # refine the 1d-physical grid
            old_g = gb.nodes_of_edge(e)[0]
            new_g = pp.refinement.remesh_1d(old_g, num_nodes=num_nodes_1d)
            new_g.compute_geometry()

            gb.update_nodes({old_g: new_g})
            mg = d["mortar_grid"]
            pp.mortars.update_physical_low_grid(mg, new_g, tol=1e-4)
        return gb

    def solve(self, gb, method=None):
        key = "flow"
        if method is None:
            method = pp.Tpfa(key)
        elif method == "mpfa":
            method = pp.Mpfa(key)
        assembler = test_utils.setup_flow_assembler(gb, method, key)
        A_flow, b_flow, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
        p = sps.linalg.spsolve(A_flow, b_flow)
        assembler.distribute_variable(gb, p, block_dof, full_dof)

        return p

    def test_tpfa_matching_grids_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=True, multi_point=False)

        p = self.solve(gb)

        self.assertTrue(np.all(p[:3] == 1))
        self.assertTrue(np.all(p[3:] == 0))

    def test_tpfa_matching_grids_refine_1d_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(gb, no_flow=True, multi_point=False)

        p = self.solve(gb)
        self.assertTrue(np.all(p[:4] == 1))
        self.assertTrue(np.all(p[4:] == 0))

    def test_tpfa_matching_grids_refine_mortar_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=True, multi_point=False)

        p = self.solve(gb)
        self.assertTrue(np.all(p[:3] == 1))
        self.assertTrue(np.all(p[3:] == 0))

    def test_tpfa_matching_grids_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, multi_point=False, kn=kn)

        self.solve(gb)

        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        self.assertTrue(np.allclose(p_2d, g_2d.cell_centers[1], rtol=kn))

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        self.assertTrue(np.allclose(p_1d, g_1d.cell_centers[1], rtol=kn))

    def test_tpfa_matching_grids_refine_1d_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(gb, no_flow=False, multi_point=False, kn=kn)

        self.solve(gb)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        self.assertTrue(np.allclose(p_2d, g_2d.cell_centers[1], rtol=1.0 / kn))

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        self.assertTrue(np.allclose(p_1d, g_1d.cell_centers[1]))

    def test_tpfa_matching_grids_refine_mortar_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, multi_point=False, kn=kn)

        self.solve(gb)

        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        self.assertTrue(np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4))

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        self.assertTrue(np.allclose(p_1d, g_1d.cell_centers[1]))

    def test_tpfa_matching_grids_refine_2d_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[2, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, multi_point=False, kn=kn)

        self.solve(gb)

        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        self.assertTrue(np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4))

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        self.assertTrue(np.allclose(p_1d, g_1d.cell_centers[1]))

    def test_tpfa_matching_grids_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(
            N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(gb, no_flow=False, multi_point=False, kn=kn)

        self.solve(gb)

        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        self.assertTrue(np.allclose(p_2d, g_2d.cell_centers[1], rtol=kn))

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        self.assertTrue(np.allclose(p_1d, g_1d.cell_centers[1], rtol=kn))

    def test_tpfa_matching_grids_refine_1d_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(
            N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3, physdims=[2, 1]
        )
        self.set_param_flow(gb, no_flow=False, multi_point=False, kn=kn)

        self.solve(gb)

        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        self.assertTrue(np.allclose(p_2d, g_2d.cell_centers[1], rtol=1.0 / kn))

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        self.assertTrue(np.allclose(p_1d, g_1d.cell_centers[1]))

    def test_tpfa_matching_grids_refine_mortar_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(
            N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(gb, no_flow=False, multi_point=False, kn=kn)

        self.solve(gb)

        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        self.assertTrue(np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4))

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        self.assertTrue(np.allclose(p_1d, g_1d.cell_centers[1]))

    def test_tpfa_matching_grids_refine_2d_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(
            N=[2, 2], num_nodes_mortar=2, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(gb, no_flow=False, multi_point=False, kn=kn)

        self.solve(gb)

        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        self.assertTrue(np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4))

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        self.assertTrue(np.allclose(p_1d, g_1d.cell_centers[1]))

    def test_mpfa_matching_grids_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=True, multi_point=True)

        p = self.solve(gb, "mpfa")

        self.assertTrue(np.all(p[:3] == 1))
        self.assertTrue(np.all(p[3:] == 0))

    def test_mpfa_matching_grids_refine_1d_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(gb, no_flow=True, multi_point=True)

        p = self.solve(gb, "mpfa")

        self.assertTrue(np.all(p[:4] == 1))
        self.assertTrue(np.all(p[4:] == 0))

    def test_mpfa_matching_grids_refine_mortar_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=True, multi_point=True)

        p = self.solve(gb, "mpfa")

        self.assertTrue(np.all(p[:3] == 1))
        self.assertTrue(np.all(p[3:] == 0))

    def test_mpfa_matching_grids_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, multi_point=True, kn=kn)

        self.solve(gb, "mpfa")

        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        self.assertTrue(np.allclose(p_2d, g_2d.cell_centers[1], rtol=kn))

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        self.assertTrue(np.allclose(p_1d, g_1d.cell_centers[1], rtol=kn))

    def test_mpfa_matching_grids_refine_1d_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(gb, no_flow=False, multi_point=True, kn=kn)

        self.solve(gb, "mpfa")

        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        self.assertTrue(np.allclose(p_2d, g_2d.cell_centers[1], rtol=1.0 / kn))

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        self.assertTrue(np.allclose(p_1d, g_1d.cell_centers[1]))

    def test_mpfa_matching_grids_refine_mortar_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, multi_point=True, kn=kn)

        self.solve(gb, "mpfa")

        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        self.assertTrue(np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4))

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        self.assertTrue(np.allclose(p_1d, g_1d.cell_centers[1]))

    def test_mpfa_matching_grids_refine_2d_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[2, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, multi_point=True, kn=kn)

        self.solve(gb, "mpfa")

        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        self.assertTrue(np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4))

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        self.assertTrue(np.allclose(p_1d, g_1d.cell_centers[1]))

    def test_mpfa_matching_grids_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(
            N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(gb, no_flow=False, multi_point=True, kn=kn)

        self.solve(gb, "mpfa")

        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        self.assertTrue(np.allclose(p_2d, g_2d.cell_centers[1], rtol=kn))

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        self.assertTrue(np.allclose(p_1d, g_1d.cell_centers[1], rtol=kn))

    def test_mpfa_matching_grids_refine_1d_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(
            N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3, physdims=[2, 1]
        )
        self.set_param_flow(gb, no_flow=False, multi_point=True, kn=kn)

        self.solve(gb, "mpfa")

        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        self.assertTrue(np.allclose(p_2d, g_2d.cell_centers[1], rtol=1.0 / kn))

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        self.assertTrue(np.allclose(p_1d, g_1d.cell_centers[1]))

    def test_mpfa_matching_grids_refine_mortar_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(
            N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(gb, no_flow=False, multi_point=True, kn=kn)

        self.solve(gb, "mpfa")

        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        self.assertTrue(np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4))

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        self.assertTrue(np.allclose(p_1d, g_1d.cell_centers[1]))

    def test_mpfa_matching_grids_refine_2d_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(
            N=[2, 2], num_nodes_mortar=2, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(gb, no_flow=False, multi_point=True, kn=kn)

        self.solve(gb, "mpfa")

        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        self.assertTrue(np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4))

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        self.assertTrue(np.allclose(p_1d, g_1d.cell_centers[1]))


class TestMortar2DSimplexGridStandardMeshing(unittest.TestCase):
    def setup(
        self,
        num_fracs=1,
        remove_tags=False,
        alpha_1d=None,
        alpha_mortar=None,
        alpha_2d=None,
    ):

        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

        if num_fracs == 0:
            p = np.zeros((2, 0))
            e = np.zeros((2, 0))

        elif num_fracs == 1:
            p = np.array([[0, 1], [0.5, 0.5]])
            e = np.array([[0], [1]])
        #            p = [np.array([[0.5, 0.5], [0, 1]])]
        elif num_fracs == 2:
            raise ValueError("Not implemented")
        mesh_size = {"value": 0.3, "bound_value": 0.3}
        network = pp.FractureNetwork2d(p, e, domain)
        gb = network.mesh(mesh_size)
        #        gb = meshing.cart_grid([np.array([[0.5, 0.5], [0, 1]])],np.array([10, 10]),
        #                               physdims=np.array([1, 1]))

        gmap = {}

        # Refine 2D grid?
        if alpha_2d is not None:
            mesh_size = {"value": 0.3 * alpha_2d, "bound_value": 0.3 * alpha_2d}
            gbn = network.mesh(mesh_size)
            go = gb.grids_of_dimension(2)[0]
            gn = gbn.grids_of_dimension(2)[0]
            gn.compute_geometry()
            gmap[go] = gn

        # Refine 1d grids
        if alpha_1d is not None:
            for g, d in gb:
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
            for e, d in gb.edges():
                mg = d["mortar_grid"]
                if mg.dim == 1:
                    mg_map[mg] = {}
                    for s, g in mg.side_grids.items():
                        num_nodes = int(g.num_nodes * alpha_mortar)
                        mg_map[mg][s] = pp.refinement.remesh_1d(g, num_nodes=num_nodes)

        gb = pp.mortars.replace_grids_in_bucket(gb, gmap, mg_map, tol=1e-4)

        gb.assign_node_ordering()

        self.set_params(gb)

        return gb

    def set_params(self, gb):
        kw = "flow"
        for g, d in gb:
            parameter_dictionary = {}

            perm = pp.SecondOrderTensor(g.dim, kxx=np.ones(g.num_cells))
            parameter_dictionary["second_order_tensor"] = perm

            aperture = np.power(1e-3, gb.dim_max() - g.dim)
            parameter_dictionary["aperture"] = aperture * np.ones(g.num_cells)
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

        gb.add_edge_props("kn")
        kn = 1e7
        for e, d in gb.edges():
            mg = d["mortar_grid"]

            flow_dictionary = {"normal_diffusivity": kn * np.ones(mg.num_cells)}
            d[pp.PARAMETERS] = pp.Parameters(
                keywords=["flow"], dictionaries=[flow_dictionary]
            )
            d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

    def verify_cv(self, gb, tol=1e-2):
        # The tolerance level here is a bit touchy: With an unstructured grid,
        # and with the flux between subdomains computed as differences between
        # point pressures, uniform flow may not be reproduced if the meshes
        # are not matching (one may get lucky, though). Thus the coarse error
        # tolerance. The current value turned out to be sufficient for all
        # tests considered herein.
        for g, _ in gb.nodes():
            p = gb.node_props(g, "pressure")
            self.assertTrue(np.allclose(p, g.cell_centers[1], rtol=tol, atol=tol))

    def run_mpfa(self, gb):
        key = "flow"
        method = pp.Mpfa(key)
        assembler = test_utils.setup_flow_assembler(gb, method, key)
        A_flow, b_flow, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
        p = sps.linalg.spsolve(A_flow, b_flow)
        assembler.distribute_variable(gb, p, block_dof, full_dof)

    def run_vem(self, gb):
        key = "flow"
        method = pp.MVEM(key)
        assembler = test_utils.setup_flow_assembler(gb, method, key)
        A_flow, b_flow, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
        p = sps.linalg.spsolve(A_flow, b_flow)
        assembler.distribute_variable(gb, p, block_dof, full_dof)
        for g, d in gb:
            d["pressure"] = d["pressure"][g.num_faces :]

    def test_mpfa_one_frac(self):
        gb = self.setup(num_fracs=1)
        self.run_mpfa(gb)
        self.verify_cv(gb)

    def test_mpfa_one_frac_refine_2d(self):
        gb = self.setup(num_fracs=1, alpha_2d=2)
        self.run_mpfa(gb)
        self.verify_cv(gb)

    def test_mpfa_one_frac_coarsen_2d(self):
        gb = self.setup(num_fracs=1, alpha_2d=0.5)
        self.run_mpfa(gb)
        self.verify_cv(gb)

    def test_mpfa_one_frac_refine_1d(self):
        gb = self.setup(num_fracs=1, alpha_1d=2)
        self.run_mpfa(gb)
        self.verify_cv(gb)

    def test_mpfa_one_frac_coarsen_1d(self):
        gb = self.setup(num_fracs=1, alpha_1d=0.5)
        self.run_mpfa(gb)
        self.verify_cv(gb)

    def test_mpfa_one_frac_refine_mg(self):
        gb = self.setup(num_fracs=1, alpha_mortar=2)
        self.run_mpfa(gb)
        self.verify_cv(gb)

    def test_mpfa_one_frac_coarsen_mg(self):
        gb = self.setup(num_fracs=1, alpha_mortar=0.5)
        self.run_mpfa(gb)
        self.verify_cv(gb)

    def test_vem_one_frac(self):
        gb = self.setup(num_fracs=1, remove_tags=True)
        self.run_vem(gb)
        self.verify_cv(gb, tol=1e-7)

    def test_vem_one_frac_refine_2d(self):
        gb = self.setup(num_fracs=1, alpha_2d=2, remove_tags=True)
        self.run_vem(gb)
        self.verify_cv(gb)

    def test_vem_one_frac_coarsen_2d(self):
        gb = self.setup(num_fracs=1, alpha_2d=0.5, remove_tags=True)
        self.run_vem(gb)
        self.verify_cv(gb)

    def test_vem_one_frac_refine_1d(self):
        gb = self.setup(num_fracs=1, alpha_1d=2, remove_tags=True)
        self.run_vem(gb)
        self.verify_cv(gb)

    def test_vem_one_frac_coarsen_1d(self):
        gb = self.setup(num_fracs=1, alpha_1d=0.5, remove_tags=True)
        self.run_vem(gb)
        self.verify_cv(gb)

    def test_vem_one_frac_refine_mg(self):
        gb = self.setup(num_fracs=1, alpha_mortar=2, remove_tags=True)
        self.run_vem(gb)
        self.verify_cv(gb)

    def test_vem_one_frac_coarsen_mg(self):
        gb = self.setup(num_fracs=1, alpha_mortar=0.5, remove_tags=True)
        self.run_vem(gb)
        self.verify_cv(gb)


#    def test_mpfa_two_fracs(self):
#        gb = self.setup(num_fracs=2)
#        self.run_mpfa(gb)
#        self.verify_cv(gb)
#
#    def test_mpfa_two_fracs_refine_2d(self):
#        gb = self.setup(num_fracs=2, alpha_2d=2)
#        self.run_mpfa(gb)
#        self.verify_cv(gb)
#
#    def test_mpfa_two_fracs_coarsen_2d(self):
#        gb = self.setup(num_fracs=2, alpha_2d=0.5)
#        self.run_mpfa(gb)
#        self.verify_cv(gb)
#
#    def test_mpfa_two_fracs_refine_1d(self):
#        gb = self.setup(num_fracs=2, alpha_1d=2)
#        self.run_mpfa(gb)
#        self.verify_cv(gb)
#
#    def test_mpfa_two_fracs_coarsen_1d(self):
#        gb = self.setup(num_fracs=2, alpha_1d=0.5)
#        self.run_mpfa(gb)
#        self.verify_cv(gb)
#
#    def test_mpfa_two_fracs_refine_mg(self):
#        gb = self.setup(num_fracs=2, alpha_mortar=2)
#        self.run_mpfa(gb)
#        self.verify_cv(gb)
#
#    def test_mpfa_two_fracs_coarsen_mg(self):
#        gb = self.setup(num_fracs=2, alpha_mortar=0.5)
#        self.run_mpfa(gb)
#        self.verify_cv(gb)


class TestMortar3D(unittest.TestCase):
    def setup(self, num_fracs=1, remove_tags=False):

        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}

        if num_fracs == 0:
            fl = []

        elif num_fracs == 1:
            fl = [
                pp.Fracture(
                    np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])
                )
            ]
        elif num_fracs == 2:
            fl = [
                pp.Fracture(
                    np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])
                ),
                pp.Fracture(
                    np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0], [0, 0, 1, 1]])
                ),
            ]

        elif num_fracs == 3:
            fl = [
                pp.Fracture(
                    np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])
                ),
                pp.Fracture(
                    np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0], [0, 0, 1, 1]])
                ),
                pp.Fracture(
                    np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
                ),
            ]

        network = pp.FractureNetwork3d(fl, domain)
        mesh_args = {"mesh_size_frac": 0.5, "mesh_size_min": 0.5}
        gb = network.mesh(mesh_args)

        self.set_params(gb)

        return gb

    def set_params(self, gb):
        kw = "flow"
        for g, d in gb:
            parameter_dictionary = {}

            perm = pp.SecondOrderTensor(g.dim, kxx=np.ones(g.num_cells))
            parameter_dictionary["second_order_tensor"] = perm

            aperture = np.power(1e-6, gb.dim_max() - g.dim)
            parameter_dictionary["aperture"] = aperture * np.ones(g.num_cells)

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
        for e, d in gb.edges():
            mg = d["mortar_grid"]

            flow_dictionary = {"normal_diffusivity": kn * np.ones(mg.num_cells)}
            d[pp.PARAMETERS] = pp.Parameters(
                keywords=["flow"], dictionaries=[flow_dictionary]
            )
            d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

    def verify_cv(self, gb):
        for g, _ in gb.nodes():
            p = gb.node_props(g, "pressure")
            self.assertTrue(np.allclose(p, g.cell_centers[1], rtol=1e-3, atol=1e-3))

    def run_mpfa(self, gb):
        key = "flow"
        method = pp.Mpfa(key)
        assembler = test_utils.setup_flow_assembler(gb, method, key)
        A_flow, b_flow, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
        p = sps.linalg.spsolve(A_flow, b_flow)
        assembler.distribute_variable(gb, p, block_dof, full_dof)

    def run_vem(self, gb):
        solver_flow = pp.MVEM("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        up = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "up", up)
        solver_flow.extract_p(gb, "up", "pressure")

    def test_mpfa_no_fracs(self):
        gb = self.setup(num_fracs=0)
        self.run_mpfa(gb)
        self.verify_cv(gb)


#    def test_mpfa_1_frac_no_refinement(self):
#
#
#        if False:
#            gb = self.setup(num_fracs=1)
#            self.run_mpfa(gb)
#        else:
#        # Choose and define the solvers and coupler
#            gb = self.setup(num_fracs=3, remove_tags=True)
#            self.run_vem(gb)
#
#        self.verify_cv(gb)


# TODO: Add check that mortar flux scales with mortar area


class TestMortar2DSimplexGrid(unittest.TestCase):
    def grid_2d(self, pert_node=False):
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
        # g.face_normals[1, [2, 3]] = -0.5
        # g.face_normals[1, [7, 8]] = 0.5
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

    def setup(self, remove_tags=False, num_1d=3, pert_node=False):
        g2 = self.grid_2d()
        g1 = self.grid_1d()
        gb = pp.meshing._assemble_in_bucket([[g2], [g1]])

        gb.add_edge_props("face_cells")
        for e, d in gb.edges():
            a = np.zeros((g2.num_faces, g1.num_cells))
            a[2, 1] = 1
            a[3, 0] = 1
            a[7, 0] = 1
            a[8, 1] = 1
            d["face_cells"] = sps.csc_matrix(a.T)
        pp.meshing.create_mortar_grids(gb)

        g_new_2d = self.grid_2d(pert_node)
        g_new_1d = self.grid_1d(num_1d)
        pp.mortars.replace_grids_in_bucket(gb, g_map={g2: g_new_2d, g1: g_new_1d})

        gb.assign_node_ordering()

        self.set_params(gb)
        return gb

    def set_params(self, gb):
        kw = "flow"
        for g, d in gb:
            parameter_dictionary = {}

            perm = pp.SecondOrderTensor(g.dim, kxx=np.ones(g.num_cells))
            parameter_dictionary["bc_values"] = perm

            aperture = np.power(1e-3, gb.dim_max() - g.dim)
            parameter_dictionary["aperture"] = aperture * np.ones(g.num_cells)

            bv = np.zeros(g.num_faces)
            if g.dim == 2:
                bound_faces = np.array([0, 10])
                labels = np.array(["dir"] * bound_faces.size)
                parameter_dictionary["bc"] = pp.BoundaryCondition(
                    g, bound_faces, labels
                )
                bound_faces = 10
                bv[bound_faces] = 1
            else:
                parameter_dictionary["bc"] = pp.BoundaryCondition(g)
            parameter_dictionary["bc_values"] = bv

            d[pp.PARAMETERS] = pp.Parameters(g, [kw], [parameter_dictionary])
            d[pp.DISCRETIZATION_MATRICES] = {}
        gb.add_edge_props("kn")
        kn = 1e7
        for e, d in gb.edges():
            mg = d["mortar_grid"]

            flow_dictionary = {"normal_diffusivity": kn * np.ones(mg.num_cells)}
            d[pp.PARAMETERS] = pp.Parameters(
                keywords=["flow"], dictionaries=[flow_dictionary]
            )
            d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

    def verify_cv(self, gb, tol=1e-5):
        # The tolerance level here is a bit touchy: With an unstructured grid,
        # and with the flux between subdomains computed as differences between
        # point pressures, uniform flow may not be reproduced if the meshes
        # are not matching (one may get lucky, though). Thus the coarse error
        # tolerance. The current value turned out to be sufficient for all
        # tests considered herein.
        for g, _ in gb.nodes():
            p = gb.node_props(g, "pressure")
            #            print(p)
            #            print(g.cell_centers[1])
            self.assertTrue(np.allclose(p, g.cell_centers[1], rtol=tol, atol=tol))

    def run_mpfa(self, gb):
        key = "flow"
        method = pp.Mpfa(key)
        assembler = test_utils.setup_flow_assembler(gb, method, key)
        A_flow, b_flow, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
        p = sps.linalg.spsolve(A_flow, b_flow)
        assembler.distribute_variable(gb, p, "pressure", block_dof, full_dof)
        assembler.distribute_variable(gb, p, "mortar_solution", block_dof, full_dof)

    def run_vem(self, gb):
        solver_flow = pp.MVEM("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        up = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "up", up)
        solver_flow.extract_p(gb, "up", "pressure")


#    def test_mpfa_one_frac(self):
#        gb = self.setup(False)
#        self.run_mpfa(gb)
#        self.verify_cv(gb)
#
#    def test_mpfa_one_frac_pert_2d_node(self):
#        gb = self.setup(False, pert_node=True)
#        self.run_mpfa(gb)
#        self.verify_cv(gb)
#
#    def test_mpfa_one_frac_refined_1d(self):
#        gb = self.setup(False, num_1d=4)
#        self.run_mpfa(gb)
#        self.verify_cv(gb)
#
#    def test_mpfa_one_frac_refined_1d_pert_2d_node(self):
#        gb = self.setup(False, num_1d=4, pert_node=True)
#        self.run_mpfa(gb)
#        self.verify_cv(gb)
#
#    def test_mpfa_one_frac_coarsened_1d(self):
#        gb = self.setup(False, num_1d=2)
#        self.run_mpfa(gb)
#        self.verify_cv(gb)
#
#    def test_mpfa_one_frac_coarsened_1d_pert_2d_node(self):
#        gb = self.setup(False, num_1d=2, pert_node=True)
#        self.run_mpfa(gb)
#        self.verify_cv(gb)

if __name__ == "__main__":
    unittest.main()
