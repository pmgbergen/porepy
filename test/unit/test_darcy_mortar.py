#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:25:01 2017

@author: Eirik Keilegavlens
"""
import numpy as np
import scipy.sparse as sps
import unittest

from porepy.grids.structured import TensorGrid
from porepy.grids.simplex import TriangleGrid
from porepy.grids import refinement, mortar_grid
from porepy.fracs import meshing, mortars
from porepy.fracs.fractures import Fracture

from porepy.params.data import Parameters
from porepy.params import bc
from porepy.params.bc import BoundaryCondition
from porepy.grids.grid import Grid
from porepy.params import tensor


from porepy.numerics.vem import vem_dual, vem_source
from porepy.numerics.fv import tpfa, mpfa


class TestMortar2dSingleFractureCartesianGrid(unittest.TestCase):
    def set_param_flow(self, gb, no_flow=False, kn=1e3):
        # Set up flow field with uniform flow in y-direction
        gb.add_node_props(["param"])
        for g, d in gb:
            param = Parameters(g)

            perm = tensor.SecondOrderTensor(g.dim, kxx=np.ones(g.num_cells))
            param.set_tensor("flow", perm)

            aperture = np.power(1e-3, gb.dim_max() - g.dim)
            param.set_aperture(aperture * np.ones(g.num_cells))

            if g.dim == 2:
                b_val = np.zeros(g.num_faces)
                bound_faces = bc.face_on_side(g, ["ymin", "ymax"])
                if no_flow:
                    b_val[bound_faces[0]] = 1
                    b_val[bound_faces[1]] = 1
                bound_faces = np.hstack((bound_faces[0], bound_faces[1]))
                labels = np.array(["dir"] * bound_faces.size)
                param.set_bc("flow", bc.BoundaryCondition(g, bound_faces, labels))

                bound_faces = bc.face_on_side(g, "ymax")[0]
                b_val[bound_faces] = 1
                param.set_bc_val("flow", b_val)

            d["param"] = param

        gb.add_edge_props("kn")
        for e, d in gb.edges():
            mg = d["mortar_grid"]
            gn = gb.nodes_of_edge(e)
            d["kn"] = kn * np.ones(mg.num_cells)

    def set_grids(self, N, num_nodes_mortar, num_nodes_1d, physdims=[1, 1]):
        f1 = np.array([[0, physdims[0]], [.5, .5]])

        gb = meshing.cart_grid([f1], N, **{"physdims": physdims})
        gb.compute_geometry()
        gb.assign_node_ordering()

        for e, d in gb.edges():
            mg = d["mortar_grid"]
            new_side_grids = {
                s: refinement.remesh_1d(g, num_nodes=num_nodes_mortar)
                for s, g in mg.side_grids.items()
            }

            mortars.update_mortar_grid(mg, new_side_grids, tol=1e-4)

            # refine the 1d-physical grid
            old_g = gb.nodes_of_edge(e)[0]
            new_g = refinement.remesh_1d(old_g, num_nodes=num_nodes_1d)
            new_g.compute_geometry()

            gb.update_nodes(old_g, new_g)
            mg = d["mortar_grid"]
            mortars.update_physical_low_grid(mg, new_g, tol=1e-4)
        return gb

    def test_tpfa_matching_grids_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=True)

        solver_flow = tpfa.TpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        assert np.all(p[:3] == 1)
        assert np.all(p[3:] == 0)

    def test_tpfa_matching_grids_refine_1d_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(gb, no_flow=True)

        solver_flow = tpfa.TpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        assert np.all(p[:4] == 1)
        assert np.all(p[4:] == 0)

    def test_tpfa_matching_grids_refine_mortar_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=True)

        solver_flow = tpfa.TpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)
        for e, d in gb.edges():
            mg = d["mortar_grid"]

        p = sps.linalg.spsolve(A_flow, b_flow)
        assert np.all(p[:3] == 1)
        assert np.all(p[3:] == 0)

    def test_tpfa_matching_grids_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = tpfa.TpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=kn)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1], rtol=kn)

    def test_tpfa_matching_grids_refine_1d_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = tpfa.TpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1. / kn)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_tpfa_matching_grids_refine_mortar_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = tpfa.TpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_tpfa_matching_grids_refine_2d_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[2, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = tpfa.TpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_tpfa_matching_grids_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(
            N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = tpfa.TpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=kn)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1], rtol=kn)

    def test_tpfa_matching_grids_refine_1d_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(
            N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3, physdims=[2, 1]
        )
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = tpfa.TpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1. / kn)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_tpfa_matching_grids_refine_mortar_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(
            N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = tpfa.TpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_tpfa_matching_grids_refine_2d_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(
            N=[2, 2], num_nodes_mortar=2, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = tpfa.TpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_mpfa_matching_grids_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=True)

        solver_flow = mpfa.MpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        assert np.all(p[:3] == 1)
        assert np.all(p[3:] == 0)

    def test_mpfa_matching_grids_refine_1d_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(gb, no_flow=True)

        solver_flow = mpfa.MpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        assert np.all(p[:4] == 1)
        assert np.all(p[4:] == 0)

    def test_mpfa_matching_grids_refine_mortar_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=True)

        solver_flow = mpfa.MpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)
        for e, d in gb.edges():
            mg = d["mortar_grid"]

        p = sps.linalg.spsolve(A_flow, b_flow)
        assert np.all(p[:3] == 1)
        assert np.all(p[3:] == 0)

    def test_mpfa_matching_grids_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = mpfa.MpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=kn)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1], rtol=kn)

    def test_mpfa_matching_grids_refine_1d_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = mpfa.MpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1. / kn)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_mpfa_matching_grids_refine_mortar_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = mpfa.MpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_mpfa_matching_grids_refine_2d_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[2, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = mpfa.MpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_mpfa_matching_grids_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(
            N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = mpfa.MpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=kn)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1], rtol=kn)

    def test_mpfa_matching_grids_refine_1d_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(
            N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3, physdims=[2, 1]
        )
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = mpfa.MpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1. / kn)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_mpfa_matching_grids_refine_mortar_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(
            N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = mpfa.MpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_mpfa_matching_grids_refine_2d_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(
            N=[2, 2], num_nodes_mortar=2, num_nodes_1d=2, physdims=[2, 1]
        )
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = mpfa.MpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_props(g_2d, "pressure")
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_props(g_1d, "pressure")
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])


# -----------------------------------------------------------------------------#


# -----------------------------------------------------------------------------#


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

        elif num_fracs == 1:
            p = [np.array([[0, 1], [0.5, 0.5]])]
        #            p = [np.array([[0.5, 0.5], [0, 1]])]
        elif num_fracs == 2:
            p = [
                np.array([[0, 0.5], [0.5, 0.5]]),
                np.array([[0.5, 1], [0.5, 0.5]]),
                np.array([[0.5, 0.5], [0, 0.5]]),
                np.array([[0.5, 0.5], [0.5, 1]]),
            ]
        mesh_size = {"value": 0.3, "bound_value": 0.3}
        gb = meshing.simplex_grid(
            fracs=p, domain=domain, mesh_size=mesh_size, verbose=0
        )
        #        gb = meshing.cart_grid([np.array([[0.5, 0.5], [0, 1]])],np.array([10, 10]),
        #                               physdims=np.array([1, 1]))

        gmap = {}

        # Refine 2D grid?
        if alpha_2d is not None:
            mesh_size = {"value": 0.3 * alpha_2d, "bound_value": 0.3 * alpha_2d}
            gbn = meshing.simplex_grid(
                fracs=p, domain=domain, mesh_size=mesh_size, verbose=0
            )
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
                    gmap[g] = refinement.remesh_1d(g, num_nodes=num_nodes)
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
                        mg_map[mg][s] = refinement.remesh_1d(g, num_nodes=num_nodes)

        gb = mortars.replace_grids_in_bucket(gb, gmap, mg_map, tol=1e-4)

        #        if remove_tags:
        #            internal_flag = FaceTag.FRACTURE
        #            [g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]

        gb.assign_node_ordering()

        self.set_params(gb)

        return gb

    def set_params(self, gb):

        for g, d in gb:
            param = Parameters(g)

            perm = tensor.SecondOrderTensor(g.dim, kxx=np.ones(g.num_cells))
            param.set_tensor("flow", perm)

            aperture = np.power(1e-3, gb.dim_max() - g.dim)
            param.set_aperture(aperture * np.ones(g.num_cells))

            yf = g.face_centers[1]
            bound_faces = [
                np.where(np.abs(yf - 1) < 1e-4)[0],
                np.where(np.abs(yf) < 1e-4)[0],
            ]
            bound_faces = np.hstack((bound_faces[0], bound_faces[1]))
            labels = np.array(["dir"] * bound_faces.size)
            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))

            bv = np.zeros(g.num_faces)
            bound_faces = np.where(np.abs(yf - 1) < 1e-4)[0]
            bv[bound_faces] = 1
            param.set_bc_val("flow", bv)

            d["param"] = param

        gb.add_edge_props("kn")
        kn = 1e7
        for e, d in gb.edges():
            mg = d["mortar_grid"]
            d["kn"] = kn * np.ones(mg.num_cells)

    def verify_cv(self, gb, tol=1e-2):
        # The tolerance level here is a bit touchy: With an unstructured grid,
        # and with the flux between subdomains computed as differences between
        # point pressures, uniform flow may not be reproduced if the meshes
        # are not matching (one may get lucky, though). Thus the coarse error
        # tolerance. The current value turned out to be sufficient for all
        # tests considered herein.
        for g, _ in gb.nodes():
            p = gb.node_props(g, "pressure")
            # print(g.cell_centers[1] - p)
            import pdb

            # pdb.set_trace()
            #            if g.dim == 1:

            assert np.allclose(p, g.cell_centers[1], rtol=tol, atol=tol)

    def run_mpfa(self, gb):
        solver_flow = mpfa.MpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)

    def run_vem(self, gb):
        solver_flow = vem_dual.DualVEMMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        up = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "up", up)
        solver_flow.extract_p(gb, "up", "pressure")

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
            fl = None

        elif num_fracs == 1:
            fl = [
                Fracture(np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]]))
            ]
        elif num_fracs == 2:
            fl = [
                Fracture(np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])),
                Fracture(np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0], [0, 0, 1, 1]])),
            ]

        elif num_fracs == 3:
            fl = [
                Fracture(np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])),
                Fracture(np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0], [0, 0, 1, 1]])),
                Fracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])),
            ]

        gb = meshing.simplex_grid(
            fracs=fl, domain=domain, h_min=0.5, h_ideal=0.5, verbose=0
        )

        #        if remove_tags:
        #            internal_flag = FaceTag.FRACTURE
        #            [g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]

        self.set_params(gb)

        return gb

    def set_params(self, gb):

        for g, d in gb:
            param = Parameters(g)

            perm = tensor.SecondOrderTensor(g.dim, kxx=np.ones(g.num_cells))
            param.set_tensor("flow", perm)

            aperture = np.power(1e-6, gb.dim_max() - g.dim)
            param.set_aperture(aperture * np.ones(g.num_cells))

            yf = g.face_centers[1]
            bound_faces = [
                np.where(np.abs(yf - 1) < 1e-4)[0],
                np.where(np.abs(yf) < 1e-4)[0],
            ]
            bound_faces = np.hstack((bound_faces[0], bound_faces[1]))
            labels = np.array(["dir"] * bound_faces.size)
            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))

            bv = np.zeros(g.num_faces)
            bound_faces = np.where(np.abs(yf - 1) < 1e-4)[0]
            bv[bound_faces] = 1
            param.set_bc_val("flow", bv)

            d["param"] = param

        gb.add_edge_props("kn")
        kn = 1e7
        for e, d in gb.edges():
            mg = d["mortar_grid"]
            d["kn"] = kn * np.ones(mg.num_cells)

    def verify_cv(self, gb):
        for g, _ in gb.nodes():
            p = gb.node_props(g, "pressure")
            assert np.allclose(p, g.cell_centers[1], rtol=1e-3, atol=1e-3)

    def run_mpfa(self, gb):
        solver_flow = mpfa.MpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)

    def run_vem(self, gb):
        solver_flow = vem_dual.DualVEMMixedDim("flow")
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

        g = Grid(2, nodes, face_nodes, cell_faces, "TriangleGrid")
        g.compute_geometry()
        g.tags["fracture_faces"][[2, 3, 7, 8]] = 1
        # g.face_normals[1, [2, 3]] = -0.5
        # g.face_normals[1, [7, 8]] = 0.5
        g.global_point_ind = np.arange(nodes.shape[1])

        return g

    def grid_1d(self, num_pts=3):
        g = TensorGrid(np.arange(num_pts))
        g.nodes = np.vstack(
            (np.linspace(0, 1, num_pts), 0.5 * np.ones(num_pts), np.zeros(num_pts))
        )
        g.compute_geometry()
        g.global_point_ind = np.arange(g.num_nodes)
        return g

    def setup(self, remove_tags=False, num_1d=3, pert_node=False):
        g2 = self.grid_2d()
        g1 = self.grid_1d()
        gb = meshing._assemble_in_bucket([[g2], [g1]])

        gb.add_edge_props("face_cells")
        for e, d in gb.edges():
            a = np.zeros((g2.num_faces, g1.num_cells))
            a[2, 1] = 1
            a[3, 0] = 1
            a[7, 0] = 1
            a[8, 1] = 1
            d["face_cells"] = sps.csc_matrix(a.T)
        meshing.create_mortar_grids(gb)

        g_new_2d = self.grid_2d(pert_node)
        g_new_1d = self.grid_1d(num_1d)
        mortars.replace_grids_in_bucket(gb, g_map={g2: g_new_2d, g1: g_new_1d})

        #        if remove_tags:
        #            internal_flag = FaceTag.FRACTURE
        #            [g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]

        gb.assign_node_ordering()

        self.set_params(gb)
        return gb

    def set_params(self, gb):

        for g, d in gb:
            param = Parameters(g)

            perm = tensor.SecondOrderTensor(g.dim, kxx=np.ones(g.num_cells))
            param.set_tensor("flow", perm)

            aperture = np.power(1e-3, gb.dim_max() - g.dim)
            param.set_aperture(aperture * np.ones(g.num_cells))
            if g.dim == 2:
                bound_faces = np.array([0, 10])
                labels = np.array(["dir"] * bound_faces.size)
                param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))

                bv = np.zeros(g.num_faces)
                bound_faces = 10
                bv[bound_faces] = 1
                param.set_bc_val("flow", bv)

            d["param"] = param

        gb.add_edge_props("kn")
        kn = 1e7
        for e, d in gb.edges():
            mg = d["mortar_grid"]
            d["kn"] = kn * np.ones(mg.num_cells)

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
            assert np.allclose(p, g.cell_centers[1], rtol=tol, atol=tol)

    def run_mpfa(self, gb):
        solver_flow = mpfa.MpfaMixedDim("flow")
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)

    def run_vem(self, gb):
        solver_flow = vem_dual.DualVEMMixedDim("flow")
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

# TestGridRefinement1d().test_mortar_grid_darcy()
# a = TestMortar2dSingleFractureCartesianGrid()
# a.test_mpfa_one_frac()
# a.test_tpfa_matching_grids_refine_2d_uniform_flow_larger_domain()
if __name__ == "__main__":
    unittest.main()
# gb = a.setup()
# a = TestMortar3D()
# a.test_mpfa_1_frac_no_refinement()
# a = TestMortar2DSimplexGridStandardMeshing()
# a.test_mpfa_one_frac()
# a.test_tpfa_one_frac_refine_2d()
# TestMortar2DSimplexGrid().test_mpfa_one_frac_coarsened_1d()
# a.test_mpfa_one_frac()
# a = TestMortar2DSimplexGrid()
# a.test_mpfa_one_frac_coarsened_1d_pert_2d_node()
# a.test_mpfa_one_frac_pert_node()
# a.grid_2d()
# a.test_vem_one_frac_coarsen_2d()
# a.test_mpfa_1_frac_no_refinement()
# a.test_mpfa_one_frac_refine_mg()
