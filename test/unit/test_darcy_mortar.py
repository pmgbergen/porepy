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
from porepy.grids.grid import FaceTag, Grid
from porepy.params import tensor


from porepy.numerics.vem import vem_dual, vem_source
from porepy.numerics.fv import tpfa, mpfa

class TestGridRefinement1d(unittest.TestCase):

#------------------------------------------------------------------------------#

    def est_mortar_grid_darcy(self):

        f1 = np.array([[0, 1], [.5, .5]])

        N = [2, 2] #[1, 2]
        gb = meshing.cart_grid([f1], N, **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()
        tol = 1e-6

        g_map = {}
        mg_map = {}

        for e, d in gb.edges_props():
            mg = d['mortar_grid']
            mg_map[mg] = {s: refinement.new_grid_1d(g, num_nodes=N[0]+2) \
                              for s, g in mg.side_grids.items()}

        gb = mortars.replace_grids_in_bucket(gb, g_map, mg_map, tol)
        gb.assign_node_ordering()


        np.set_printoptions( linewidth=9999, precision=4)
        for e, d in gb.edges_props():
            mg = d['mortar_grid']
            print(mg.high_to_mortar_int.todense())
            print(mg.mortar_to_high_int().todense())



        internal_flag = FaceTag.FRACTURE
        [g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]

        gb.add_node_props(['param'])
        for g, d in gb:
            param = Parameters(g)

            perm = tensor.SecondOrder(g.dim, kxx=np.ones(g.num_cells))
            param.set_tensor("flow", perm)

            aperture = np.power(1e-3, gb.dim_max() - g.dim)
            param.set_aperture(aperture*np.ones(g.num_cells))

            mask = np.logical_and(g.cell_centers[1, :] < 0.3,
                                  g.cell_centers[0, :] < 0.3)
            source = np.zeros(g.num_cells)
            source[mask] = g.cell_volumes[mask]*aperture
            param.set_source("flow", source)

            bound_faces = g.get_domain_boundary_faces()
            labels = np.array(['dir'] * bound_faces.size)
            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", np.zeros(g.num_faces))

            d['param'] = param

        gb.add_edge_prop('kn')
        for e, d in gb.edges_props():
            gn = gb.sorted_nodes_of_edge(e)
            aperture = np.power(1e-3, gb.dim_max() - gn[0].dim)
            d['kn'] = np.ones(gn[0].num_cells) / aperture

        # Choose and define the solvers and coupler
        solver_flow = vem_dual.DualVEMMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        solver_source = vem_source.IntegralMixedDim('flow')
        A_source, b_source = solver_source.matrix_rhs(gb)

        up = sps.linalg.spsolve(A_flow+A_source, b_flow+b_source)
        solver_flow.split(gb, "up", up)

        solver_flow.extract_p(gb, "up", "p")

        from porepy.viz.exporter import Exporter
        save = Exporter(gb, "vem", folder="vem")
        save.write_vtk(["p"])

#------------------------------------------------------------------------------#

    def est_mortar_grid_darcy_2_fracs(self):

        f1 = np.array([[0, 1], [.5, .5]])
        f2 = np.array([[.5, .5], [0, 1]])

        s = 10
        N = [2*s, 2*s] #[1, 2]
        gb = meshing.cart_grid([f1, f2], N, **{'physdims': [1, 1]})
        gb.compute_geometry()

#        g_map = {}
#        for g, d in gb:
#            if g.dim == 1:
#                # refine the 1d-physical grid
#                #g_map[g] = refinement.refine_grid_1d(g)
#                g_map[g] = refinement.new_grid_1d(g, num_nodes=N[0]+1)
#                g_map[g].compute_geometry()
#
#        mg_map = {}
#        for e, d in gb.edges_props():
#            mg = d['mortar_grid']
#            mg_map[mg] = {s: refinement.new_grid_1d(g, num_nodes=N[0]+2) \
#                                        for s, g in mg.side_grids.items()}


#        gb = mortars.replace_grids_in_bucket(gb, g_map, mg_map)
        gb.assign_node_ordering()

#        from porepy.viz import plot_grid
#        plot_grid.plot_grid(gb, alpha=0, info='c')

        internal_flag = FaceTag.FRACTURE
        [g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]

        gb.add_node_props(['param'])
        for g, d in gb:
            param = Parameters(g)

            perm = tensor.SecondOrder(g.dim, kxx=np.ones(g.num_cells))
            param.set_tensor("flow", perm)

            aperture = np.power(1e-3, gb.dim_max() - g.dim)
            param.set_aperture(aperture*np.ones(g.num_cells))

            mask = np.logical_and(g.cell_centers[1, :] < 0.3,
                                  g.cell_centers[0, :] < 0.3)
            source = np.zeros(g.num_cells)
            source[mask] = g.cell_volumes[mask]*aperture
            param.set_source("flow", source)

            bound_faces = g.get_domain_boundary_faces()
            if bound_faces.size == 0:
                bc =  BoundaryCondition(g, np.empty(0), np.empty(0))
                param.set_bc("flow", bc)
            else:
                labels = np.array(['dir'] * bound_faces.size)
                param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
                param.set_bc_val("flow", np.zeros(g.num_faces))

            d['param'] = param

        gb.add_edge_prop('kn')
        for e, d in gb.edges_props():
            gn = gb.sorted_nodes_of_edge(e)
            aperture = np.power(1e-3, gb.dim_max() - gn[0].dim)
            d['kn'] = np.ones(gn[0].num_cells) / aperture

        # Choose and define the solvers and coupler
        solver_flow = vem_dual.DualVEMMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)
        np.set_printoptions(linewidth=500)
        print(A_flow.shape)
        print(A_flow.todense())

        solver_source = vem_source.IntegralMixedDim('flow')
        A_source, b_source = solver_source.matrix_rhs(gb)

        up = sps.linalg.spsolve(A_flow+A_source, b_flow+b_source)
        solver_flow.split(gb, "up", up)

        solver_flow.extract_p(gb, "up", "p")

        from porepy.viz.exporter import Exporter
        save = Exporter(gb, "vem", folder="vem")
        save.write_vtk(["p"])

#------------------------------------------------------------------------------#

    def wietse(self):
        from porepy.fracs import importer

        tol = 1e-5
        mesh_kwargs = {}
        mesh_size = 0.045
        mesh_kwargs['mesh_size'] = {'mode': 'constant',
                                'value': mesh_size, 'bound_value': mesh_size}
        domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}

        file_name = 'wietse.csv'
        gb = importer.mesh_from_csv(file_name, mesh_kwargs, domain)

        g_map = {}
        for g, d in gb:
            if g.dim == 1:
                if g.nodes[1, 0] < .51:
                    # refine the 1d-physical grid
                    #g_map[g] = refinement.refine_grid_1d(g)
                    num_nodes = g.num_nodes+40
                    g_map[g] = refinement.new_grid_1d(g, num_nodes=num_nodes)
                    g_map[g].compute_geometry()

        mg_map = {}
        for e, d in gb.edges_props():
            mg = d['mortar_grid']
            g_l = gb.sorted_nodes_of_edge(e)[0]
            if g_l.nodes[1, 0] < .51:
                num_nodes = g_l.num_nodes+100
                print(num_nodes)
                mg_map[mg] = {s: refinement.new_grid_1d(g, num_nodes=num_nodes) \
                                            for s, g in mg.side_grids.items()}


        gb = mortars.replace_grids_in_bucket(gb, g_map, mg_map)
        gb.assign_node_ordering()

        internal_flag = FaceTag.FRACTURE
        [g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]

        gb.add_node_props(['param'])
        for g, d in gb:
            param = Parameters(g)

            perm = tensor.SecondOrder(g.dim, kxx=np.ones(g.num_cells))
            param.set_tensor("flow", perm)

            aperture = np.power(1e-3, gb.dim_max() - g.dim)
            param.set_aperture(aperture*np.ones(g.num_cells))

            param.set_source("flow", np.zeros(g.num_cells))

            bound_faces = g.get_domain_boundary_faces()
            if bound_faces.size == 0:
                bc =  BoundaryCondition(g, np.empty(0), np.empty(0))
                param.set_bc("flow", bc)
            else:
                bound_face_centers = g.face_centers[:, bound_faces]

                top = bound_face_centers[1, :] > domain['ymax'] - tol
                bottom = bound_face_centers[1, :] < domain['ymin'] + tol

                labels = np.array(['neu'] * bound_faces.size)
                labels[np.logical_or(top, bottom)] = 'dir'

                bc_val = np.zeros(g.num_faces)
                bc_val[bound_faces[top]] = 1

                param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
                param.set_bc_val("flow", bc_val)

            d['param'] = param

        gb.add_edge_prop('kn')
        for e, d in gb.edges_props():
            gn = gb.sorted_nodes_of_edge(e)
            aperture = np.power(1e-3, gb.dim_max() - gn[0].dim)
            d['kn'] = np.ones(gn[0].num_cells) / aperture

        # Choose and define the solvers and coupler
        solver_flow = vem_dual.DualVEMMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        solver_source = vem_source.IntegralMixedDim('flow')
        A_source, b_source = solver_source.matrix_rhs(gb)

        up = sps.linalg.spsolve(A_flow+A_source, b_flow+b_source)
        solver_flow.split(gb, "up", up)

        solver_flow.extract_p(gb, "up", "p")

        from porepy.viz.exporter import Exporter
        save = Exporter(gb, "sol", folder="wietse")
        save.write_vtk(["p"])


#------------------------------------------------------------------------------#

class TestMortar2dSingleFractureCartesianGrid(unittest.TestCase):

    def set_param_flow(self, gb, no_flow=False, kn=1e3):
        # Set up flow field with uniform flow in y-direction
        gb.add_node_props(['param'])
        for g, d in gb:
            param = Parameters(g)

            perm = tensor.SecondOrder(g.dim, kxx=np.ones(g.num_cells))
            param.set_tensor("flow", perm)

            aperture = np.power(1e-3, gb.dim_max() - g.dim)
            param.set_aperture(aperture*np.ones(g.num_cells))

            if g.dim == 2:
                b_val = np.zeros(g.num_faces)
                bound_faces = bc.face_on_side(g, ['ymin', 'ymax'])
                if no_flow:
                    b_val[bound_faces[0]] = 1
                    b_val[bound_faces[1]] = 1
                bound_faces = np.hstack((bound_faces[0], bound_faces[1]))
                labels = np.array(['dir'] * bound_faces.size)
                param.set_bc("flow", bc.BoundaryCondition(g, bound_faces, labels))

                bound_faces = bc.face_on_side(g, 'ymax')[0]
                b_val[bound_faces] = 1
                param.set_bc_val("flow", b_val)

            d['param'] = param

        gb.add_edge_prop('kn')
        for e, d in gb.edges_props():
            mg = d['mortar_grid']
            gn = gb.sorted_nodes_of_edge(e)
            d['kn'] = kn * np.ones(mg.num_cells)

    def set_grids(self, N, num_nodes_mortar, num_nodes_1d, physdims=[1, 1]):
        f1 = np.array([[0, physdims[0]], [.5, .5]])

        gb = meshing.cart_grid([f1], N, **{'physdims': physdims})
        gb.compute_geometry()
        gb.assign_node_ordering()

        for e, d in gb.edges_props():
            mg = d['mortar_grid']
            new_side_grids = {s: refinement.new_grid_1d(g, num_nodes=num_nodes_mortar) \
                              for s, g in mg.side_grids.items()}

            mortars.update_mortar_grid(mg, new_side_grids, tol=1e-4)

            # refine the 1d-physical grid
            old_g = gb.sorted_nodes_of_edge(e)[0]
            new_g = refinement.new_grid_1d(old_g, num_nodes=num_nodes_1d)
            new_g.compute_geometry()

            gb.update_nodes(old_g, new_g)
            mg = d['mortar_grid']
            mortars.update_physical_low_grid(mg, new_g, tol=1e-4)
        return gb

    def test_tpfa_matching_grids_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=True)

        solver_flow = tpfa.TpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        assert np.all(p[:3] == 1)
        assert np.all(p[3:] == 0)

    def test_tpfa_matching_grids_refine_1d_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(gb, no_flow=True)

        solver_flow = tpfa.TpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        assert np.all(p[:4] == 1)
        assert np.all(p[4:] == 0)

    def test_tpfa_matching_grids_refine_mortar_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=True)

        solver_flow = tpfa.TpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

        p = sps.linalg.spsolve(A_flow, b_flow)
        assert np.all(p[:3] == 1)
        assert np.all(p[3:] == 0)

    def test_tpfa_matching_grids_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = tpfa.TpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_prop(g_2d, 'pressure')
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=kn)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_prop(g_1d, 'pressure')
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1], rtol=kn)

    def test_tpfa_matching_grids_refine_1d_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = tpfa.TpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_prop(g_2d, 'pressure')
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1./kn)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_prop(g_1d, 'pressure')
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_tpfa_matching_grids_refine_mortar_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = tpfa.TpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_prop(g_2d, 'pressure')
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_prop(g_1d, 'pressure')
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_tpfa_matching_grids_refine_2d_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[2, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = tpfa.TpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_prop(g_2d, 'pressure')
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_prop(g_1d, 'pressure')
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_tpfa_matching_grids_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2,
                            physdims=[2, 1])
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = tpfa.TpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_prop(g_2d, 'pressure')
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=kn)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_prop(g_1d, 'pressure')
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1], rtol=kn)

    def test_tpfa_matching_grids_refine_1d_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3,
                            physdims=[2, 1])
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = tpfa.TpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_prop(g_2d, 'pressure')
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1./kn)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_prop(g_1d, 'pressure')
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_tpfa_matching_grids_refine_mortar_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2,
                            physdims=[2, 1])
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = tpfa.TpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_prop(g_2d, 'pressure')
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_prop(g_1d, 'pressure')
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_tpfa_matching_grids_refine_2d_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(N=[2, 2], num_nodes_mortar=2, num_nodes_1d=2,
                            physdims=[2, 1])
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = tpfa.TpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_prop(g_2d, 'pressure')
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_prop(g_1d, 'pressure')
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_mpfa_matching_grids_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=True)

        solver_flow = mpfa.MpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        assert np.all(p[:3] == 1)
        assert np.all(p[3:] == 0)

    def test_mpfa_matching_grids_refine_1d_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(gb, no_flow=True)

        solver_flow = mpfa.MpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        assert np.all(p[:4] == 1)
        assert np.all(p[4:] == 0)

    def test_mpfa_matching_grids_refine_mortar_no_flow(self):
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=True)

        solver_flow = mpfa.MpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

        p = sps.linalg.spsolve(A_flow, b_flow)
        assert np.all(p[:3] == 1)
        assert np.all(p[3:] == 0)

    def test_mpfa_matching_grids_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = mpfa.MpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_prop(g_2d, 'pressure')
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=kn)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_prop(g_1d, 'pressure')
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1], rtol=kn)

    def test_mpfa_matching_grids_refine_1d_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3)
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = mpfa.MpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_prop(g_2d, 'pressure')
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1./kn)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_prop(g_1d, 'pressure')
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_mpfa_matching_grids_refine_mortar_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = mpfa.MpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_prop(g_2d, 'pressure')
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_prop(g_1d, 'pressure')
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_mpfa_matching_grids_refine_2d_uniform_flow(self):

        kn = 1e4
        gb = self.set_grids(N=[2, 2], num_nodes_mortar=2, num_nodes_1d=2)
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = mpfa.MpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_prop(g_2d, 'pressure')
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_prop(g_1d, 'pressure')
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_mpfa_matching_grids_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=2,
                            physdims=[2, 1])
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = mpfa.MpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_prop(g_2d, 'pressure')
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=kn)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_prop(g_1d, 'pressure')
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1], rtol=kn)

    def test_mpfa_matching_grids_refine_1d_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=2, num_nodes_1d=3,
                            physdims=[2, 1])
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = mpfa.MpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_prop(g_2d, 'pressure')
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1./kn)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_prop(g_1d, 'pressure')
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_mpfa_matching_grids_refine_mortar_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(N=[1, 2], num_nodes_mortar=3, num_nodes_1d=2,
                            physdims=[2, 1])
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = mpfa.MpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_prop(g_2d, 'pressure')
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_prop(g_1d, 'pressure')
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

    def test_mpfa_matching_grids_refine_2d_uniform_flow_larger_domain(self):

        kn = 1e4
        gb = self.set_grids(N=[2, 2], num_nodes_mortar=2, num_nodes_1d=2,
                            physdims=[2, 1])
        self.set_param_flow(gb, no_flow=False, kn=kn)

        solver_flow = mpfa.MpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)
        g_2d = gb.grids_of_dimension(2)[0]
        p_2d = gb.node_prop(g_2d, 'pressure')
        # NOTE: This will not be entirely correct due to impact of normal permeability at fracture
        assert np.allclose(p_2d, g_2d.cell_centers[1], rtol=1e-4)

        g_1d = gb.grids_of_dimension(1)[0]
        p_1d = gb.node_prop(g_1d, 'pressure')
        # NOTE: This will not be entirely correct,
        assert np.allclose(p_1d, g_1d.cell_centers[1])

#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#

class TestMortar2DSimplexGridStandardMeshing(unittest.TestCase):

    def setup(self, num_fracs=1, remove_tags=False, alpha_1d=None, alpha_mortar=None, alpha_2d=None):

        domain = {'xmin':0, 'xmax': 1, 'ymin': 0, 'ymax':1}

        if num_fracs == 0:
            p = np.zeros((2, 0))

        elif num_fracs == 1:
            p = [np.array([[0, 1], [0.5, 0.5]])]
#            p = [np.array([[0.5, 0.5], [0, 1]])]
        elif num_fracs == 2:
            p = [np.array([[0, 0.5], [0.5, 0.5]]),
                 np.array([[0.5, 1], [0.5, 0.5]]),
                 np.array([[0.5, 0.5], [0, 0.5]]),
                 np.array([[0.5, 0.5], [0.5, 1]])]
        mesh_size = {'value': 0.3, 'bound_value': 0.3}
        gb = meshing.simplex_grid(fracs=p, domain=domain, mesh_size=mesh_size, verbose=0)
#        gb = meshing.cart_grid([np.array([[0.5, 0.5], [0, 1]])],np.array([10, 10]),
#                               physdims=np.array([1, 1]))

        gmap = {}

        # Refine 2D grid?
        if alpha_2d is not None:
            mesh_size = {'value': 0.3 * alpha_2d, 'bound_value': 0.3 * alpha_2d}
            gbn = meshing.simplex_grid(fracs=p, domain=domain,
                                      mesh_size=mesh_size, verbose=0)
            go = gb.grids_of_dimension(2)[0]
            gn = gbn.grids_of_dimension(2)[0]
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
                    gmap[g] = refinement.new_grid_1d(g, num_nodes=num_nodes)
                    gmap[g].compute_geometry()
        # Refine mortar grid
        mg_map = {}
        if alpha_mortar is not None:
            for e, d in gb.edges_props():
                mg = d['mortar_grid']
                if mg.dim == 1:
                    mg_map[mg] = {}
                    for s, g in mg.side_grids.items():
                        num_nodes = int(g.num_nodes*alpha_mortar)
                        mg_map[mg][s] = refinement.new_grid_1d(g, num_nodes=num_nodes)

        gb = mortars.replace_grids_in_bucket(gb, gmap, mg_map, tol=1e-4)

        if remove_tags:
            internal_flag = FaceTag.FRACTURE
            [g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]


        gb.assign_node_ordering()

        self.set_params(gb)

        return gb

    def set_params(self, gb):


        for g, d in gb:
            param = Parameters(g)

            perm = tensor.SecondOrder(g.dim, kxx=np.ones(g.num_cells))
            param.set_tensor("flow", perm)

            aperture = np.power(1e-3, gb.dim_max() - g.dim)
            param.set_aperture(aperture*np.ones(g.num_cells))

            yf = g.face_centers[1]
            bound_faces = [np.where(np.abs(yf - 1) < 1e-4)[0],
                           np.where(np.abs(yf) < 1e-4)[0]]
            bound_faces = np.hstack((bound_faces[0], bound_faces[1]))
            labels = np.array(['dir'] * bound_faces.size)
            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))

            bv = np.zeros(g.num_faces)
            bound_faces = np.where(np.abs(yf-1) < 1e-4)[0]
            bv[bound_faces] = 1
            param.set_bc_val("flow", bv)

            d['param'] = param

        gb.add_edge_prop('kn')
        kn = 1e7
        for e, d in gb.edges_props():
            mg = d['mortar_grid']
            d['kn'] = kn * np.ones(mg.num_cells)

    def verify_cv(self, gb, tol=1e-5):
        # The tolerance level here is a bit touchy: With an unstructured grid,
        # and with the flux between subdomains computed as differences between
        # point pressures, uniform flow may not be reproduced if the meshes
        # are not matching (one may get lucky, though). Thus the coarse error
        # tolerance. The current value turned out to be sufficient for all
        # tests considered herein.
        for g in gb.nodes():
            p = gb.node_prop(g, 'pressure')
            if g.dim == 1:
#                print(g.cell_centers[1])
                assert np.allclose(p, g.cell_centers[1], rtol=tol, atol=tol)

    def run_mpfa(self, gb):
        solver_flow = mpfa.MpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)

    def run_vem(self, gb):
        solver_flow = vem_dual.DualVEMMixedDim('flow')
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
        gb = self.setup(num_fracs=1, test_vem_one_fracalpha_1d=2)
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

        domain = {'xmin':0, 'xmax': 1, 'ymin': 0, 'ymax':1, 'zmin':0, 'zmax':1}

        if num_fracs == 0:
            fl = None

        elif num_fracs == 1:
            fl = [Fracture(np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5],
                                     [0, 0, 1, 1]]))]
        elif num_fracs == 2:
            fl = [Fracture(np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5],
                                     [0, 0, 1, 1]])),
                  Fracture(np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0],
                                     [0, 0, 1, 1]]))]

        elif num_fracs == 3:
            fl = [Fracture(np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5],
                                     [0, 0, 1, 1]])),
                  Fracture(np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0],
                                     [0, 0, 1, 1]])),
                  Fracture(np.array([[0, 1, 1, 0],
                                     [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]]))]

        gb = meshing.simplex_grid(fracs=fl, domain=domain, h_min=0.5, h_ideal=0.5, verbose=0)

        if remove_tags:
            internal_flag = FaceTag.FRACTURE
            [g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]

        self.set_params(gb)

        return gb

    def set_params(self, gb):


        for g, d in gb:
            param = Parameters(g)

            perm = tensor.SecondOrder(g.dim, kxx=np.ones(g.num_cells))
            param.set_tensor("flow", perm)

            aperture = np.power(1e-6, gb.dim_max() - g.dim)
            param.set_aperture(aperture*np.ones(g.num_cells))

            yf = g.face_centers[1]
            bound_faces = [np.where(np.abs(yf - 1) < 1e-4)[0],
                           np.where(np.abs(yf) < 1e-4)[0]]
            bound_faces = np.hstack((bound_faces[0], bound_faces[1]))
            labels = np.array(['dir'] * bound_faces.size)
            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))

            bv = np.zeros(g.num_faces)
            bound_faces = np.where(np.abs(yf-1) < 1e-4)[0]
            bv[bound_faces] = 1
            param.set_bc_val("flow", bv)

            d['param'] = param

        gb.add_edge_prop('kn')
        kn = 1e7
        for e, d in gb.edges_props():
            mg = d['mortar_grid']
            d['kn'] = kn * np.ones(mg.num_cells)

    def verify_cv(self, gb):
        for g in gb.nodes():
            p = gb.node_prop(g, 'pressure')
            assert np.allclose(p, g.cell_centers[1], rtol=1e-3, atol=1e-3)

    def run_mpfa(self, gb):
        solver_flow = mpfa.MpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)


    def test_mpfa_no_fracs(self):
        gb = self.setup(num_fracs=0)
        self.run_mpfa(gb)
        self.verify_cv(gb)

    def test_mpfa_1_frac_no_refinement(self):
        gb = self.setup(num_fracs=1)

        if False:
            self.run_mpfa(gb)
        else:
        # Choose and define the solvers and coupler
            gb = self.setup(num_fracs=1, remove_tags=True)

            solver_flow = vem_dual.DualVEMMixedDim('flow')
            A_flow, b_flow = solver_flow.matrix_rhs(gb)

            solver_source = vem_source.IntegralMixedDim('flow')
            A_source, b_source = solver_source.matrix_rhs(gb)

            up = sps.linalg.spsolve(A_flow+A_source, b_flow+b_source)
            solver_flow.split(gb, "up", up)

            solver_flow.extract_p(gb, "up", "pressure")


        self.verify_cv(gb)


        # TODO: Add check that mortar flux scales with mortar area

class TestMortar2DSimplexGrid(unittest.TestCase):

    def grid_2d(self):
        nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0.5, 0.5, 0],
                          [0, 0.5, 0],
                          [0, 0.5, 0], [0.5, 0.5, 0], [1, 0.5, 0], [1, 1, 0],
                          [0, 1, 0]]).T

        fn = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 3], [3, 1],
                       [5, 6], [6, 7], [7, 8], [8, 9], [9, 5], [9, 6], [6, 8]]).T
        cf = np.array([[3, 4, 5], [0, 6, 5], [1, 2, 6],
                       [7, 12, 11], [12, 13, 10], [8, 9, 13]]).T
        cols = np.tile(np.arange(fn.shape[1]), (fn.shape[0], 1)).ravel('F')
        face_nodes = sps.csc_matrix((np.ones_like(cols),
                                     (fn.ravel('F'), cols)))

        cols = np.tile(np.arange(cf.shape[1]), (cf.shape[0], 1)).ravel('F')
        cell_faces = sps.csc_matrix((np.ones_like(cols),
                                     (cf.ravel('F'), cols)))

        g = Grid(2, nodes, face_nodes, cell_faces, 'TriangleGrid')
        g.compute_geometry()
        g.global_point_ind = np.arange(nodes.shape[1])

        return g

    def grid_1d(self):
        g = TensorGrid(np.array([0, 1, 2]))
        g.nodes = np.array([[0, 0.5, 1], [0.5, 0.5, 0.5], [0, 0, 0]])
        g.compute_geometry()
        g.global_point_ind = np.arange(g.num_nodes)
        return g

    def setup(self):
        g2 = self.grid_2d()
        g1 = self.grid_1d()
        gb = meshing.assemble_in_bucket([[g2], [g1]])

        gb.add_edge_prop('face_cells')
        for e, d in gb.edges_props():
            a = np.zeros((g2.num_faces, g1.num_cells))
            a[2, 1] = 1
            a[3, 0] = 1
            a[7, 0] = 1
            a[8, 1] = 1
            d['face_cells'] = sps.csc_matrix(a.T)
        meshing.create_mortar_grids(gb)

        gb.assign_node_ordering()

        self.set_params(gb)
        return gb

    def set_params(self, gb):


        for g, d in gb:
            param = Parameters(g)

            perm = tensor.SecondOrder(g.dim, kxx=np.ones(g.num_cells))
            param.set_tensor("flow", perm)

            aperture = np.power(1e-3, gb.dim_max() - g.dim)
            param.set_aperture(aperture*np.ones(g.num_cells))
            if g.dim == 2:
                yf = g.face_centers[1]
                bound_faces = [np.where(np.abs(yf) < 1e-4)[0],
                               np.where(np.abs(yf-1) < 1e-4)[0]]
                bound_faces = np.hstack((bound_faces[0], bound_faces[1]))
                bound_faces = np.array([0, 10])
                labels = np.array(['dir'] * bound_faces.size)
                param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))

                bv = np.zeros(g.num_faces)
                bound_faces = np.where(np.abs(yf-1) < 1e-4)[0]
                bound_faces = 10
                bv[bound_faces] = 1
                param.set_bc_val("flow", bv)

            d['param'] = param

        gb.add_edge_prop('kn')
        kn = 1e7
        for e, d in gb.edges_props():
            mg = d['mortar_grid']
            d['kn'] = kn * np.ones(mg.num_cells)

    def verify_cv(self, gb, tol=1e-5):
        # The tolerance level here is a bit touchy: With an unstructured grid,
        # and with the flux between subdomains computed as differences between
        # point pressures, uniform flow may not be reproduced if the meshes
        # are not matching (one may get lucky, though). Thus the coarse error
        # tolerance. The current value turned out to be sufficient for all
        # tests considered herein.
        for g in gb.nodes():
            p = gb.node_prop(g, 'pressure')

#                print(g.cell_centers[1])
            assert np.allclose(p, g.cell_centers[1], rtol=tol, atol=tol)

    def run_mpfa(self, gb):
        solver_flow = mpfa.MpfaMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)

        p = sps.linalg.spsolve(A_flow, b_flow)
        solver_flow.split(gb, "pressure", p)

    def test_mpfa_one_frac(self):
        gb = self.setup()
        self.run_mpfa(gb)
        self.verify_cv(gb)


#TestGridRefinement1d().test_mortar_grid_darcy()
a = TestMortar2DSimplexGrid()
gb = a.setup()
a.test_mpfa_one_frac()
#a = TestMortar2DSimplexGrid()
#a.grid_2d()
#a.test_vem_one_frac_coarsen_2d()
#a.test_mpfa_1_frac_no_refinement()
#a.test_mpfa_one_frac_refine_mg()
