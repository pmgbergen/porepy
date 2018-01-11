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
from porepy.grids import refinement, mortar_grid
from porepy.fracs import meshing, mortars

from porepy.params.data import Parameters
from porepy.params.bc import BoundaryCondition
from porepy.grids.grid import FaceTag
from porepy.params import tensor

from porepy.numerics.vem import vem_dual, vem_source

class TestGridRefinement1d(unittest.TestCase):

#------------------------------------------------------------------------------#

    def test_mortar_grid_darcy(self):

        f1 = np.array([[0, 1], [.5, .5]])

        N = [20, 20] #[1, 2]
        gb = meshing.cart_grid([f1], N, **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        for e, d in gb.edges_props():
            mg = d['mortar_grid']
            new_side_grids = {s: refinement.new_grid_1d(g, num_nodes=N[0]+5) \
                              for s, g in mg.side_grids.items()}

            mortars.refine_mortar(mg, new_side_grids)

            # refine the 1d-physical grid
            old_g = gb.sorted_nodes_of_edge(e)[0]
            new_g = refinement.new_grid_1d(old_g, num_nodes=N[0]-5)
            new_g.compute_geometry()

            gb.update_nodes(old_g, new_g)
            mg = d['mortar_grid']
            mortars.refine_co_dimensional_grid(mg, new_g)

#        from porepy.viz import plot_grid
#        plot_grid.plot_grid(gb, alpha=0, info='cfo')

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

    def test_mortar_grid_darcy_2_fracs(self):

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

#TestGridRefinement1d().test_mortar_grid_darcy()
#TestGridRefinement1d().test_mortar_grid_darcy_2_fracs()
TestGridRefinement1d().wietse()
