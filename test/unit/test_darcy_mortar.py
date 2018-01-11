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

TestGridRefinement1d().test_mortar_grid_darcy()
