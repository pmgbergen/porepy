#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of the propagation of fractures. This involves extension of the fracture
grids and splitting of faces in the matrix grid. The post-propagation buckets
are compared to equivalent buckets where the final fracture geometry is applied
at construction.
"""
import unittest

import numpy as np
import porepy as pp

from test.integration.fracture_propagation_utils import check_equivalent_buckets
from test.integration import setup_mixed_dimensional_grids as setup_gb
from test.test_utils import compare_arrays


class BasicsTest(unittest.TestCase):
    def test_propagation_visulization_2d(self):
        """
        Setup intended for visual inspection before and after fracture
        propagation of a single 1d fracture in 2d. No actual testing performed.
        """
        f = np.array([[0, 1], [0.5, 0.5]])
        gb = pp.meshing.cart_grid([f], [4, 2], physdims=[2, 1])
        #        for g, d in gb:
        #            d['cell_tags'] = np.ones(g.num_cells) * g.dim
        #        pp.plot_grid(gb, 'cell_tags', info='f')

        face_2d = [np.array([16])]
        pp.propagate_fracture.propagate_fractures(gb, face_2d)

    #        for g, d in gb:
    #            d['cell_tags'] = np.ones(g.num_cells) * g.dim
    #        pp.plot_grid(gb, 'cell_tags', info='f')

    def test_propagation_visualization_3d(self):
        """
        Setup intended for visual inspection before and after fracture
        propagation of a single 2d fracture in 3d. No actual testing performed.
        """
        f = np.array([[0.0, 1, 1, 0], [0.25, 0.25, 0.75, 0.75], [0.5, 0.5, 0.5, 0.5]])
        gb = pp.meshing.cart_grid([f], [4, 4, 2], physdims=[2, 1, 1])

        #        e = pp.Exporter(gb, 'grid_before', 'test_propagation_3d')
        #        e.write_vtk()
        faces_3d = [np.array([102, 106, 110])]
        #        pp.propagate_fracture.propagate_fracture(gb, gh, gl, faces_h=face_3d)
        pp.propagate_fracture.propagate_fractures(gb, faces_3d)

    #        e_after = pp.Exporter(gb, 'grid_after', 'test_propagation_3d')
    #        e_after.write_vtk()

    def test_equivalence_2d(self):
        """
        This central test checks that certain geometry, connectivity and tags
        are independent of whether the fracture is
        1 premade in its final form
        2-3 grown in a single step (from two different inital fractures)
        4 grown in two steps
        """

        gb_1 = setup_gb.grid_2d_1d([4, 2], 0, 0.75)
        gb_2 = setup_gb.grid_2d_1d([4, 2], 0, 0.25)
        gb_3 = setup_gb.grid_2d_1d([4, 2], 0, 0.50)
        gb_4 = setup_gb.grid_2d_1d([4, 2], 0, 0.25)
        # Split
        pp.propagate_fracture.propagate_fractures(gb_2, [[15, 16]])
        pp.propagate_fracture.propagate_fractures(gb_3, [[16]])
        pp.propagate_fracture.propagate_fractures(gb_4, [[15]])
        pp.propagate_fracture.propagate_fractures(gb_4, [[16]])
        # Check that the four grid buckets are equivalent
        check_equivalent_buckets([gb_1, gb_2, gb_3, gb_4])

    def test_equivalence_3d(self):
        """
        3d version of previous test.
        This central test checks that certain geometry, connectivity and tags
        are independent of whether the fracture is
        1 premade in its final form
        2-3 grown in a single step (from two different inital fractures)
        4 grown in two steps
        """
        # Make buckets
        gb_1 = setup_gb.grid_3d_2d([4, 2, 2], 0, 0.75)
        gb_2 = setup_gb.grid_3d_2d([4, 2, 2], 0, 0.25)
        gb_3 = setup_gb.grid_3d_2d([4, 2, 2], 0, 0.50)
        gb_4 = setup_gb.grid_3d_2d([4, 2, 2], 0, 0.25)
        # Split
        pp.propagate_fracture.propagate_fractures(gb_2, [[53, 54]])
        pp.propagate_fracture.propagate_fractures(gb_3, [[54]])
        pp.propagate_fracture.propagate_fractures(gb_4, [[53]])
        pp.propagate_fracture.propagate_fractures(gb_4, [[54]])
        # Check that the four grid buckets are equivalent
        check_equivalent_buckets([gb_1, gb_2, gb_3, gb_4])

    def test_two_fractures_2d(self):
        """
        Two fractures growing towards each other, but not meeting.

        Tests simultanous growth of two fractures in multiple steps, and growth
        of one fracture in the presence of an inactive one.
        """
        f_1 = np.array([[0, 0.25], [0.5, 0.5]])
        f_2 = np.array([[1.75, 2], [0.5, 0.5]])
        gb = pp.meshing.cart_grid([f_1, f_2], [8, 2], physdims=[2, 1])

        f_1 = np.array([[0, 0.5], [0.5, 0.5]])
        f_2 = np.array([[1.5, 2], [0.5, 0.5]])
        gb_1 = pp.meshing.cart_grid([f_1, f_2], [8, 2], physdims=[2, 1])

        f_1 = np.array([[0, 0.75], [0.5, 0.5]])
        f_2 = np.array([[1.25, 2], [0.5, 0.5]])
        gb_2 = pp.meshing.cart_grid([f_1, f_2], [8, 2], physdims=[2, 1])

        f_1 = np.array([[0, 1.0], [0.5, 0.5]])
        gb_3 = pp.meshing.cart_grid([f_1, f_2], [8, 2], physdims=[2, 1])

        faces = [np.array([27]), np.array([32])]
        pp.propagate_fracture.propagate_fractures(gb, faces)
        check_equivalent_buckets([gb, gb_1])

        faces = [np.array([28]), np.array([31])]
        pp.propagate_fracture.propagate_fractures(gb, faces)
        check_equivalent_buckets([gb, gb_2])

        faces = [np.array([29]), np.empty(0, dtype=int)]
        pp.propagate_fracture.propagate_fractures(gb, faces)
        check_equivalent_buckets([gb, gb_3])


class Propagation3dSingleCartGrid(unittest.TestCase):
    def _make_grid(self):
        frac_1 = np.array([[0, 1, 1, 0], [0, 0, 3, 3], [1, 1, 1, 1]])
        gb = pp.meshing.cart_grid([frac_1], [2, 3, 2])

        for g, _ in gb:
            hit = g.nodes[0] > 1.5
            g.nodes[2, hit] += 1

        gb.compute_geometry()

        pp.contact_conditions.set_projections(gb)

        return gb

    def test_simple_propagation_order(self):

        gb = self._make_grid()

        faces_to_split = [[43], [41, 45]]

        g = gb.grids_of_dimension(2)[0]
        cc = g.cell_centers
        fc = g.face_centers
        cv = g.cell_volumes
        fa = g.face_areas

        gh = gb.grids_of_dimension(3)[0]

        new_fc = [
            np.array([[1.5, 2, 1.5], [1, 1.5, 2], [1.5, 2, 1.5]]),
            np.array([[1.5, 2, 2, 1.5], [0, 0.5, 2.5, 3], [1.5, 2, 2, 1.5]]),
        ]

        new_cell_volumes = [np.array([np.sqrt(2)]), np.sqrt(2) * np.ones(2)]

        # The first splitting will create no new nodes (only more tip nodes)
        # Second splitting generates 8 new nodes, since the domain is split in two
        num_nodes = gh.num_nodes + np.array([0, 8])

        for si, split in enumerate(faces_to_split):
            pp.propagate_fracture.propagate_fractures(gb, [split])
            cc, fc, cv = self._verify(
                gb, split, cc, fc, cv, new_cell_volumes[si], new_fc[si], num_nodes[si]
            )

    def test_propagation_from_bottom(self):
        # Three propagation steps. Should be equally simple as the simple_propagation_order
        gb = self._make_grid()

        faces_to_split = [[41], [43], [45]]

        g = gb.grids_of_dimension(2)[0]
        cc = g.cell_centers
        fc = g.face_centers
        cv = g.cell_volumes
        fa = g.face_areas

        gh = gb.grids_of_dimension(3)[0]

        new_fc = [
            np.array([[1.5, 2, 1.5], [0, 0.5, 1], [1.5, 2, 1.5]]),
            np.array([[2, 1.5], [1.5, 2], [2, 1.5]]),
            np.array([[2, 1.5], [2.5, 3], [2, 1.5]]),
        ]

        new_cell_volumes = [np.sqrt(2), np.sqrt(2), np.sqrt(2)]

        num_nodes = gh.num_nodes + np.array([2, 4, 8])

        for si, split in enumerate(faces_to_split):
            pp.propagate_fracture.propagate_fractures(gb, [split])
            cc, fc, cv = self._verify(
                gb, split, cc, fc, cv, new_cell_volumes[si], new_fc[si], num_nodes[si]
            )

    def test_propagation_from_sides(self):
        # Open the same face from two side simultanously.
        gb = self._make_grid()

        faces_to_split = [[41, 45], [43, 43]]

        g = gb.grids_of_dimension(2)[0]
        cc = g.cell_centers
        fc = g.face_centers
        cv = g.cell_volumes
        fa = g.face_areas

        gh = gb.grids_of_dimension(3)[0]

        new_fc = [
            np.array(
                [
                    [1.5, 2, 1.5, 1.5, 2, 1.5],
                    [0, 0.5, 1, 2, 2.5, 3],
                    [1.5, 2, 1.5, 1.5, 2, 1.5],
                ]
            ),
            np.array([[2], [1.5], [2]]),
        ]

        new_cell_volumes = [np.sqrt(2) * np.ones(2), np.sqrt(2)]

        num_nodes = gh.num_nodes + np.array([4, 8])

        for si, split in enumerate(faces_to_split):
            pp.propagate_fracture.propagate_fractures(gb, [split])
            cc, fc, cv = self._verify(
                gb, split, cc, fc, cv, new_cell_volumes[si], new_fc[si], num_nodes[si]
            )

    def _verify(self, gb, split, cc, fc, cv, new_cell_volumes, new_fc, num_nodes):
        gh = gb.grids_of_dimension(gb.dim_max())[0]
        g = gb.grids_of_dimension(gb.dim_max() - 1)[0]
        new_cc = gh.face_centers[:, split]
        if len(split) == 1:
            new_cc = new_cc.reshape((-1, 1))
        cc = np.append(cc, new_cc, axis=1)
        compare_arrays(g.cell_centers, cc)

        fc = np.append(fc, new_fc, axis=1)
        compare_arrays(g.face_centers, fc)

        cv = np.append(cv, new_cell_volumes)

        self.assertTrue(np.allclose(g.cell_volumes, cv))

        proj = gb.node_props(g)["tangential_normal_projection"]
        self.assertTrue(proj.normals.shape[1] == g.num_cells)
        self.assertTrue(
            np.logical_or(np.all(proj.normals[2] < 0), np.all(proj.normals[2] > 0))
        )

        hit = np.logical_and(g.face_centers[0] > 1.1, g.face_centers[0] < 1.9)
        self.assertTrue(np.allclose(g.face_areas[hit], np.sqrt(2)))
        self.assertTrue(np.allclose(g.face_areas[np.logical_not(hit)], 1))

        self.assertTrue(num_nodes == gh.num_nodes)

        return cc, fc, cv


if __name__ == "__main__":
    unittest.main()
