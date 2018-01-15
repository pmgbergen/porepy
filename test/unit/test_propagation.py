#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:54:32 2018

@author: ivar
"""

import numpy as np
import unittest
import logging
import time
from porepy.fracs import meshing, propagate_fracture
from porepy.viz import plot_grid, exporter
from porepy.utils import setmembership as sm
from porepy.utils import tags

logger = logging.getLogger(__name__)
#------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

#------------------------------------------------------------------------------#

    def test_propagation_visulization_2d(self):
        """
        Setup intended for visual inspection before and after fracture
        propagation of a single 1d fracture in 2d.
        """
        f = np.array([[0, 1],
                      [.5, .5]])
        gb = meshing.cart_grid([f], [4, 2], physdims=[2, 1])
        for g, d in gb:
            d['cell_tags'] = np.ones(g.num_cells) * g.dim
        plot_grid.plot_grid(gb, 'cell_tags', info='f')

        face_2d = np.array([16])
        new_apertures = 0.1*np.ones(face_2d.size)
        gh = gb.grids_of_dimension(2)[0]
        gl = gb.grids_of_dimension(1)[0]
        propagate_fracture.propagate_fracture(gb, gh, gl,
                                              faces_h=face_2d,
                                              apertures_l=new_apertures)
        for g, d in gb:
            d['cell_tags'] = np.ones(g.num_cells) * g.dim
        plot_grid.plot_grid(gb, 'cell_tags', info='f')
        return gh, gl

    def test_propagation_visualization_3d(self):
        """
        Setup intended for visual inspection before and after fracture
        propagation of a single 2d fracture in 3d.
        """
        f = np.array([[0., 1, 1, 0],
                      [.25, .25, .75, .75],
                      [.5, .5, .5, .5]])
        gb = meshing.cart_grid([f], [4, 4, 2], physdims=[2, 1, 1])

        e = exporter.Exporter(gb, 'grid_before', 'test_propagation_3d')
        e.write_vtk()
        face_3d = np.array([102, 106, 110])
        new_apertures = 0.1*np.ones(face_3d.size)
        gh = gb.grids_of_dimension(3)[0]
        gl = gb.grids_of_dimension(2)[0]
        propagate_fracture.propagate_fracture(gb, gh, gl,
                                              faces_h=face_3d,
                                              apertures_l=new_apertures)
        e_after = exporter.Exporter(gb, 'grid_after', 'test_propagation_3d')
        e_after.write_vtk()

        return gh, gl

    def test_equivalence_2d(self):
        """
        This central test checks that certain geometry, connectivity and tags
        are independent of whether the fracture is
        1 premade in its final form
        2-3 grown in a single step (from two different inital fractures)
        4 grown in two steps
        """
        f_final = np.array([[0.0, 1.5],
                            [0.5, 0.5]])
        f_small_initial = np.array([[0.0, 0.5],
                                    [0.5, 0.5]])
        f_large_initial = np.array([[0.0, 1.0],
                                    [0.5, 0.5]])

        gb_1 = meshing.cart_grid([f_final], [4, 2], physdims=[2, 1])
        gb_2 = meshing.cart_grid([f_small_initial], [4, 2], physdims=[2, 1])
        gb_3 = meshing.cart_grid([f_large_initial], [4, 2], physdims=[2, 1])
        gb_4 = meshing.cart_grid([f_small_initial], [4, 2], physdims=[2, 1])
        # Split
        propagate_simple(gb_2, [15, 16])
        propagate_simple(gb_3, [16])
        propagate_simple(gb_4, [15])
        propagate_simple(gb_4, [16])
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

        f_final = np.array([[0.0, .75, .75, 0.0],
                            [0.0, 0.0, 0.5, 0.5],
                            [0.5, 0.5, 0.5, 0.5]])

        f_small = np.array([[0.0, .25, .25, 0.0],
                            [0.0, 0.0, 0.5, 0.5],
                            [0.5, 0.5, 0.5, 0.5]])
        f_large = np.array([[0.0, 0.5, 0.5, 0.0],
                            [0.0, 0.0, 0.5, 0.5],
                            [0.5, 0.5, 0.5, 0.5]])

        gb_1 = meshing.cart_grid([f_final], [4, 2, 2], physdims=[1, 1, 1])
        gb_2 = meshing.cart_grid([f_small], [4, 2, 2], physdims=[1, 1, 1])
        gb_3 = meshing.cart_grid([f_large], [4, 2, 2], physdims=[1, 1, 1])
        gb_4 = meshing.cart_grid([f_small], [4, 2, 2], physdims=[1, 1, 1])
        # Split
        propagate_simple(gb_2, [53, 54])
        propagate_simple(gb_3, [54])
        propagate_simple(gb_4, [53])
        propagate_simple(gb_4, [54])
        # Check that the four grid buckets are equivalent
        check_equivalent_buckets([gb_1, gb_2, gb_3, gb_4])


def propagate_simple(gb, faces):
    """
    Wrapper for the fracture propagation in a bucket containing a single
    fracture along specified higher-dimensional faces.
    """
    dim_h = gb.dim_max()
    faces = np.array(faces)
    propagate_fracture.propagate_fracture(gb,
                                          gb.grids_of_dimension(dim_h)[0],
                                          gb.grids_of_dimension(dim_h - 1)[0],
                                          faces,
                                          0.1*np.ones(faces.size))


def check_equivalent_buckets(buckets):
    """
    Checks agreement between number of cells, faces and nodes, their
    coordinates and the connectivity matrices cell_faces and face_nodes. Also
    checks the face tags.
    TODO Check on face_cells between dimensions.
    """
    dim_h = buckets[0].dim_max()
    n = len(buckets)
    for d in range(dim_h-1, dim_h+1):
        n_cells, n_faces, n_nodes = np.empty(0), np.empty(0), np.empty(0)
        nodes, face_centers, cell_centers = [], [], []
        cell_faces, face_nodes = [], []
        for bucket in buckets:
            g = bucket.grids_of_dimension(d)[0]
            n_cells = np.append(n_cells, g.num_cells)
            n_faces = np.append(n_faces, g.num_faces)
            n_nodes = np.append(n_nodes, g.num_nodes)
            cell_faces.append(g.cell_faces)
            face_nodes.append(g.face_nodes)
            cell_centers.append(g.cell_centers)
            face_centers.append(g.face_centers)
            nodes.append(g.nodes)
        # Check that all buckets have the same number of cells, faces and nodes
        assert np.unique(n_cells).size == 1
        assert np.unique(n_faces).size == 1
        assert np.unique(n_nodes).size == 1
        # Check that the coordinates agree
        for i in range(1, n):
            assert np.all(sm.ismember_rows(cell_centers[0],
                                           cell_centers[i])[0])
            assert np.all(sm.ismember_rows(face_centers[0],
                                           face_centers[i])[0])
            assert np.all(sm.ismember_rows(nodes[0], nodes[i])[0])
        # Now we know all nodes, faces and cells are in all grids, we map them
        # to prepare cell_faces and face_nodes comparison
        g_0 = buckets[0].grids_of_dimension(d)[0]
        for i in range(1, n):
            bucket = buckets[i]
            g = bucket.grids_of_dimension(d)[0]
            cell_map, face_map, node_map = make_maps(g_0, g, bucket.dim_max())
            mapped_cf = g.cell_faces[face_map][:, cell_map]
            mapped_fn = g.face_nodes[node_map][:, face_map]

            assert np.sum(np.abs(g_0.cell_faces) != np.abs(mapped_cf)) == 0
            assert np.sum(np.abs(g_0.face_nodes) != np.abs(mapped_fn)) == 0

            # Also loop on the standard face tags to check that they are
            # identical between the buckets.
            tag_keys = tags.standard_face_tags()
            for key in tag_keys:
                assert np.all(np.isclose(g_0.tags[key], g.tags[key][face_map]))


def make_maps(g0, g1, n_digits=8, offset=0.11):
    """
    Given two grid with the same nodes, faces and cells, the mappings between
    these entities are concstructed. Handles non-unique nodes and faces on next
    to fractures by exploiting neighbour information.
    Builds maps from g1 to g0, so g1.x[x_map]=g0.x, e.g.
    g1.tags[some_key][face_map] = g0.tags[some_key].
    g0 Reference grid
    g1 Other grid
    n_digits Tolerance in rounding before coordinate comparison
    offset: Weight determining how far the fracture neighbour nodes and faces
    are shifted (normally away from fracture) to ensure unique coordinates.
    """
    cell_map = sm.ismember_rows(np.around(g0.cell_centers, n_digits),
                                np.around(g1.cell_centers, n_digits),
                                sort=False, simple_version=True)[1]
    # Make face_centers unique by dragging them slightly away from the fracture

    fc0 = g0.face_centers.copy()
    fc1 = g1.face_centers.copy()
    n0 = g0.nodes.copy()
    n1 = g1.nodes.copy()
    fi0 = g0.tags['fracture_faces']
    if np.any(fi0):
        fi1 = g1.tags['fracture_faces']
        d0 = np.reshape(np.tile(g0.cell_faces[fi0, :].data, 3), (3, sum(fi0)))
        fn0 = g0.face_normals[:, fi0] * d0
        d1 = np.reshape(np.tile(g1.cell_faces[fi1, :].data, 3), (3, sum(fi1)))
        fn1 = g1.face_normals[:, fi1] * d1
        fc0[:, fi0] += fn0*offset
        fc1[:, fi1] += fn1*offset
        (ni0, fid0) = g0.face_nodes[:, fi0].nonzero()
        (ni1, fid1) = g1.face_nodes[:, fi1].nonzero()
        un, inv = np.unique(ni0, return_inverse=True)
        for i, node in enumerate(un):
            n0[:, node] += offset * np.mean(fn0[:, fid0[inv == i]], axis=1)
        un, inv = np.unique(ni1, return_inverse=True)
        for i, node in enumerate(un):
            n1[:, node] += offset * np.mean(fn1[:, fid1[inv == i]], axis=1)

    face_map = sm.ismember_rows(np.around(fc0, n_digits),
                                np.around(fc1, n_digits),
                                sort=False, simple_version=True)[1]

    node_map = sm.ismember_rows(np.around(n0, n_digits),
                                np.around(n1, n_digits),
                                sort=False, simple_version=True)[1]
    return cell_map, face_map, node_map


if __name__ == '__main__':
    t1 = time.time()
    BasicsTest().test_equivalence_2d()
    print(time.time()-t1)
    t1 = time.time()
    BasicsTest().test_equivalence_3d()
    print(time.time()-t1)
