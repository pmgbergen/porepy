#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module contains hard coded grid geometries that can be used for several
tests.
"""
import numpy as np
import scipy.sparse as sps

import porepy as pp


class SimplexGrid2dDomainOneImmersedFracture:
    """ Grid of 18 cells, that contains a single fully immersed fracture,
        initially aligned with the y-axis.

    By tuning parameters, the fracture can be rotated, nodes in the 2d grids
    on the fracture surface can be perturbed, and normal vectors on the fracture
    surface can be flipped.
    """

    def grid_2d(self, flip_normals=False, pert_node=False, rotate_fracture=False):
        """ flip_normals: Some, but not all, faces in 2d grid on fracture
                surface are rotated. Divergence operator is modified accordingly.
                The solution should be invariant under this change.
            perturb_node: One node in 2d grid on the fracture surface is shifted.
                This breaks symmetry of the grid across the fracture.
                Should not be combined with rotate_fracture.
            rotate_fracture: Fracture is rotated from aligned with the y-axis
                to a slightly tilted position. Should not be combined with
                perturb_node
        """

        nodes = np.array(
            [
                [0, 0, 0],  # 0
                [1, 0, 0],  # 1
                [1, 0.5, 0],  # 2
                [1, 1, 0],  # 3
                [0, 1, 0],  # 4
                [0, 0.5, 0],  # 5
                [0.75, 0.375, 0],  # 6
                [0.75, 0.75, 0],  # 7
                [0.25, 0.75, 0],  # 8
                [0.25, 0.25, 0],  # 9
                [0.5, 0.25, 0],  # 10
                [0.5, 0.5, 0],  # 11
                [0.5, 0.75, 0],  # 12
                [0.5, 0.5, 0],  # 13
            ]
        ).T

        if pert_node:
            if rotate_fracture:
                raise ValueError("Incompatible options to grid construction")
            nodes[1, 13] = 0.6
        if rotate_fracture:
            nodes[0, 10] = 0.4
            nodes[0, 12] = 0.6

        fn = np.array(  # Faces always go from node of low to high index
            [
                [0, 1],  # 0
                [1, 2],  # 1
                [2, 3],  # 2
                [3, 4],  # 3
                [4, 5],  # 4
                [0, 5],  # 5
                [0, 9],  # 6
                [0, 10],  # 7
                [9, 10],  # 8
                [5, 9],  # 9
                [9, 13],  # 10
                [8, 13],  # 11
                [5, 8],  # 12
                [8, 9],  # 13
                [4, 8],  # 14
                [8, 12],  # 15
                [4, 12],  # 16
                [1, 10],  # 17
                [1, 6],  # 18
                [6, 10],  # 19
                [6, 11],  # 20
                [6, 7],  # 21
                [7, 11],  # 22
                [2, 6],  # 23
                [2, 7],  # 24
                [7, 12],  # 25
                [3, 7],  # 26
                [3, 12],  # 27
                [10, 13],  # 28
                [12, 13],  # 29
                [10, 11],  # 30
                [11, 12],  # 31
            ]
        ).T
        cols = np.tile(np.arange(fn.shape[1]), (fn.shape[0], 1)).ravel("F")
        face_nodes = sps.csc_matrix((np.ones_like(cols), (fn.ravel("F"), cols)))

        cf = np.array(
            [
                [0, 17, 7],  # 0
                [6, 7, 8],  # 1
                [5, 6, 9],  # 2
                [8, 28, 10],  # 3
                [10, 11, 13],  # 4
                [9, 13, 12],  # 5
                [11, 29, 15],  # 6
                [4, 12, 14],  # 7
                [14, 15, 16],  # 8
                [3, 16, 27],  # 9
                [25, 31, 22],  # 10
                [25, 26, 27],  # 11
                [2, 26, 24],  # 12
                [21, 23, 24],  # 13
                [20, 21, 22],  # 14
                [19, 20, 30],  # 15
                [17, 18, 19],  # 16
                [1, 23, 18],  # 17
            ]
        ).T
        cols = np.tile(np.arange(cf.shape[1]), (cf.shape[0], 1)).ravel("F")
        data = np.array(
            [
                [1, 1, -1],  # 0
                [-1, 1, -1],  # 1
                [-1, 1, -1],  # 2
                [1, 1, -1],  # 3
                [1, -1, 1],  # 4
                [1, -1, -1],  # 5
                [1, -1, -1],  # 6
                [1, 1, -1],  # 7
                [1, 1, -1],  # 8
                [1, 1, -1],  # 9
                [1, -1, -1],  # 10
                [-1, -1, 1],  # 11
                [1, 1, -1],  # 12
                [-1, -1, 1],  # 13
                [-1, 1, 1],  # 14
                [-1, 1, -1],  # 15
                [-1, 1, 1],  # 16
                [1, 1, -1],  # 17
            ]
        ).ravel()

        cell_faces = sps.csc_matrix((data, (cf.ravel("F"), cols)))
        g = pp.Grid(2, nodes, face_nodes, cell_faces, "TriangleGrid")
        g.compute_geometry()
        fracture_faces = [28, 29, 30, 31]
        g.tags["fracture_faces"][fracture_faces] = 1
        g.global_point_ind = np.arange(nodes.shape[1])

        if flip_normals:
            g.face_normals[:, fracture_faces] *= -1
            g.cell_faces[fracture_faces, [3, 6, 15, 10]] *= -1
            g.face_normals[:, 28] *= -1
            g.cell_faces[28, 3] *= -1

        return g

    def grid_1d(self, num_pts=3, rotate_fracture=False):
        g = pp.TensorGrid(np.arange(num_pts))
        if rotate_fracture:
            g.nodes = np.vstack(
                (
                    np.linspace(0.4, 0.6, num_pts),
                    np.linspace(0.25, 0.75, num_pts),
                    np.zeros(num_pts),
                )
            )
        else:
            g.nodes = np.vstack(
                (
                    0.5 * np.ones(num_pts),
                    np.linspace(0.25, 0.75, num_pts),
                    np.zeros(num_pts),
                )
            )
        g.compute_geometry()
        g.global_point_ind = np.arange(g.num_nodes)
        return g

    def generate_grid(
        self, num_1d=3, pert_node=False, flip_normals=False, rotate_fracture=False
    ):
        g2 = self.grid_2d(
            flip_normals=flip_normals,
            pert_node=pert_node,
            rotate_fracture=rotate_fracture,
        )
        g1 = self.grid_1d(num_pts=num_1d, rotate_fracture=rotate_fracture)
        gb = pp.meshing._assemble_in_bucket([[g2], [g1]])

        gb.add_edge_props("face_cells")
        for _, d in gb.edges():
            a = np.zeros((g2.num_faces, g1.num_cells))
            a[28, 0] = 1
            a[29, 1] = 1
            a[30, 0] = 1
            a[31, 1] = 1
            d["face_cells"] = sps.csc_matrix(a.T)
        pp.meshing.create_mortar_grids(gb)

        g_new_2d = self.grid_2d(
            flip_normals=flip_normals,
            pert_node=pert_node,
            rotate_fracture=rotate_fracture,
        )
        g_new_1d = self.grid_1d(num_pts=num_1d, rotate_fracture=rotate_fracture)

        gb.replace_grids(g_map={g2: g_new_2d, g1: g_new_1d})

        gb.assign_node_ordering()

        return gb
