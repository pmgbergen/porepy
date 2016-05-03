# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:55:43 2016

@author: keile
"""

import numpy as np
from enum import Enum
from scipy import sparse as sps


class GridType(Enum):
    """
    Enumeration to define types of grids. Not quite sure what I want to use
    them for; right now the primary motivation is to test which type of grid
    this is.
    Possible future usage could assign dimension to the fields, etc.
    """
    triangle = 1
    cartesian_2D = 2
    tensor_2D = 3


class Grid(object):

    def __init__(self, dim, nodes, face_nodes, cell_faces, name):
        self.dim = dim
        self.nodes = nodes
        self.cell_faces = cell_faces
        self.face_nodes = face_nodes
        self.name = name

        self.num_nodes = nodes.shape[1]
        self.num_faces = face_nodes.shape[1]
        self.num_cells = cell_faces.shape[1]

    def compute_geometry(self):
        if self.dim == 2:
            self.__compute_geometry_2D()
        else:
            raise NotImplementedError('3D not handled yet')

    def __compute_geometry_2D(self):

        xn = self.nodes

        fn = self.face_nodes.indices
        edge1 = fn[::2]
        edge2 = fn[1::2]

        xe1 = xn[:, edge1]
        xe2 = xn[:, edge2]

        edge_length_x = xe2[0] - xe1[0]
        edge_length_y = xe2[1] - xe1[1]
        self.face_areas = np.sqrt(np.power(edge_length_x, 2) +
                                  np.power(edge_length_y, 2))
        self.face_centers = 0.5 * (xe1 + xe2)
        self.face_normals = np.vstack((edge_length_y, -edge_length_x))

        cellFaces, cellno = self.cell_faces.nonzero()

        nCellFaces = np.bincount(cellno)

        cx = np.bincount(cellno, weights=self.face_centers[0, cellFaces])
        cy = np.bincount(cellno, weights=self.face_centers[1, cellFaces])
        cCenters = np.vstack((cx, cy)) / nCellFaces

        a = xe1[:, cellFaces] - cCenters[:, cellno]
        b = xe2[:, cellFaces] - cCenters[:, cellno]

        subVol = 0.5 * np.abs(a[0] * b[1] - a[1] * b[0])
        self.cell_volumes = np.bincount(cellno, weights=subVol)
        
        subCentroid = (cCenters[:, cellno] + 2 *
                       self.face_centers[:, cellFaces]) / 3

        ccx = np.bincount(cellno, weights=subVol * subCentroid[0])
        ccy = np.bincount(cellno, weights=subVol * subCentroid[1])

        self.cell_centers = np.vstack((ccx, ccy)) / self.cell_volumes

        # Ensure that normal vector direction corresponds with sign convention
        # in self.cellFaces
        def nrm(u):
            return np.sqrt(u[0]*u[0] + u[1]*u[1])

        [fi, ci, val] = sps.find(self.cell_faces)
        _, idx = np.unique(fi, return_index=True)
        sgn = val[idx]
        fc = self.face_centers[:, fi[idx]]
        cc = self.cell_centers[:, ci[idx]]
        v = fc - cc
        # Prolong the vector from cell to face center in the direction of the
        # normal vector. If the prolonged vector is shorter, the normal should
        # flipped
        vn = v + nrm(v) * self.face_normals[:, fi[idx]] * 0.001
        flip = np.logical_or(np.logical_and(nrm(v) > nrm(vn), sgn > 0),
                             np.logical_and(nrm(v) < nrm(vn), sgn < 0))
        self.face_normals[:, flip] *= -1  # Is it correct not to have fi here?

    def cell_nodes(self):
        mat = (self.face_nodes * np.abs(self.cell_faces) *
               sps.eye(self.num_cells)) > 0
        return mat

    def num_cell_nodes(self):
        return self.cell_nodes().sum(axis=0).A.ravel(1)
