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
    """
    Parent class for all grids.

    The grid stores topological information, as well as geometric
    information (after a call to self.compute_geometry().

    As of yet, there is no structure for tags (face or cell) is the grid.
    This will be introduced later.

    Attributes:
        Comes in two classes. Topologogical information, defined at
        construction time:

        dim (int): dimension. Should be 2 or 3
        nodes (np.ndarray): node coordinates. size: dim x num_nodes
        face_nodes (sps.csc-matrix): Face-node relationships. Matrix size:
            num_faces x num_cells.
        cell_faces (sps.csc-matrix): Cell-face relationships. Matrix size:
            num_faces x num_cells. Matrix elements have value +-1, where +
            corresponds to the face normal vector being outwards.
        name (str): Placeholder field to give information on the grid. Will
            be changed to something meaningful in the future
        num_nodes (int): Number of nodes in the grid
        num_faces (int): Number of faces in the grid
        num_cells (int): Number of cells in the grid

        ---
        Geometric information, available after compute_geometry() has been
        called on the object:

        face_areas (np.ndarray): Areas of all faces
        face_centers (np.ndarray): Centers of all faces. Dimensions dim x
            num_faces
        face_normals (np.ndarray): Normal vectors of all faces. Dimensions
            dim x num_faces. See also cell_faces.
        cell_centers (np.ndarray): Centers of all cells. Dimensions dim x
            num_cells
        cell_volumes (np.ndarray): Volumes of all cells

    """

    def __init__(self, dim, nodes, face_nodes, cell_faces, name):
        """Initialize the grid

        See class documentation for further description of parameters.

        Parameters
        ----------
        dim (int): grid dimension
        nodes (np.ndarray): node coordinates.
        face_nodes (sps.csc_matrix): Face-node relations.
        cell_faces (sps.csc_matrix): Cell-face relations
        name (str): Name of grid
        """
        self.dim = dim
        self.nodes = nodes
        self.cell_faces = cell_faces
        self.face_nodes = face_nodes
        self.name = name

        # Infer bookkeeping from size of parameters
        self.num_nodes = nodes.shape[1]
        self.num_faces = face_nodes.shape[1]
        self.num_cells = cell_faces.shape[1]

    def compute_geometry(self):
        """Compute geometric quantities for the grid.

        This method initializes class variables describing the grid
        geometry, see class documentation for details.

        The method could have been called from the constructor, however,
        in cases where the grid is modified after the initial construction (
        say, grid refinement), this may lead to costly, unnecessary
        computations.
        """
        if self.dim == 2:
            self.__compute_geometry_2d()
        else:
            raise NotImplementedError('3D not handled yet')

    def __compute_geometry_2d(self):
        "Compute 2D geometry, with method motivated by similar MRST function"

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

        cell_faces, cellno = self.cell_faces.nonzero()

        num_cell_faces = np.bincount(cellno)

        cx = np.bincount(cellno, weights=self.face_centers[0, cell_faces])
        cy = np.bincount(cellno, weights=self.face_centers[1, cell_faces])
        cell_centers = np.vstack((cx, cy)) / num_cell_faces

        a = xe1[:, cell_faces] - cell_centers[:, cellno]
        b = xe2[:, cell_faces] - cell_centers[:, cellno]

        sub_volumes = 0.5 * np.abs(a[0] * b[1] - a[1] * b[0])
        self.cell_volumes = np.bincount(cellno, weights=sub_volumes)
        
        sub_centroids = (cell_centers[:, cellno] + 2 *
                         self.face_centers[:, cell_faces]) / 3

        ccx = np.bincount(cellno, weights=sub_volumes * sub_centroids[0])
        ccy = np.bincount(cellno, weights=sub_volumes * sub_centroids[1])

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
        self.face_normals[:, flip] *= -1

    def cell_nodes(self):
        mat = (self.face_nodes * np.abs(self.cell_faces) *
               sps.eye(self.num_cells)) > 0
        return mat

    def num_cell_nodes(self):
        return self.cell_nodes().sum(axis=0).A.ravel(1)
