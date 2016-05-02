# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 21:23:30 2016

@author: keile
"""
import numpy as np
import scipy as sp
import scipy.sparse as sps

from core.grids.grid import Grid, GridType


class TensorGrid(Grid):

    def __init__(self, x, y, z=None):

        if not hasattr(self, 'type'):
            self.type = GridType.tensor_2D

        if z is None:
            self._create_2d_grid(x, y)
        else:
            raise NotImplementedError('Only 2D supported for now')

    def _create_2d_grid(self, x, y):
        self.dim = 2

        sx = x.size - 1
        sy = y.size - 1

        num_cells = sx * sy
        num_nodes = (sx + 1) * (sy + 1)
        num_faces_x = (sx + 1) * sy
        num_faces_y = sx * (sy + 1)
        num_faces = num_faces_x + num_faces_y

        self.Nc = num_cells
        self.Nf = num_faces
        self.Nn = num_nodes

        x_coord, y_coord = sp.meshgrid(x, y)

        self.nodes = np.vstack((x_coord.flatten(), y_coord.flatten()))

        # Face nodes
        node_array = np.arange(0, num_nodes).reshape(sy+1, sx+1)
        fn1 = node_array[:-1, ::].ravel()
        fn2 = node_array[1:, ::].ravel()
        face_nodes_x = np.vstack((fn1, fn2)).ravel(order='F')

        fn1 = node_array[::, :-1].ravel(order='C')
        fn2 = node_array[::, 1:].ravel(order='C')
        face_nodes_y = np.vstack((fn1, fn2)).ravel(order='F')

        num_nodes_per_face = 2
        indptr = np.append(np.arange(0, num_nodes_per_face*num_faces,
                                     num_nodes_per_face),
                           num_nodes_per_face * num_faces)
        face_nodes = np.hstack((face_nodes_x, face_nodes_y))
        data = np.ones(face_nodes.shape, dtype=bool)
        self.faceNodes = sps.csc_matrix((data, face_nodes, indptr),
                                        shape=(num_nodes, num_faces))

        # Cell faces
        face_x = np.arange(num_faces_x).reshape(sy, sx+1)
        face_y = num_faces_x + np.arange(num_faces_y).reshape(sy+1, sx)

        face_west = face_x[::, :-1].ravel(order='C')
        face_east = face_x[::, 1:].ravel(order='C')
        face_south = face_y[:-1, ::].ravel(order='C')
        face_north = face_y[1:, ::].ravel(order='C')

        cell_faces = np.vstack((face_west, face_east,
                                face_south, face_north)).ravel(order='F')

        num_faces_per_cell = 4
        indptr = np.append(np.arange(0, num_faces_per_cell*num_cells,
                                     num_faces_per_cell),
                           num_faces_per_cell * num_cells)
        data = np.vstack((-np.ones(face_west.size), np.ones(face_east.size),
                          -np.ones(face_south.size), np.ones(
            face_north.size))).ravel(order='F')
        self.cellFaces = sps.csc_matrix((data, cell_faces, indptr),
                                        shape=(num_faces, num_cells))


class CartGrid(TensorGrid):
    def __init__(self, nx, physdims=None):

        self.type = GridType.cartesian_2D

        if physdims is None:
            physdims = nx

        dims = nx.shape
        if dims[0] == 1:
            raise NotImplementedError('only 2D supported for now')
        elif dims[0] == 2:
            x = np.linspace(0, physdims[0], nx[0]+1)
            y = np.linspace(0, physdims[1], nx[1]+1)
            super(self.__class__, self).__init__(x, y)
        elif dims[0] == 3:
            raise NotImplementedError('only 2D supported for now')
        else:
            raise ValueError('Cartesian grid only implemented for up to three \
            dimensions')
