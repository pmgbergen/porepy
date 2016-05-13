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
    """Representation of grid formed by a tensor product of line point
    distributions.

    See documentation of Grid for further details
    """

    def __init__(self, x, y, z=None, name=None):
        """
        Constructor for 2D or 3D tensor grid

        The resulting grid is 2D or 3D, depending of the number of
        coordinate lines are provided

        Parameters
            x (np.ndarray): Node coordinates in x-direction
            y (np.ndarray): Node coordinates in y-direction
            z (np.ndarray): Node coordinates in z-direction. Defaults to
                None, in which case the grid is 2D.
            name (str): Name of grid, passed to super constructor
        """
        if name is None:
            name = 'TensorGrid'

        if z is None:
            nodes, face_nodes, cell_faces = self._create_2d_grid(x, y)
            self.cart_dims = np.array([x.size, y.size])
            super(TensorGrid, self).__init__(2, nodes, face_nodes,
                                             cell_faces, name)
        else:
            nodes, face_nodes, cell_faces = self._create_3d_grid(x, y, z)
            self.cart_dims = np.array([x.size, y.size, z.size])
            super(TensorGrid, self).__init__(3, nodes, face_nodes,
                                             cell_faces, name)

    def _create_2d_grid(self, x, y):
        """
        Compute grid topology for 2D grids.

        This is really a part of the constructor, but put it here to improve
        readability. Not sure if that is the right choice..

        """

        sx = x.size - 1
        sy = y.size - 1

        num_cells = sx * sy
        num_nodes = (sx + 1) * (sy + 1)
        num_faces_x = (sx + 1) * sy
        num_faces_y = sx * (sy + 1)
        num_faces = num_faces_x + num_faces_y

        num_cells = num_cells
        num_faces = num_faces
        num_nodes = num_nodes

        x_coord, y_coord = sp.meshgrid(x, y)

        nodes = np.vstack((x_coord.flatten(), y_coord.flatten()))

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
        face_nodes = sps.csc_matrix((data, face_nodes, indptr),
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
        cell_faces = sps.csc_matrix((data, cell_faces, indptr),
                                    shape=(num_faces, num_cells))
        return nodes, face_nodes, cell_faces

    def _create_3d_grid(self, x, y, z):

        sx = x.size - 1
        sy = y.size - 1
        sz = z.size - 1

        num_cells = sx * sy * sz
        num_nodes = (sx + 1) * (sy + 1) * (sz + 1)
        num_faces_x = (sx + 1) * sy * sz
        num_faces_y = sx * (sy + 1) * sz
        num_faces_z = sx * sy * (sz + 1)
        num_faces = num_faces_x + num_faces_y + num_faces_z

        num_cells = num_cells
        num_faces = num_faces
        num_nodes = num_nodes

        x_coord, y_coord, z_coord = np.meshgrid(x, y, z)
        # This rearangement turned out to work. Not the first thing I tried..
        x_coord = np.swapaxes(x_coord, 1, 0).ravel(order='F')
        y_coord = np.swapaxes(y_coord, 1, 0).ravel(order='F')
        z_coord = np.swapaxes(z_coord, 1, 0).ravel(order='F')

        nodes = np.vstack((x_coord, y_coord, z_coord))

        # Face nodes
        node_array = np.arange(num_nodes).reshape(sx + 1, sy + 1, sz + 1,
                                                  order='F')

        # Define face-node relations for all x-faces.
        # The code here is a bit different from the corresponding part in
        # 2d, I did learn some tricks in python the past month
        fn1 = node_array[:, :-1, :-1].ravel(order='F')
        fn2 = node_array[:, 1:, :-1].ravel(order='F')
        fn3 = node_array[:, 1:, 1:].ravel(order='F')
        fn4 = node_array[:, :-1, 1:].ravel(order='F')
        face_nodes_x = np.vstack((fn1, fn2, fn3, fn4)).ravel(order='F')

        # Define face-node relations for all y-faces
        fn1 = node_array[:-1:, :, :-1].ravel(order='F')
        fn2 = node_array[:-1, :, 1:].ravel(order='F')
        fn3 = node_array[1:, :, 1:].ravel(order='F')
        fn4 = node_array[1:, :, :-1].ravel(order='F')
        face_nodes_y = np.vstack((fn1, fn2, fn3, fn4)).ravel(order='F')

        # Define face-node relations for all y-faces
        fn1 = node_array[:-1:, :-1, :].ravel(order='F')
        fn2 = node_array[1:, :-1, :].ravel(order='F')
        fn3 = node_array[1:, 1:, :].ravel(order='F')
        fn4 = node_array[:-1, 1:, :].ravel(order='F')
        face_nodes_z = np.vstack((fn1, fn2, fn3, fn4)).ravel(order='F')

        # Test
        assert face_nodes_x.size == face_nodes_y.size
        assert face_nodes_x.size == face_nodes_z.size

        num_nodes_per_face = 4
        indptr = np.append(np.arange(0, num_nodes_per_face * num_faces,
                                     num_nodes_per_face),
                           num_nodes_per_face * num_faces)
        face_nodes = np.hstack((face_nodes_x, face_nodes_y, face_nodes_z))
        data = np.ones(face_nodes.shape, dtype=bool)
        face_nodes = sps.csc_matrix((data, face_nodes, indptr),
                                    shape=(num_nodes, num_faces))

        # Cell faces
        face_x = np.arange(num_faces_x).reshape(sx + 1, sy, sz, order='F')
        face_y = num_faces_x + np.arange(num_faces_y).reshape(sx, sy + 1,
                                                              sx, order='F')
        face_z = num_faces_x + num_faces_y + \
                 np.arange(num_faces_y).reshape(sx, sy , sz + 1, order='F')

        face_west = face_x[:-1, :, :].ravel(order='F')
        face_east = face_x[1:, :, :].ravel(order='F')
        face_south = face_y[:, :-1, :].ravel(order='F')
        face_north = face_y[:, 1:, :].ravel(order='F')
        face_top = face_z[:, :, :-1].ravel(order='F')
        face_bottom = face_z[:, :, 1:].ravel(order='F')

        cell_faces = np.vstack((face_west, face_east,
                                face_south, face_north, face_top,
                                face_bottom)).ravel(order='F')

        num_faces_per_cell = 6
        indptr = np.append(np.arange(0, num_faces_per_cell * num_cells,
                                     num_faces_per_cell),
                           num_faces_per_cell * num_cells)
        data = np.vstack((-np.ones(num_cells), np.ones(num_cells),
                          -np.ones(num_cells), np.ones(num_cells),
                          -np.ones(num_cells), np.ones(num_cells))
                         ).ravel(order='F')
        cell_faces = sps.csc_matrix((data, cell_faces, indptr),
                                    shape=(num_faces, num_cells))
        return nodes, face_nodes, cell_faces


class CartGrid(TensorGrid):
    """Representation of a 2D or 3D Cartesian grid.

    See main Grid class for further explanation.
    """

    def __init__(self, nx, physdims=None):
        """
        Constructor for Cartesian grid

        Parameters
        ----------
        nx (np.ndarray): Number of cells in each direction. Should be 2D or 3D
        physdims (np.ndarray): Physical dimensions in each direction.
            Defaults to same as nx, that is, cells of unit size.
        """

        if physdims is None:
            physdims = nx

        dims = nx.shape

        # Create point distribution, and then leave construction to
        # TensorGrid constructor
        if dims[0] == 1:
            # This may actually work, but hasn't been tried
            raise NotImplementedError('only 2D and 3D supported for now')
        elif dims[0] == 2:
            x = np.linspace(0, physdims[0], nx[0]+1)
            y = np.linspace(0, physdims[1], nx[1]+1)
            super(self.__class__, self).__init__(x, y)
        elif dims[0] == 3:
            x = np.linspace(0, physdims[0], nx[0] + 1)
            y = np.linspace(0, physdims[1], nx[1] + 1)
            z = np.linspace(0, physdims[2], nx[2] + 1)
            super(self.__class__, self).__init__(x, y, z)
        else:
            raise ValueError('Cartesian grid only implemented for up to three \
            dimensions')
