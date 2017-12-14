""" Module containing the parent class for all grids.

See documentation of the Grid class for further details.

Acknowledgements:
    The data structure for the grid is inspired by that used in the Matlab
    Reservoir Simulation Toolbox (MRST) developed by SINTEF ICT, see
    www.sintef.no/projectweb/mrst/ . Some of the methods, in particular
    compute_geometry() and its subfunctions is to a large degree translations
    of the corresponding functions in MRST.

"""
from __future__ import division
import numpy as np
import itertools
from enum import Enum
from scipy import sparse as sps

from porepy.utils import matrix_compression, mcolon

from porepy.utils import comp_geom as cg

class MortarGrid(object):
    """
    Parent class for all grids.

    The grid stores topological information, as well as geometric
    information (after a call to self.compute_geometry().

    As of yet, there is no structure for tags (face or cell) is the grid.
    This will be introduced later.

    Attributes:
        Comes in two classes. Topologogical information, defined at
        construction time:

        dim (int): dimension. Should be 0 or 1 or 2 or 3
        nodes (np.ndarray): node coordinates. size: dim x num_nodes
        face_nodes (sps.csc-matrix): Face-node relationships. Matrix size:
            num_faces x num_cells. To use compute_geometry() later, he field
            face_nodes.indices should store the nodes of each face sorted.
            For more information, see information on compute_geometry()
            below.
        cell_faces (sps.csc-matrix): Cell-face relationships. Matrix size:
            num_faces x num_cells. Matrix elements have value +-1, where +
            corresponds to the face normal vector being outwards.
        name (list): Information on the formation of the grid, such as the
            constructor, computations of geometry etc.
        num_nodes (int): Number of nodes in the grid
        num_faces (int): Number of faces in the grid
        num_cells (int): Number of cells in the grid

        ---
        compute_geometry():
        Assumes the nodes of each face is ordered according to the right
        hand rule.
        face_nodes.indices[face_nodes.indptr[i]:face_nodes.indptr[i+1]]
        are the nodes of face i, which should be ordered counter-clockwise.
        By counter-clockwise we mean as seen from cell cell_faces[i,:]==1.
        Equivalently the nodes will be clockwise as seen from cell
        cell_faces[i,:] == -1. Note that operations on the face_nodes matrix
        (such as converting it to a csr-matrix) may change the ordering of
        the nodes (face_nodes.indices), which will break compute_geometry().
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

    def __init__(self, dim, nodes, cell_nodes, face_cells, name=''):
        """Initialize the grid

        See class documentation for further description of parameters.

        Parameters
        ----------
        dim (int): grid dimension
        nodes (np.ndarray): node coordinates.
        cell_nodes (sps.csc_matrix): Cell-node relations
        name (str): Name of grid
        """
        assert dim >= 0 and dim < 3
        self.dim = dim

        if isinstance(name, list):
            self.name = name
        else:
            self.name = [name]

        self.update_geometry(nodes, cell_nodes)

        # face_cells mapping from the higher dimensional grid to the mortar grid
        # also here we assume that, in the beginning the mortar grids are equal
        # to the co-dimensional grid
        cells, faces, data = sps.find(face_cells)
        cells[faces > np.median(faces)] += cell_nodes.shape[1]

        shape = (self.num_cells, face_cells.shape[1])
        self.high_to_mortar = sps.csc_matrix((data.astype(np.float),
                                             (cells, faces)), shape=shape)

        # cell_cells mapping from the mortar grid to the lower dimensional grid
        self.mortar_to_low = sps.identity(self.num_cells, dtype=np.float,
                                          format='csc')

#    def copy(self):
#        """
#        Create a deep copy of the grid.
#
#        Returns:
#            grid: A deep copy of self. All attributes will also be copied.
#
#        """
#        h = Grid(self.dim, self.nodes.copy(), self.face_nodes.copy(),
#                 self.cell_faces.copy(), self.name)
#        if hasattr(self, 'cell_volumes'):
#            h.cell_volumes = self.cell_volumes.copy()
#        if hasattr(self, 'cell_centers'):
#            h.cell_centers = self.cell_centers.copy()
#        if hasattr(self, 'face_centers'):
#            h.face_centers = self.face_centers.copy()
#        if hasattr(self, 'face_normals'):
#            h.face_normals = self.face_normals.copy()
#        if hasattr(self, 'face_areas'):
#            h.face_areas = self.face_areas.copy()
#        if hasattr(self, 'face_tags'):
#            h.face_tags = self.face_tags.copy()
#        return h
#
    def __repr__(self):
        """
        Implementation of __repr__

        """
        s = 'Mortar grid with history ' + ', '.join(self.name) + '\n' + \
            'Number of cells ' + str(self.num_cells) + '\n' + \
            'Number of nodes ' + str(self.num_nodes) + '\n' + \
            'Dimension ' + str(self.dim) + '\n' + \
            'Face_cells mapping from the higher dimensional grid to the mortar grid\n' + \
            str(self.high_to_mortar) + '\n' + \
            'Cell_cells mapping from the mortar grid to the lower dimensional grid\n' + \
            str(self.mortar_to_low)

        return s

    def __str__(self):
        """ Implementation of __str__
        """
        s = str()

        # Special treatment of point grids.
        if 'PointGrid' in self.name:
            n = self.nodes
            s = 'Point grid.\n' + \
                'Coordinate: (' + str(n[0]) + ', ' + str(n[1]) + \
                ', ' + str(n[2]) + ')\n'
            return s

        # More or less uniform treatment of the types of grids.
        s += 'Number of cells ' + str(self.num_cells) + '\n' + \
             'Number of nodes ' + str(self.num_nodes) + '\n' + \
             'Mapping from the faces of the higher dimensional grid to' + \
             ' the cells of the mortar grid.\nRows indicate the mortar' + \
             ' cell id, columns indicate the (higher dimensional) face id' + \
             '\n' + str(self.high_to_mortar) + '\n' + \
             'Mapping from the cells of the mortar grid to the cells' + \
             ' of the lower dimensional grid.\nRows indicate the mortar' + \
             ' cell id, columns indicate the (lower dimensional) cell id' + \
             '\n' + str(self.mortar_to_low)

        return s

    def update_geometry(self, nodes, cell_nodes):

        self.nodes = np.hstack((nodes, nodes))
        self._cell_nodes = sps.bmat([[cell_nodes, None],
                                     [None, cell_nodes]], format='csc')

        # Infer bookkeeping from size of parameters
        self.num_nodes = self.nodes.shape[1]
        self.num_cells = self._cell_nodes.shape[1]

        self.compute_geometry()

    def refine_mortar(self, matrix):
        matrix = sps.bmat([[matrix, None], [None, matrix]])
        self.mortar_to_low = matrix * self.mortar_to_low
        self.high_to_mortar = matrix * self.high_to_mortar

    def refine_low(self, matrix):
        self.mortar_to_low = sps.bmat([[matrix.T], [matrix.T]])

    def compute_geometry(self):
        """Compute geometric quantities for the grid.

        This method initializes class variables describing the grid
        geometry, see class documentation for details.

        The method could have been called from the constructor, however,
        in cases where the grid is modified after the initial construction (
        say, grid refinement), this may lead to costly, unnecessary
        computations.
        """

        self.name.append('Compute geometry')

        if self.dim == 0:
            self.__compute_geometry_0d()
        elif self.dim == 1:
            self.__compute_geometry_1d()
        else:
            self.__compute_geometry_2d()

    def __compute_geometry_0d(self):
        "Compute 0D geometry"

        self.cell_volumes = np.ones(1)
        self.cell_centers = self.nodes

    def __compute_geometry_1d(self):
        "Compute 1D geometry"

        cn = self._cell_nodes.indices
        x1 = self.nodes[:, cn[::2]]
        x2 = self.nodes[:, cn[1::2]]

        self.cell_volumes = np.linalg.norm(x1 - x2, axis=0)
        self.cell_centers = 0.5 * (x1 + x2)

    def __compute_geometry_2d(self):
        "Compute 2D geometry, with method motivated by similar MRST function"

        TO_FIX
        R = cg.project_plane_matrix(self.nodes, check_planar=False)
        self.nodes = np.dot(R, self.nodes)

        fn = self.face_nodes.indices
        edge1 = fn[::2]
        edge2 = fn[1::2]

        xe1 = self.nodes[:, edge1]
        xe2 = self.nodes[:, edge2]

        edge_length_x = xe2[0] - xe1[0]
        edge_length_y = xe2[1] - xe1[1]
        edge_length_z = xe2[2] - xe1[2]
        self.face_areas = np.sqrt(np.power(edge_length_x, 2) +
                                  np.power(edge_length_y, 2) +
                                  np.power(edge_length_z, 2))
        self.face_centers = 0.5 * (xe1 + xe2)
        n = edge_length_z.shape[0]
        self.face_normals = np.vstack(
            (edge_length_y, -edge_length_x, np.zeros(n)))

        cell_faces, cellno = self.cell_faces.nonzero()
        cx = np.bincount(cellno, weights=self.face_centers[0, cell_faces])
        cy = np.bincount(cellno, weights=self.face_centers[1, cell_faces])
        cz = np.bincount(cellno, weights=self.face_centers[2, cell_faces])
        self.cell_centers = np.vstack((cx, cy, cz)) / np.bincount(cellno)

        a = xe1[:, cell_faces] - self.cell_centers[:, cellno]
        b = xe2[:, cell_faces] - self.cell_centers[:, cellno]

        sub_volumes = 0.5 * np.abs(a[0] * b[1] - a[1] * b[0])
        self.cell_volumes = np.bincount(cellno, weights=sub_volumes)

        sub_centroids = (self.cell_centers[:, cellno] + 2 *
                         self.face_centers[:, cell_faces]) / 3

        ccx = np.bincount(cellno, weights=sub_volumes * sub_centroids[0])
        ccy = np.bincount(cellno, weights=sub_volumes * sub_centroids[1])
        ccz = np.bincount(cellno, weights=sub_volumes * sub_centroids[2])

        self.cell_centers = np.vstack((ccx, ccy, ccz)) / self.cell_volumes

        # Ensure that normal vector direction corresponds with sign convention
        # in self.cellFaces
        def nrm(u):
            return np.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])

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

        self.nodes = np.dot(R.T, self.nodes)
        self.cell_centers = np.dot(R.T, self.cell_centers)

    def cell_nodes(self):
        """
        Obtain mapping between cells and nodes.

        Returns:
            sps.csc_matrix, size num_nodes x num_cells: Value 1 indicates a
                connection between cell and node.

        """
        return self._cell_nodes
