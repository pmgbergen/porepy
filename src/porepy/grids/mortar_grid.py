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

class SideTag(np.uint8, Enum):
    """
    FaceTag contains the following types:
        NONE: None of the below
        LEFT: Left part of the domain
        RIGHT: Right part of the domain
    """
    NONE = 0
    LEFT = 1
    RIGHT = 2
    WHOLE = np.iinfo(type(NONE)).max

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

    def __init__(self, dim, side_grids, face_cells, name=''):
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
        assert np.all([g.dim == dim for g in side_grids.values()])

        self.dim = dim
        self.side_grids = side_grids
        self.sides = np.array(self.side_grids.keys)

        if isinstance(name, list):
            self.name = name
        else:
            self.name = [name]

        num_cells = list(self.side_grids.values())[0].num_cells

        # face_cells mapping from the higher dimensional grid to the mortar grid
        # also here we assume that, in the beginning the mortar grids are equal
        # to the co-dimensional grid
        cells, faces, data = sps.find(face_cells)
        cells[faces > np.median(faces)] += num_cells

        shape = (num_cells*self.num_sides(), face_cells.shape[1])
        self.high_to_mortar = sps.csc_matrix((data.astype(np.float),
                                             (cells, faces)), shape=shape)

        # cell_cells mapping from the mortar grid to the lower dimensional grid
        identity = sps.identity(num_cells)
        self.mortar_to_low = sps.bmat([[identity], [identity]], format='csc')

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

        s+= "".join(['Side '+str(s)+' with grid:\n'+str(g) for s, g in
                                                       self.side_grids.items()])

        s += 'Mapping from the faces of the higher dimensional grid to' + \
             ' the cells of the mortar grid.\nRows indicate the mortar' + \
             ' cell id, columns indicate the (higher dimensional) face id' + \
             '\n' + str(self.high_to_mortar) + '\n' + \
             'Mapping from the cells of the mortar grid to the cells' + \
             ' of the lower dimensional grid.\nRows indicate the mortar' + \
             ' cell id, columns indicate the (lower dimensional) cell id' + \
             '\n' + str(self.mortar_to_low)

        return s

    def compute_geometry(self):
        [g.compute_geometry(is_embedded=True) for g in self.side_grids.values()]

    def refine_mortar(self, side_matrix):
        matrix = np.empty((self.num_sides(), self.num_sides()), dtype=np.object)
        for pos, (side, _) in enumerate(self.side_grids.items()):
            matrix[pos, pos] = side_matrix[side]

        matrix = sps.bmat(matrix)
        self.mortar_to_low = matrix * self.mortar_to_low
        self.high_to_mortar = matrix * self.high_to_mortar


    def refine_low(self, side_matrix):
        matrix = np.empty((self.num_sides(), 1), dtype=np.object)
        for pos, (side, _) in enumerate(self.side_grids.items()):
            matrix[pos, 0] = side_matrix[side]

        self.mortar_to_low = sps.bmat(matrix, format='csc')

    def num_sides(self):
        return len(self.side_grids)
