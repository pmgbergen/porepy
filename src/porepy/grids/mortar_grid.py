""" Module containing the class for the mortar grid.
"""
from __future__ import division
import numpy as np
from enum import Enum
from scipy import sparse as sps


# Module level constants, used to define sides of a mortar grid.
# This is in essence an Enum, but that led to trouble in pickling a GridBucket.
NONE_SIDE = 0
LEFT_SIDE = 1
RIGHT_SIDE = 2
WHOLE_SIDE = np.iinfo(type(NONE_SIDE)).max


class MortarGrid(object):
    """
    Parent class for mortar grids it contains two grids representing the left
    and right part of the mortar grid and the weighted mapping from the higher
    dimensional grid (as set of faces) to the mortar grids and from the lower
    dimensional grid (as set of cells) to the mortar grids. The two mortar grids
    can be different.

    Attributes:

        dim (int): dimension. Should be 0 or 1 or 2.
        side_grids (dictionary of Grid): grid for each side. The key is a
            SideTag and the value is a Grid.
        sides (array of SideTag): ordering of the sides.
        high_to_mortar_int (sps.csc-matrix): Face-cell relationships between the
            high dimensional grid and the mortar grids. Matrix size:
            num_faces x num_cells. In the beginning we assume matching grids,
            but it can be modified by calling refine_mortar(). The matrix
            elements represent the ratio between the geometrical objects.
        low_to_mortar_int (sps.csc-matrix): Cell-cell relationships between the
            mortar grids and the low dimensional grid. Matrix size:
            num_cells x num_cells. Matrix elements represent the ratio between
            the geometrical objects.
        name (list): Information on the formation of the grid, such as the
            constructor, computations of geometry etc.

    """

    # ------------------------------------------------------------------------------#

    def __init__(self, dim, side_grids, face_cells, name=""):
        """Initialize the mortar grid

        See class documentation for further description of parameters.
        The high_to_mortar_int and low_to_mortar_int are identity mapping.

        Parameters
        ----------
        dim (int): grid dimension
        side_grids (dictionary of Grid): grid on each side.
        face_cells (sps.csc_matrix): Cell-face relations between the higher
            dimensional grid and the lower dimensional grid
        name (str): Name of grid
        """

        assert dim >= 0 and dim < 3
        assert np.all([g.dim == dim for g in side_grids.values()])

        self.dim = dim
        self.side_grids = side_grids
        self.sides = np.array(self.side_grids.keys)

        assert self.num_sides() == 1 or self.num_sides() == 2

        if isinstance(name, list):
            self.name = name
        else:
            self.name = [name]

        self.num_cells = np.sum([g.num_cells for g in self.side_grids.values()])
        self.cell_volumes = np.hstack(
            [g.cell_volumes for g in self.side_grids.values()]
        )

        # face_cells mapping from the higher dimensional grid to the mortar grid
        # also here we assume that, in the beginning the mortar grids are equal
        # to the co-dimensional grid. If this assumption is not satisfied we
        # need to change the following lines

        # Creation of the high_to_mortar_int, besically we start from the face_cells
        # map and we split the relation
        # low_dimensional_cell -> 2 high_dimensional_face
        # as
        # low_dimensional_cell -> high_dimensional_face
        # The mapping consider the cell ids of the second mortar grid shifted by
        # the g.num_cells of the first grid. We keep this convention through the
        # implementation. The ordering is given by sides or the keys of
        # side_grids.
        num_cells = list(self.side_grids.values())[0].num_cells
        cells, faces, data = sps.find(face_cells)
        if self.num_sides() == 2:
            cells[faces > np.median(faces)] += num_cells

        shape = (num_cells * self.num_sides(), face_cells.shape[1])
        self.high_to_mortar_int = sps.csc_matrix(
            (data.astype(np.float), (cells, faces)), shape=shape
        )

        # cell_cells mapping from the mortar grid to the lower dimensional grid.
        # It is composed by two identity matrices since we are assuming matching
        # grids here.
        identity = [[sps.identity(num_cells)]] * self.num_sides()
        self.low_to_mortar_int = sps.bmat(identity, format="csc")

    # ------------------------------------------------------------------------------#

    def __repr__(self):
        """
        Implementation of __repr__

        """
        s = (
            "Mortar grid with history "
            + ", ".join(self.name)
            + "\n"
            + "Dimension "
            + str(self.dim)
            + "\n"
            + "Face_cells mapping from the higher dimensional grid to the mortar grid\n"
            + str(self.high_to_mortar_int)
            + "\n"
            + "Cell_cells mapping from the mortar grid to the lower dimensional grid\n"
            + str(self.low_to_mortar_int)
        )

        return s

    # ------------------------------------------------------------------------------#

    def __str__(self):
        """ Implementation of __str__
        """
        s = str()

        s += "".join(
            [
                "Side " + str(s) + " with grid:\n" + str(g)
                for s, g in self.side_grids.items()
            ]
        )

        s += (
            "Mapping from the faces of the higher dimensional grid to"
            + " the cells of the mortar grid.\nRows indicate the mortar"
            + " cell id, columns indicate the (higher dimensional) face id"
            + "\n"
            + str(self.high_to_mortar_int)
            + "\n"
            + "Mapping from the cells of the mortar grid to the cells"
            + " of the lower dimensional grid.\nRows indicate the mortar"
            + " cell id, columns indicate the (lower dimensional) cell id"
            + "\n"
            + str(self.low_to_mortar_int)
        )

        return s

    # ------------------------------------------------------------------------------#

    def compute_geometry(self):
        """
        Compute the geometry of the mortar grids.
        We assume that they are not aligned with x (1d) or x, y (2d).
        """
        [g.compute_geometry() for g in self.side_grids.values()]

    # ------------------------------------------------------------------------------#

    def update_mortar(self, side_matrix):
        """
        Update the low_to_mortar_int and high_to_mortar_int maps when the mortar grids
        are changed.

        Parameter:
            side_matrix (dict): for each SideTag key a matrix representing the
            new mapping between the old and new mortar grids.
        """

        # In the case of different side ordering between the input data and the
        # stored we need to remap it. The resulting matrix will be a block
        # diagonal matrix, where in each block we have the mapping between the
        # (relative to side) old grid and the new one.
        matrix = np.empty((self.num_sides(), self.num_sides()), dtype=np.object)

        # Loop on all the side grids, if not given an identity matrix is
        # considered
        for pos, (side, g) in enumerate(self.side_grids.items()):
            matrix[pos, pos] = side_matrix.get(side, sps.identity(g.num_cells))

        # Once the global matrix is constructed the new low_to_mortar_int and
        # high_to_mortar_int maps are updated.
        matrix = sps.bmat(matrix)
        self.low_to_mortar_int = matrix * self.low_to_mortar_int
        self.high_to_mortar_int = matrix * self.high_to_mortar_int

        self.num_cells = np.sum([g.num_cells for g in self.side_grids.values()])
        self.cell_volumes = np.hstack(
            [g.cell_volumes for g in self.side_grids.values()]
        )
        self._check_mappings()

    # ------------------------------------------------------------------------------#

    def update_low(self, side_matrix):
        """
        Update the low_to_mortar_int map when the lower dimensional grid is changed.

        Parameter:
            side_matrix (dict): for each SideTag key a matrix representing the
            new mapping between the new lower dimensional grid and the mortar
            grids.
        """

        # In the case of different side ordering between the input data and the
        # stored we need to remap it. The resulting matrix will be a block
        # matrix, where in each block we have the mapping between the
        # (relative to side) the new grid and the mortar grid.
        matrix = np.empty((self.num_sides(), 1), dtype=np.object)

        for pos, (side, _) in enumerate(self.side_grids.items()):
            matrix[pos, 0] = side_matrix[side]

        # Update the low_to_mortar_int map. No need to update the high_to_mortar_int.
        self.low_to_mortar_int = sps.bmat(matrix, format="csc")
        self._check_mappings()

    # ------------------------------------------------------------------------------#

    def update_high(self, matrix):
        # Make a comment here
        self.high_to_mortar_int = self.high_to_mortar_int * matrix
        self._check_mappings()

    # ------------------------------------------------------------------------------#

    def num_sides(self):
        """
        Shortcut to compute the number of sides, it has to be 2 or 1.

        Return:
            Number of sides.
        """
        return len(self.side_grids)

    # ------------------------------------------------------------------------------#

    def mortar_to_high_int(self):
        return self.high_to_mortar_avg().T

    # ------------------------------------------------------------------------------#

    def high_to_mortar_avg(self):
        row_sum = self.high_to_mortar_int.sum(axis=1).A.ravel()
        return sps.diags(1. / row_sum) * self.high_to_mortar_int

    # ------------------------------------------------------------------------------#

    def low_to_mortar_avg(self):
        row_sum = self.low_to_mortar_int.sum(axis=1).A.ravel()
        return sps.diags(1. / row_sum) * self.low_to_mortar_int

    # ------------------------------------------------------------------------------#

    def cell_diameters(self):
        diams = np.empty(self.num_sides(), dtype=np.object)
        for pos, (_, g) in enumerate(self.side_grids.items()):
            diams[pos] = g.cell_diameters()
        return np.concatenate(diams).ravel()

    # ------------------------------------------------------------------------------#

    def _check_mappings(self, tol=1e-4):
        row_sum = self.high_to_mortar_int.sum(axis=1)
        assert row_sum.min() > tol
        #        assert row_sum.max() < 1 + tol

        row_sum = self.low_to_mortar_int.sum(axis=1)
        assert row_sum.min() > tol


#        assert row_sum.max() < 1 + tol


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#


class BoundaryMortar(object):
    """
    Parent class for a mortar grid between two grids of the same dimension.  It
    contains a mortar grid and the weighted mapping from the LEFT_SIDE grid (as
    set of faces) to the mortar grid and from the RIGHT_SIDE grid (as set of
    faces) to the mortar grids.

    Attributes:

        dim (int): dimension. Should be 0 or 1 or 2.
        side_grids (dictionary of Grid): grid for each side. The key is a
            SideTag and the value is a Grid.
        sides (array of SideTag): ordering of the sides.
        left_to_mortar_int (sps.csc-matrix): Face-cell relationships between the
            LEFT_SIDE grid and the mortar grid. Matrix size:
            num_faces x num_cells. In the beginning we assume matching grids,
            but it can be modified by calling refine_mortar(). The matrix
            elements represent the ratio between the geometrical objects.
        right_to_mortar_int (sps.csc-matrix): face-cell relationships between
            right mortar grid and the mortar grid. Matrix size:
            num_faces x num_cells. Matrix elements represent the ratio between
            the geometrical objects.
        name (list): Information on the formation of the grid, such as the
            constructor, computations of geometry etc.


    """

    def __init__(self, dim, side_grids, face_faces, name=""):
        """Initialize the mortar grid

        See class documentation for further description of parameters.
        The left_to_mortar_int and right_to_mortar_int are identity mapping.

        Parameters
        ----------
        dim (int): grid dimension
        side_grids (dictionary of Grid): grid on each side.
        face_faces (sps.csc_matrix): face-face relations between the left
            grid and the right dimensional grid
        name (str): Name of grid
        """

        assert dim >= 0 and dim < 3
        assert np.all([g.dim == dim for g in side_grids.values()])

        self.dim = dim
        self.side_grids = side_grids
        self.sides = np.array(self.side_grids.keys)

        assert self.num_sides() == 1 or self.num_sides() == 2

        if isinstance(name, list):
            self.name = name
        else:
            self.name = [name]

        # face_cells mapping from the Left_SIDE grid to the mortar grid
        # also here we assume that, in the beginning the mortar grids are equal
        # to the co-dimensional grid. If this assumption is not satisfied we
        # need to change the following lines
        # The face_faces gives a map from the LEFT_SIDE grid to the RIGHT_SIDE
        # grid. The mortar cells are sorted after the rows of the face_faces
        # mapping.
        right_f, left_f, data = sps.find(face_faces)

        cells = np.argsort(left_f)
        self.num_cells = cells.size
        self.cell_volumes = np.hstack(
            [g.cell_volumes for g in self.side_grids.values()]
        )
        #        self.cell_volumes = side_grids[LEFT_SIDE].face_areas[left_f]
        #        self.cell_volumes = g.face_areas[left_f]

        shape_left = (self.num_cells, face_faces.shape[1])
        shape_right = (self.num_cells, face_faces.shape[0])
        self.left_to_mortar_int = sps.csc_matrix(
            (data.astype(np.float), (cells, left_f)), shape=shape_left
        )
        self.right_to_mortar_int = sps.csc_matrix(
            (data.astype(np.float), (cells, right_f)), shape=shape_right
        )

    def __repr__(self):
        """
        Implementation of __repr__

        """
        s = (
            "Mortar grid with history "
            + ", ".join(self.name)
            + "\n"
            + "Dimension "
            + str(self.dim)
            + "\n"
            + "Face_cell mapping from the LEFT_SIDE grid to the mortar grid\n"
            + str(self.left_to_mortar_int)
            + "\n"
            + "Face_cell mapping from the RIGHT_SIDE grid to the mortar grid\n"
            + str(self.right_to_mortar_int)
        )

        return s

    def __str__(self):
        """
        Implementation of __str__
        """
        s = str()

        s += "".join(
            [
                "Side " + str(s) + " with grid:\n" + str(g)
                for s, g in self.side_grids.items()
            ]
        )

        s += (
            "Mapping from the faces of the left_side grid to"
            + " the cells of the mortar grid. \nRows indicate the mortar"
            + " cell id, columns indicate the left_grid face id"
            + "\n"
            + str(self.left_to_mortar_int)
            + "\n"
            + "Mapping from the cells of the face of the right_side grid"
            + "to the cells of the mortar grid. \nRows indicate the mortar"
            + " cell id, columns indicate the right_grid face id"
            + "\n"
            + str(self.right_to_mortar_int)
        )

        return s

    # ------------------------------------------------------------------------------#

    def num_sides(self):
        """
        Shortcut to compute the number of sides, it has to be 2 or 1.

        Return:
            Number of sides.
        """
        return len(self.side_grids)

    # ------------------------------------------------------------------------------#

    def mortar_to_left_int(self):

        return self.left_to_mortar_avg().T

    # ------------------------------------------------------------------------------#

    def left_to_mortar_avg(self):
        row_sum = self.left_to_mortar_int.sum(axis=1).A.ravel()
        return sps.diags(1. / row_sum) * self.left_to_mortar_int

    # ------------------------------------------------------------------------------#

    def right_to_mortar_avg(self):
        row_sum = self.right_to_mortar_int.sum(axis=1).A.ravel()
        return sps.diags(1. / row_sum) * self.right_to_mortar_int

    # ------------------------------------------------------------------------------#

    def mortar_to_right_int(self):
        return self.right_to_mortar_avg().T

    # ------------------------------------------------------------------------------#
    def compute_geometry(self):
        """
        Compute the geometry of the mortar grids.
        We assume that they are not aligned with x (1d) or x, y (2d).
        """
        [g.compute_geometry() for g in self.side_grids.values()]
