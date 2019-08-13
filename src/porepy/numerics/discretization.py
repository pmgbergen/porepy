#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module with a do-nothing discretization class.


"""

import numpy as np
import scipy.sparse as sps


class VoidDiscretization:
    """ Do-nothing discretization object. Used if a discretizaiton object
    is needed for technical reasons, but not really necessary.

    Attributes:
        keyword (str): Keyword used to identify parameters and discretization
            matrices for this object.
        ndof_cell (int): Number of degrees of freedom per cell in a grid.
        ndof_face (int): Number of degrees of freedom per face in a grid.
        ndof_node (int): Number of degrees of freedom per node in a grid.

    """

    def __init__(self, keyword, ndof_cell=0, ndof_face=0, ndof_node=0):
        """ Set the discretization, with the keyword used for storing various
        information associated with the discretization.

        Paramemeters:
            keyword (str): Identifier of all information used for this
                discretization.
            ndof_cell (int, optional): Number of degrees of freedom per cell
                in a grid. Defaults to 0.
            ndof_face (int, optional): Number of degrees of freedom per face
                in a grid. Defaults to 0.
            ndof_node (int, optional): Number of degrees of freedom per node
                in a grid. Defaults to 0.

        """
        self.keyword = keyword
        self.ndof_cell = ndof_cell
        self.ndof_face = ndof_face
        self.ndof_node = ndof_node

    def _key(self):
        """ Get the keyword of this object, on a format friendly to access relevant
        fields in the data dictionary

        Returns:
            String, on the form self.keyword + '_'.

        """
        return self.keyword + "_"

    def ndof(self, g):
        """ Abstract method. Return the number of degrees of freedom associated to the
        method.

        Parameters
            g (grid): Computational grid

        Returns:
            int: the number of degrees of freedom.

        """
        return (
            g.num_cells * self.ndof_cell
            + g.num_faces * self.ndof_face
            + g.num_nodes * self.ndof_node
        )

    def discretize(self, g, data):
        """ Construct discretization matrices. Operation is void for this discretization.

        Parameters:
            g (pp.Grid): Grid to be discretized.
            data (dictionary): With discretization parameters.

        """
        pass

    def assemble_matrix_rhs(self, g, data):
        """ Assemble discretization matrix and rhs vector, both empty.

        Parameters:
            g (pp.Grid): Grid to be discretized.
            data (dictionary): With discretization parameters.

        Returns:
            sps.csc_matrix: Of specified dimensions relative to the grid. Empty.
            np.array: Of specified dimensions relative to the grid. All zeros.

        """
        ndof = self.ndof(g)

        return sps.csc_matrix((ndof, ndof)), np.zeros(ndof)
