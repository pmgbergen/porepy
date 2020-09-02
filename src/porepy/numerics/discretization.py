""" Module contains two classes:
    1) The abstract superclass for all discretizations
    2) A do-nothing discretization

"""
import abc
import numpy as np
import scipy.sparse as sps
from typing import Dict, Union

import porepy as pp


class Discretization(abc.ABC):
    """ Interface for all discretizations. Specifies methods that must be implemented
    for a discretization class to be compatible with the assembler.

    """

    def __init__(self, keyword):
        self.keyword = keyword

    @abc.abstractmethod
    def discretize(self, g: pp.Grid, data: Dict) -> None:
        """ Construct discretization matrices.

        The discretization matrices should be added to
            data[pp.DISCRETIZATION_MATRICES][self.keyword]

        Parameters:
            g (pp.Grid): Grid to be discretized.
            data (dictionary): With discretization parameters.

        """
        pass

    def update_discretization(self, g: pp.Grid, data: Dict) -> None:
        """ Partial update of discretization.

        Intended use is when the discretization should be updated, e.g. because of
        changes in parameters, grid geometry or grid topology, and it is not
        desirable to recompute the discretization on the entire grid. A typical case
        will be when the discretization operation is costly, and only a minor update
        is necessary.

        The updates can generally come as a combination of two forms:
            1) The discretization on part of the grid should be recomputed.
            2) The old discretization can be used (in parts of the grid), but the
               numbering of unknowns has changed, and the discretization should be
               reorder accordingly.

        By default, this method will simply forward the call to the standard
        discretize method. Discretization methods that wants a tailored approach
        should override the standard implementation.

        Information on the basis for the update should be stored in a field

            data['update_discretization']

        this should be a dictionary with up to six keys:

            modified_cells, modified_faces, modified_nodes

        define cells, faces and nodes that have been modified (either parameters,
        geometry or topology), and should be rediscretized. It is up to the
        discretization method to implement the change necessary by this modification.
        Note that depending on the computational stencil of the discretization method,
        a grid quantity may be rediscretized even if it is not marked as modified.
        The dictionary should further have keys:

            map_cells, map_faces, map_nodes

        these should specify 2 x n np.ndarrays, that maps old (first row) to new
        (second row) indices of unchanged grid quantities. The index of all cells,
        faces and nodes should be found in *exactly one* of the respective modified
        and map fields.

        It is up to the caller to specify which parts of the grid to recompute, and
        how to update the numbering of degrees of freedom. If the discretization
        method does not provide a tailored implementation for update, it is not
        necessary to provide this information.

        Parameters:
            g (pp.Grid): Grid to be rediscretized.
            data (dictionary): With discretization parameters.

        """
        # Default behavior is to discretize everything
        self.discretize(g, data)

    @abc.abstractmethod
    def assemble_matrix_rhs(
        self, g: pp.Grid, data: Dict
    ) -> Union[sps.spmatrix, np.ndarray]:
        """ Assemble discretization matrix and rhs vector.

        Parameters:
            g (pp.Grid): Grid to be discretized.
            data (dictionary): With discretization parameters.

        Returns:
            sps.csc_matrix: Discretization matrix.
            np.array: Right hand side term.

        """
        pass


class VoidDiscretization(Discretization):
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
