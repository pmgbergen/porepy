""" Module contains the abstract superclass for all discretizations."""

import abc
from typing import Dict, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp


class Discretization(abc.ABC):
    """Interface for all discretizations. Specifies methods that must be implemented
    for a discretization class to be compatible with the assembler.

    """

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

    def __repr__(self) -> str:
        s = f"Discretization of type {self.__class__.__name__}"
        if hasattr(self, "keyword"):
            s += f" with keyword {self.keyword}"
        return s

    @abc.abstractmethod
    def ndof(self, g: pp.Grid) -> int:
        """
        Return the number of degrees of freedom associated to the method.

        Parameters:
            g: grid, or a subclass.

        Return:
            dof: the number of degrees of freedom.

        """

    @abc.abstractmethod
    def discretize(self, g: pp.Grid, data: Dict) -> None:
        """Construct discretization matrices.

        The discretization matrices should be added to
            data[pp.DISCRETIZATION_MATRICES][self.keyword]

        Parameters:
            g (pp.Grid): Grid to be discretized.
            data (dictionary): With discretization parameters.

        """
        pass

    def update_discretization(self, g: pp.Grid, data: Dict) -> None:
        """Partial update of discretization.

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

        this should be a dictionary with up to six keys. The following optional keys:

            modified_cells, modified_faces, modified_nodes

        define cells, faces and nodes that have been modified (either parameters,
        geometry or topology), and should be rediscretized. It is up to the
        discretization method to implement the change necessary by this modification.
        Note that depending on the computational stencil of the discretization method,
        a grid quantity may be rediscretized even if it is not marked as modified.
        The dictionary could further have keys:

            cell_index_map, face_index_map, node_index_map

        these should specify sparse matrices that maps old to new indices. If not provided,
        unit mappings should be assumed, that is, no changes to the grid topology are
        accounted for.

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
        """Assemble discretization matrix and rhs vector.

        Parameters:
            g (pp.Grid): Grid to be discretized.
            data (dictionary): With discretization parameters.

        Returns:
            sps.csc_matrix: Discretization matrix.
            np.ndarray: Right hand side term.

        """
        pass


class InterfaceDiscretization(abc.ABC):
    """Superclass for all interface discretizations"""

    @abc.abstractmethod
    def discretize(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        data_primary: Dict,
        data_secondary: Dict,
        data_intf: Dict,
    ) -> None:
        """Discretize the interface law and store the discretization in the
        interface data.

        The discretization matrix will be stored in the data dictionary of this
        interface.

        Parameters:
            sd_primary: Grid of the primary domanin.
            sd_secondary: Grid of the secondary domain.
            data_primary: Data dictionary for the primary domain.
            data_secondary: Data dictionary for the secondary domain.
            data_intf: Data dictionary for the interface between the domains.

        """
        pass
